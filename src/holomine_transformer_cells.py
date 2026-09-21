# =============================================================================
# HoloMine Property Price Prediction (Task 2) — DeBERTa Fine-tune Regression
# =============================================================================
# Format: satu file, pemisah antar cell = "# %% CELL N — ...".
# Copy-paste tiap blok ke satu cell notebook Kaggle, jalankan BERURUTAN.
#
# Setup Kaggle (WAJIB sebelum Cell 1):
#   - Settings -> Accelerator = GPU (T4/P100)
#   - Settings -> Internet = ON (perlu download model HuggingFace)
#   - Upload train.csv / test.csv / sample_submission.csv sebagai dataset,
#     attach dataset ke notebook ini.
#
# Protokol (dari master_prompt_kaggle.md):
#   - Tahap 3 (validasi) selesai di CELL 2 -> fold DIBEKUKAN, eksperimen
#     berikutnya WAJIB pakai fold yang sama.
#   - Semua transformasi fit hanya di train-fold; tidak ada fit di test.
#   - OOF & test predictions disimpan sebagai artefak (bahan final blend).
#   - Checkpoint eksplisit per cell: angka tercetak, bukan "sepertinya ok".
# =============================================================================


# %% CELL 1 — TAHAP 0/1: SETUP, AUDIT, FAST BASELINE
# Disuruh apa: set seed, load 3 file CSV, audit struktur & leakage, lihat
# distribusi target + panjang teks, lalu buat baseline median untuk di-SUBMIT.
# Tujuan: pahami struktur data dengan benar + punya submission valid secepatnya.

import os
import glob
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


set_seed()

try:
    import torch
    print("GPU available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device:", torch.cuda.get_device_name(0))
except ImportError:
    print("torch belum ter-install (ok untuk cell ini, WAJIB untuk CELL 3+)")

# --- 1.1 Load data: JANGAN hardcode path — list dulu isi /kaggle/input ---
BASE = "/kaggle/input"
found = []
for root, _, files in os.walk(BASE):
    for f in files:
        found.append(os.path.join(root, f))
print("\nFile yang ter-attach:")
for p in found:
    print(" ", p)


def find_csv(name):
    hits = [p for p in found if os.path.basename(p).lower() == name]
    assert hits, f"{name} tidak ditemukan di {BASE} — cek attach dataset"
    return hits[0]


train = pd.read_csv(find_csv("train.csv"))
test = pd.read_csv(find_csv("test.csv"))
sub = pd.read_csv(find_csv("sample_submission.csv"))

# --- 1.2 Audit struktur & leakage ---
print("\n=== AUDIT STRUKTUR ===")
for name, df in [("train", train), ("test", test), ("sample_submission", sub)]:
    print(f"{name}: shape={df.shape}, kolom={list(df.columns)}")
    print(f"  null/kolom : {df.isnull().sum().to_dict()}")
    print(f"  duplikat id: {df['id'].duplicated().sum()}")

overlap = set(train["id"]) & set(test["id"])
print(f"\nOverlap id train n test: {len(overlap)} (harus 0)")
assert len(overlap) == 0, "ID bocor antara train dan test!"

# duplikat baris persis (teks sama di train & test = potensi leak)
dup_texts = set(train["text"]) & set(test["text"])
print(f"Teks identik train n test: {len(dup_texts)}")

train["text"] = train["text"].fillna("")
test["text"] = test["text"].fillna("")

# --- 1.3 Distribusi target ---
y = train["listPrice"]
print("\n=== listPrice (target) ===")
print(y.describe())
print(f"skewness: {y.skew():.2f}  (>1 = right-skewed, log1p dibenarkan)")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(y, bins=60, color="steelblue")
ax[0].set_title("listPrice (asli)")
ax[1].hist(np.log1p(y), bins=60, color="darkorange")
ax[1].set_title("log1p(listPrice)")
plt.tight_layout()
plt.show()

# --- 1.4 Panjang teks (dasar keputusan max_length=512) ---
train["n_chars"] = train["text"].str.len()
test["n_chars"] = test["text"].str.len()
print("\n=== Panjang teks (karakter) ===")
print("train:", train["n_chars"].describe()[["mean", "50%", "max"]].to_dict())
print("test :", test["n_chars"].describe()[["mean", "50%", "max"]].to_dict())
# estimasi kasar token: ~4 karakter/token untuk teks Inggris
est_tok_train = train["n_chars"] / 4
est_tok_test = test["n_chars"] / 4
print(f"estimasi >512 token: train {100 * (est_tok_train > 512).mean():.1f}%, "
      f"test {100 * (est_tok_test > 512).mean():.1f}% "
      "(cek eksak di CELL 3 dengan tokenizer)")
train = train.drop(columns="n_chars")
test = test.drop(columns="n_chars")

# --- 1.5 Format sample_submission ---
print("\n=== sample_submission ===")
print(f"kolom={list(sub.columns)}, baris={len(sub)}")
same_order = (test["id"].values == sub["id"].values).all()
print(f"urutan id test == urutan id submission: {same_order}")

# --- 1.6 FAST BASELINE: median -> SUBMIT INI SEKARANG ---
baseline_pred = float(np.median(y))
sub_baseline = sub.copy()
sub_baseline["listPrice"] = baseline_pred
sub_baseline.to_csv("submission_baseline_median.csv", index=False)
print(f"\nBaseline median = {baseline_pred:,.0f}")
print("-> submission_baseline_median.csv dibuat. DOWNLOAD & SUBMIT SEKARANG")
print("   (validasi format end-to-end + skor LB titik nol)")

# CHECKPOINT CELL 1: tabel audit + overlap=0 + baseline tersimpan.


# %% CELL 2 — TAHAP 2/3: EDA RINGKAS + VALIDATION STRATEGY (FOLD DIBEKUKAN)
# Disuruh apa: stratifikasi harga ke 10 bin, buat StratifiedKFold(5), simpan
# assignment fold, dan hitung CV-MAE baseline (titik nol CV).
# Tujuan: satu set validasi yang dipakai SEMUA eksperimen berikutnya.

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error

train["price_bin"] = pd.qcut(
    np.log1p(train["listPrice"]), q=10, labels=False, duplicates="drop"
)
print("jumlah baris per bin harga:")
print(train["price_bin"].value_counts().sort_index().to_dict())

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
train["fold"] = -1
for fold, (_, val_idx) in enumerate(skf.split(train, train["price_bin"])):
    train.loc[train.index[val_idx], "fold"] = fold

assert (train["fold"] >= 0).all()
assert train["fold"].value_counts().min() > 0

train[["id", "fold"]].to_csv("folds.csv", index=False)
print("\nfolds.csv TERSIMPAN — DIBEKUKAN. Semua eksperimen berikutnya")
print("WAJIB pakai fold ini (jangan ganti seed/jumlah fold).")

# sanity: distribusi harga per fold harus mirip
print("\n=== Sanity distribusi harga per fold ===")
print(train.groupby("fold")["listPrice"].agg(["count", "median", "mean"]))

# baseline CV: prediksi median train-fold, MAE di skala ASLI
maes = []
for fold in range(5):
    tr = train[train["fold"] != fold]
    va = train[train["fold"] == fold]
    pred = np.full(len(va), np.median(tr["listPrice"]))
    maes.append(mean_absolute_error(va["listPrice"], pred))
print(f"\nCV-MAE baseline median: {np.mean(maes):,.0f} +/- {np.std(maes):,.0f}")
print("(angka ini titik nol CV — model CELL 3 harus jauh di bawah ini)")

# CHECKPOINT CELL 2: folds.csv ada + CV-MAE baseline tercatat.


# %% CELL 3 — TAHAP 5: TRAINING DeBERTa PER FOLD (SMOKE TEST DULU!)
# Disuruh apa: fine-tune deberta-v3-base + regression head di log1p(price),
# 5 fold, simpan OOF + test prediction per fold.
# Tujuan: model kuat pertama, semua diukur di fold yang sama (folds.csv).
#
# PROTOkOL: jalankan SEKALI dengan SMOKE=True (100 step, model small) dulu
# untuk memvalidasi struktur kode & runtime. Kalau lolos, set SMOKE=False
# dan jalankan ulang dari cell ini untuk full run.

import os
import time
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from sklearn.metrics import mean_absolute_error

# ----------------------------- KONFIG ---------------------------------------
SMOKE = False          # True = validasi struktur (100 step, model small, fold 0 saja)
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LEN = 512
BS = 8                 # batch size per step
GRAD_ACCUM = 2         # effective batch = 16
EPOCHS = 3
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_FRAC = 0.1
LOSS_NAME = "l1"        # "l1" (selaras MAE) | "mse" (eksperimen pembanding di CELL 4)
RUN_TAG = f"{LOSS_NAME}_v2"  # v2 = mean pooling + target centering (jangan campur artefak v1)
USE_AMP = True          # False = fp32 murni (fallback, ~2x lebih lambat tapi aman)

if SMOKE:
    MODEL_NAME = "microsoft/deberta-v3-small"
    MAX_STEPS = 100
    FOLDS_TO_RUN = [0]
else:
    MAX_STEPS = None
    FOLDS_TO_RUN = [0, 1, 2, 3, 4]

device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda", "GPU tidak terdeteksi — cek Settings -> Accelerator"


def set_seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# compat AMP untuk berbagai versi torch
if USE_AMP:
    try:
        SCALER = torch.amp.GradScaler("cuda", enabled=True)

        def autocast_ctx():
            return torch.amp.autocast("cuda", dtype=torch.float16)
    except (AttributeError, TypeError):
        SCALER = torch.cuda.amp.GradScaler()

        def autocast_ctx():
            return torch.cuda.amp.autocast()
else:
    SCALER = torch.amp.GradScaler("cuda", enabled=False) if hasattr(torch, "amp") \
        else torch.cuda.amp.GradScaler(enabled=False)

    class _NoCtx:
        def __enter__(self):
            return None

        def __exit__(self, *a):
            return False

    def autocast_ctx():
        return _NoCtx()


def fix_fp16_grads(model_):
    """Safety net AMP: kalau ada gradient fp16 (mis. param termuat sebagai fp16
    oleh transformers 5.x), cast ke fp32 sebelum scaler.step()."""
    for p in model_.parameters():
        if p.grad is not None and p.grad.dtype == torch.float16:
            try:
                p.grad_dtype = None  # torch>=2.14: izinkan assign lintas dtype
            except (AttributeError, RuntimeError):
                pass
            p.grad = p.grad.float()


# --------------------------- DATA PIPELINE ----------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# cek eksak panjang token (validasi keputusan MAX_LEN=512)
_tok_sample = train["text"].sample(min(1000, len(train)), random_state=SEED).tolist()
_tok_lens = [len(x) for x in tokenizer(_tok_sample, truncation=True, max_length=10_000)["input_ids"]]
print(f"token length sampel 1000 baris: median={int(np.median(_tok_lens))}, "
      f"p90={int(np.percentile(_tok_lens, 90))}, "
      f"% >512: {100 * np.mean(np.array(_tok_lens) > 512):.1f}%")


class ListingDataset(Dataset):
    """Tokenisasi sekali di __init__ (tanpa padding); padding dinamis di collate."""

    def __init__(self, texts, targets=None):
        self.enc = tokenizer(list(texts), truncation=True, max_length=MAX_LEN)
        self.targets = targets

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        if self.targets is not None:
            item["labels"] = torch.tensor(float(self.targets[i]))
        return item


def collate(batch):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, attn, labels = [], [], []
    for b in batch:
        ids = b["input_ids"]
        pad = maxlen - len(ids)
        input_ids.append(torch.cat([ids, torch.zeros(pad, dtype=torch.long)]))
        attn.append(torch.cat([torch.ones(len(ids), dtype=torch.long),
                               torch.zeros(pad, dtype=torch.long)]))
        labels.append(b["labels"] if "labels" in b else torch.tensor(0.0))
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attn),
        "labels": torch.stack(labels),
    }


class PriceRegressor(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        # FIX utama: transformers 5.x memuat bobot sebagai fp16 (dtype="auto")
        # -> param fp16 -> grad fp16 -> GradScaler error
        # ("Attempting to unscale FP16 gradients"). PAKSA fp32; autocast fp16
        # tetap menangani mixed precision di forward secara normal.
        try:
            self.backbone = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        except TypeError:  # transformers 4.x pakai nama arg lama
            self.backbone = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        # assert: semua param HARUS fp32 (kalau tidak, optimizer jadi tidak stabil)
        for p in self.backbone.parameters():
            assert p.dtype == torch.float32, \
                f"param termuat {p.dtype}, harusnya fp32 — cek versi transformers"
        hidden = self.backbone.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hs = out.last_hidden_state                    # (B, L, H)
        mask = attention_mask.unsqueeze(-1).to(hs.dtype)
        # MEAN POOLING (bukan CLS): [CLS] DeBERTa-v3 tidak dilatih sebagai
        # sentence representation (model MLM) -> CLS pooling menghasilkan
        # fitur lemah (terbukti: run v1 kalah dari baseline median).
        pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return self.head(self.drop(pooled)).squeeze(-1)


@torch.no_grad()
def predict_texts(model, texts, bs=32):
    model.eval()
    ds = ListingDataset(texts)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=collate)
    preds = []
    for batch in dl:
        with autocast_ctx():
            logits = model(batch["input_ids"].to(device),
                           batch["attention_mask"].to(device))
        preds.append(logits.float().cpu().numpy())
    return np.concatenate(preds)


def train_one_fold(train_df, test_df, fold):
    set_seed_all(SEED + fold)
    tr = train_df[train_df["fold"] != fold]
    va = train_df[train_df["fold"] == fold]

    y_tr = np.log1p(tr["listPrice"].values)
    y_va = va["listPrice"].values  # skala ASLI untuk evaluasi
    # TARGET CENTERING: log-price ~11.5-18.5; tanpa centering, head mulai dari
    # prediksi ~0 vs target ~13 -> gradien awal besar, konvergensi lambat.
    # Center dihitung HANYA dari train-fold (anti-leakage), dikembalikan saat prediksi.
    y_center = float(y_tr.mean())

    ds_tr = ListingDataset(tr["text"].tolist(), y_tr - y_center)
    dl_tr = DataLoader(ds_tr, batch_size=BS, shuffle=True, collate_fn=collate,
                       num_workers=2, drop_last=True)

    model = PriceRegressor(MODEL_NAME).to(device)
    criterion = nn.L1Loss() if LOSS_NAME == "l1" else nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    steps_per_epoch = len(dl_tr) // GRAD_ACCUM
    total_steps = steps_per_epoch * EPOCHS
    sched = get_cosine_schedule_with_warmup(
        opt, int(WARMUP_FRAC * total_steps), total_steps
    )

    t0 = time.time()
    global_step, running = 0, []
    model.train()
    for epoch in range(EPOCHS):
        for step, batch in enumerate(dl_tr):
            with autocast_ctx():
                logits = model(batch["input_ids"].to(device),
                               batch["attention_mask"].to(device))
                loss = criterion(logits, batch["labels"].to(device))
            SCALER.scale(loss / GRAD_ACCUM).backward()
            running.append(loss.item())
            if (step + 1) % GRAD_ACCUM == 0:
                fix_fp16_grads(model)  # DeBERTa-v3 + AMP: grad fp16 -> fp32
                SCALER.step(opt)
                SCALER.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                global_step += 1
                if global_step % 100 == 0:
                    print(f"  fold {fold} ep{epoch} step {global_step}/{total_steps} "
                          f"loss={np.mean(running[-100:]):.4f}")
                if MAX_STEPS and global_step >= MAX_STEPS:
                    break
        if MAX_STEPS and global_step >= MAX_STEPS:
            break

    train_min = (time.time() - t0) / 60
    print(f"  fold {fold}: training selesai dalam {train_min:.1f} menit")

    # prediksi valid (skala log -> expm1 -> MAE skala asli)
    va_pred_log = predict_texts(model, va["text"].tolist()) + y_center
    va_pred = np.expm1(np.clip(va_pred_log, 0, np.log1p(train_df["listPrice"].max())))
    fold_mae = mean_absolute_error(y_va, va_pred)
    print(f"  fold {fold}: MAE = {fold_mae:,.0f}")

    # prediksi test (fold-mean di CELL 5)
    te_pred_log = predict_texts(model, test_df["text"].tolist()) + y_center
    te_pred = np.expm1(np.clip(te_pred_log, 0, np.log1p(train_df["listPrice"].max())))

    oof_part = pd.DataFrame({
        "id": va["id"].values,
        "fold": fold,
        "listPrice": y_va,
        "oof_pred": va_pred,
    })
    te_part = pd.DataFrame({"id": test_df["id"].values, f"test_pred_f{fold}": te_pred})
    if not SMOKE:
        te_part.to_csv(f"test_pred_{RUN_TAG}_fold{fold}.csv", index=False)

    del model
    torch.cuda.empty_cache()
    return oof_part, fold_mae, te_part


# ------------------------------ RUN -----------------------------------------
if not os.path.exists("folds.csv"):
    raise RuntimeError("folds.csv belum ada — jalankan CELL 2 dulu!")
folds_df = pd.read_csv("folds.csv")
# guard: kalau session Kaggle restart, reload train/test dari CSV
if "train" not in dir() or "test" not in dir():
    def _walk_find(name, _base="/kaggle/input"):
        for root, _, files in os.walk(_base):
            if name in files:
                return os.path.join(root, name)
        raise FileNotFoundError(name)
    train = pd.read_csv(_walk_find("train.csv"))
    test = pd.read_csv(_walk_find("test.csv"))
    for _df in (train, test):
        _df["text"] = _df["text"].fillna("")
# buang kolom fold/price_bin lama (kalau ada dari CELL 2) supaya merge tidak duplikat
train = train.drop(columns=[c for c in ("fold", "price_bin") if c in train.columns])
train = train.merge(folds_df, on="id", how="left")
assert train["fold"].notna().all(), "ada baris train tanpa fold!"

oof_parts, maes, te_parts = [], [], []
for fold in FOLDS_TO_RUN:
    print(f"\n===== FOLD {fold} ({MODEL_NAME}, loss={LOSS_NAME}, smoke={SMOKE}) =====")
    oof_part, fold_mae, te_part = train_one_fold(train, test, fold)
    oof_parts.append(oof_part)
    maes.append(fold_mae)
    te_parts.append(te_part)

oof = pd.concat(oof_parts, ignore_index=True)
print("\n=== HASIL CV ===")
for fold, m in zip(FOLDS_TO_RUN, maes):
    print(f"fold {fold}: MAE = {m:,.0f}")
print(f"CV-MAE: {np.mean(maes):,.0f} +/- {np.std(maes):,.0f}  (run={RUN_TAG}, smoke={SMOKE})")

if SMOKE:
    print("\nSMOKE TEST lolos: pipeline jalan end-to-end tanpa crash/NaN.")
    print("-> set SMOKE=False lalu Run All dari CELL 3 untuk full run.")
    print("   Contoh prediksi valid (skala asli):", oof["oof_pred"].head(5).round(0).tolist())
    assert oof["oof_pred"].notna().all() and np.isfinite(oof["oof_pred"]).all()
else:
    oof.to_csv(f"oof_pred_{RUN_TAG}.csv", index=False)
    print(f"-> oof_pred_{RUN_TAG}.csv + test_pred_{RUN_TAG}_fold*.csv tersimpan")

# CHECKPOINT CELL 3: CV-MAE model << CV-MAE baseline median (CELL 2),
# tidak ada NaN, runtime per fold tercatat.
# CATATAN PENTING: run v1 (CLS pooling) kalah dari baseline median (642k vs
# 550k) -> diagnosa: CLS DeBERTa-v3 lemah. v2 = mean pooling + centering.
# Kalau v2 MASIH kalah dari Ridge (CELL 3B, ~351k), berarti masalahnya bukan
# pooling — laporkan angkanya untuk analisis lanjutan.


# %% CELL 3B — KANDIDAT MURAH: TF-IDF + RIDGE (CPU, ~2 menit)
# Disuruh apa: latih Ridge di log1p(price) dengan TF-IDF, fold yang sama,
# simpan OOF + test pred, dan langsung buat submission cadangan.
# Tujuan: (a) sanity check sinyal teks, (b) submission aman yang PASTI
# mengalahkan baseline, (c) bahan blend dengan transformer di CELL 5.
# Terbukti di lokal: Ridge alpha=0.5 -> CV-MAE ~347k (vs baseline 550k).

import os as _os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

if not os.path.exists("folds.csv"):
    raise RuntimeError("folds.csv belum ada — jalankan CELL 2 dulu!")
folds_df = pd.read_csv("folds.csv")


def _walk_find(name, _base="/kaggle/input"):
    for root, _, files in _os.walk(_base):
        if name in files:
            return _os.path.join(root, name)
    raise FileNotFoundError(name)


_tr = pd.read_csv(_walk_find("train.csv"))
_te = pd.read_csv(_walk_find("test.csv"))
for _df in (_tr, _te):
    _df["text"] = _df["text"].fillna("")
_tr = _tr.merge(folds_df, on="id", how="left")
assert _tr["fold"].notna().all()

oof_ridge, te_pred_sum, fold_maes = [], np.zeros(len(_te)), []
for fold in range(5):
    trf = _tr[_tr["fold"] != fold]
    vaf = _tr[_tr["fold"] == fold]
    vec = TfidfVectorizer(max_features=100_000, ngram_range=(1, 2),
                          sublinear_tf=True, min_df=2)
    X_tr = vec.fit_transform(trf["text"])          # fit HANYA di train-fold
    X_va = vec.transform(vaf["text"])
    X_te = vec.transform(_te["text"])
    model = Ridge(alpha=0.5)  # tuning lokal: 0.5 (347k) < 1.0 (351k) < 2.0 (363k)
    model.fit(X_tr, np.log1p(trf["listPrice"].values))
    va_pred = np.expm1(model.predict(X_va))
    te_pred_sum += np.expm1(model.predict(X_te))
    fold_maes.append(mean_absolute_error(vaf["listPrice"], va_pred))
    oof_ridge.append(pd.DataFrame({"id": vaf["id"].values, "fold": fold,
                                   "listPrice": vaf["listPrice"].values,
                                   "oof_pred": va_pred}))
    print(f"ridge fold {fold}: MAE = {fold_maes[-1]:,.0f}")

print(f"\nRIDGE CV-MAE: {np.mean(fold_maes):,.0f} +/- {np.std(fold_maes):,.0f}")
oof_r = pd.concat(oof_ridge, ignore_index=True)
oof_r.to_csv("oof_pred_ridge.csv", index=False)
te_r = pd.DataFrame({"id": _te["id"].values, "test_pred": te_pred_sum / 5})
te_r.to_csv("test_pred_ridge.csv", index=False)

# submission cadangan Ridge — BISA LANGSUNG DI-SUBMIT sebagai safety net
sub_r = pd.read_csv(_walk_find("sample_submission.csv"))
price_min, price_max = _tr["listPrice"].min(), _tr["listPrice"].max()
sub_ridge = sub_r[["id"]].merge(
    te_r.assign(listPrice=te_r["test_pred"].clip(price_min, price_max))[["id", "listPrice"]],
    on="id", how="left")
assert sub_ridge["listPrice"].notna().all()
assert (sub_ridge["id"].values == sub_r["id"].values).all()
sub_ridge.to_csv("submission_ridge.csv", index=False)
print("-> oof_pred_ridge.csv, test_pred_ridge.csv, submission_ridge.csv tersimpan")
print("   (submission_ridge.csv layak submit sekarang sebagai kandidat aman)")

# CHECKPOINT CELL 3B: Ridge CV-MAE ~347k tercetak; 3 artefak tersimpan.


# %% CELL 3C — RESCUE (HANYA kalau CELL 3 di-interrupt di tengah loop)
# Disuruh apa: kalau CELL 3 dihentikan (mis. setelah fold 2 karena waktu),
# state in-memory (oof_parts) masih ada di session. Cell ini menyimpan
# OOF dari fold yang SUDAH selesai, supaya CELL 5 tetap bisa blend.
# Jalankan HANYA setelah interrupt — kalau CELL 3 selesai normal, SKIP.
# CATATAN: test_pred_{RUN_TAG}_fold*.csv sudah tersimpan otomatis per fold.

try:
    if oof_parts:  # ada state dari CELL 3 yang di-interrupt
        oof_partial = pd.concat(oof_parts, ignore_index=True)
        oof_partial.to_csv(f"oof_pred_{RUN_TAG}.csv", index=False)
        print(f"RESCUE: oof_pred_{RUN_TAG}.csv tersimpan dari {len(oof_parts)} fold:")
        print(f"  fold selesai: {FOLDS_TO_RUN[:len(oof_parts)]}, MAE: "
              f"{[round(m) for m in maes]}")
        print("  CATATAN: OOF hanya mencakup fold yang selesai -> CV-MAE blend "
              "sedikit bias (evaluasi di subset), tapi tetap leak-free.")
except NameError:
    print("Tidak ada state CELL 3 di memory — jalankan cell ini hanya "
          "segera setelah interrupt CELL 3, ATAU skip jika CELL 3 selesai normal.")


# %% CELL 3D — KANDIDAT 3: REGEX FEATURE ENGINEERING + LIGHTGBM (CPU, ~15 menit)
# Disuruh apa: ekstrak fitur domain eksplisit dari teks (beds, baths, sqft,
# acre, tipe properti, kondisi, umur) -> LightGBM, fold yang sama.
# Tujuan: diversitas INFORMASI (Tahap 5.1 master prompt) — fitur ini berbeda
# sifat dari TF-IDF (n-gram) maupun transformer (semantik): angka & kategori
# eksplisit yang relevan harga properti. Bahan blend ke-3 di CELL 5.
# Jalankan DI CPU sementara GPU sibuk training CELL 3.

import os as _os
import re
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("lightgbm tidak ter-install: !pip install lightgbm -q, lalu rerun")

if not _os.path.exists("folds.csv"):
    raise RuntimeError("folds.csv belum ada — jalankan CELL 2 dulu!")
folds_df = pd.read_csv("folds.csv")


def _walk_find(name, _base="/kaggle/input"):
    for root, _, files in _os.walk(_base):
        if name in files:
            return _os.path.join(root, name)
    raise FileNotFoundError(name)


NUM = r"(\d+(?:[\.,]\d+)?)"

PATTERNS = {
    # angka eksplisit yang menentukan harga
    "n_bed": rf"{NUM}\s*(?:-\s*bed(?:room)?s?\b|\bbr\b|\bbed\b)",
    "n_bath": rf"{NUM}\s*(?:-\s*bath(?:room)?s?\b|\bba\b)",
    "sqft": rf"{NUM}\s*(?:[,]\s*)?(?:\+/?\s*)?(?:sq\.?\s?ft|sqft|square\s+feet|sf\b)",
    "acre": rf"{NUM}\s*(?:/?\s*acre|acres)\b",
    "lot_sqft": rf"lot[^.]*?{NUM}\s*(?:sq\.?\s?ft|sqft|sf\b)",
    "hoa_fee": rf"\$\s*{NUM}\s*(?:/|per\s*)?\s*(?:month|mo)\b.*?hoa|hoa[^.]*?\$\s*{NUM}",
    "year_built": rf"built\s+in\s+(19\d{{2}}|20[0-2]\d)",
    "year_upd": r"\b(19\d{2}|20[0-2]\d)\b",  # tahun disebut (update/roof/dll)
    "car_garage": rf"{NUM}\s*[- ]car\s*garage",
}
BINARY = {
    # kondisi & fasilitas (kehadiran kata saja)
    "b_pool": r"\bpool\b",
    "b_waterfront": r"waterfront|lake\s?front|ocean\s?front|river\s?front|bay\s?views?",
    "b_renovated": r"renovat|remodel|updated|upgrade|move[- ]in[- ]ready|brand[- ]new",
    "b_fixer": r"fixer|as[- ]is|distressed|tlc|handyman|needs\s+work|investor",
    "b_new_constr": r"new\s+construction|newly\s+built|never\s+lived",
    "b_garage": r"\bgarage\b",
    "b_fireplace": r"fireplace",
    "b_basement": r"basement",
    "b_finished_bsmt": r"finished\s+(?:lower\s+level|basement)",
    "b_acreage": r"\bacre",
    "b_multifamily": r"duplex|triplex|fourplex|multi[- ]family|two[- ]family|two\s+unit",
    "b_land": r"vacant\s+land|land\s+only|farm\s+land|buildable|parcel",
    "b_condo": r"\bcondo|minium\b|apartment",
    "b_townhouse": r"townhouse|town\s?home|end[- ]unit",
    "b_ranch": r"\branch\b|ranch[- ]style",
    "b_luxury": r"luxur|estate|gated|custom\s+built|masterpiece|stunning",
    "b_view": r"\bviews?\b|panoramic",
    "b_golf": r"golf",
    "b_hoa": r"\bhoa\b|homeowners?\s+association",
}
TYPES = {
    "t_victorian": r"victorian|colonial|bungalow|craftsman|cape\s+cod|tudor",
    "t_ranch2": r"raised[- ]ranch|split[- ]level|tri[- ]level|bi[- ]level",
}


def extract_features(texts):
    rows = []
    for t in texts:
        tl = t.lower()
        f = {"n_chars": len(t), "n_words": len(tl.split())}
        for k, pat in PATTERNS.items():
            m = re.findall(pat, tl)
            if k == "year_upd":
                f[k] = len(m)
            elif m:
                # pola multi-group mengembalikan tuple -> ambil group pertama yang terisi
                last = m[-1]
                val = next((g for g in (last if isinstance(last, tuple) else (last,)) if g), np.nan)
                try:
                    f[k] = float(str(val).replace(",", ""))
                except ValueError:
                    f[k] = np.nan
            else:
                f[k] = np.nan
        for k, pat in BINARY.items():
            f[k] = 1 if re.search(pat, tl) else 0
        for k, pat in TYPES.items():
            f[k] = 1 if re.search(pat, tl) else 0
        rows.append(f)
    return pd.DataFrame(rows)


_tr = pd.read_csv(_walk_find("train.csv"))
_te = pd.read_csv(_walk_find("test.csv"))
for _df in (_tr, _te):
    _df["text"] = _df["text"].fillna("")
_tr = _tr.merge(folds_df, on="id", how="left")
assert _tr["fold"].notna().all()

print("ekstraksi fitur regex (train)...")
F_tr = extract_features(_tr["text"].tolist())
print("ekstraksi fitur regex (test)...")
F_te = extract_features(_te["text"].tolist())
print(f"dimensi fitur: {F_tr.shape[1]}")
print("coverage fitur kunci (% non-null di train):")
for c in ["n_bed", "n_bath", "sqft", "acre", "year_built", "hoa_fee"]:
    print(f"  {c:12s}: {100 * F_tr[c].notna().mean():.1f}%")

# sanity: fitur tidak boleh nyaris 100% null (coverage aktual n_bed ~18%)
for c in ["n_bed", "n_bath", "sqft"]:
    assert F_tr[c].notna().mean() > 0.1, f"fitur {c} coverage terlalu rendah — cek regex"

X_tr, X_te = F_tr.fillna(-1), F_te.fillna(-1)  # LightGBM handle missing natively
y_all = np.log1p(_tr["listPrice"].values)

oof_l, te_sum, fold_maes = [], np.zeros(len(_te)), []
params = dict(objective="l1", metric="mae", learning_rate=0.05,
              num_leaves=63, min_child_samples=40, feature_fraction=0.9,
              bagging_fraction=0.9, bagging_freq=1, lambda_l1=0.5,
              n_estimators=3000, verbose=-1, seed=42)
for fold in range(5):
    idx_tr = (_tr["fold"] != fold).values
    idx_va = (_tr["fold"] == fold).values
    model = lgb.LGBMRegressor(**params)
    model.fit(X_tr[idx_tr], y_all[idx_tr],
              eval_set=[(X_tr[idx_va], y_all[idx_va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
    va_pred = np.expm1(model.predict(X_tr[idx_va], num_iteration=model.best_iteration_))
    te_sum += np.expm1(model.predict(X_te, num_iteration=model.best_iteration_))
    fold_maes.append(mean_absolute_error(_tr.loc[idx_va, "listPrice"], va_pred))
    oof_l.append(pd.DataFrame({"id": _tr.loc[idx_va, "id"].values, "fold": fold,
                               "listPrice": _tr.loc[idx_va, "listPrice"].values,
                               "oof_pred": va_pred}))
    print(f"lgbm fold {fold}: MAE = {fold_maes[-1]:,.0f} "
          f"(best_iter={model.best_iteration_})")

print(f"\nLGBM CV-MAE: {np.mean(fold_maes):,.0f} +/- {np.std(fold_maes):,.0f}")
imp = pd.Series(model.feature_importances_, index=X_tr.columns).sort_values(ascending=False)
print("top-10 feature importance:")
print(imp.head(10).to_string())

pd.concat(oof_l, ignore_index=True).to_csv("oof_pred_lgbm.csv", index=False)
pd.DataFrame({"id": _te["id"].values, "test_pred": te_sum / 5}).to_csv("test_pred_lgbm.csv", index=False)
print("-> oof_pred_lgbm.csv + test_pred_lgbm.csv tersimpan")

# CHECKPOINT CELL 3D: LGBM CV-MAE tercetak; artefak tersimpan.
# Kalau CV-MAE LGBM >= Ridge secara jelas (> std), tetap simpan — blend
# bisa saja tetap menang karena diversitas informasi, CELL 5 yang memutuskan.


# %% CELL 7 — KANDIDAT 4: ModernBERT-base FINE-TUNE (GPU, ~1.5 jam)
# Disuruh apa: fine-tune answerdotai/ModernBERT-base + head regresi mean
# pooling, setup IDENTIK dengan CELL 3 v2 (fold beku, log1p + centering,
# L1, bs efektif 16, lr 2e-5, 3 epoch, AMP fp16) supaya comparable.
# Tujuan: kandidat transformer kedua dengan arsitektur berbeda (rotary
# embeddings, GeGLU) -> diversitas model untuk blend di CELL 8.
# Perbedaan vs CELL 3: tokenizer ModernBERT (BUKAN sentencepiece), context
# native 8192 tapi tetap MAX_LEN=512 (p90 token kita cuma 340).
# Cell ini MANDIRI (tidak butuh state CELL 3) — bisa run langsung dari CELL 6.

import os
import time
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from sklearn.metrics import mean_absolute_error

# ----------------------------- KONFIG ---------------------------------------
MB_MODEL = "answerdotai/ModernBERT-base"
MB_MAX_LEN = 512
MB_BS = 8
MB_GRAD_ACCUM = 2
MB_EPOCHS = 3
MB_LR = 2e-5
MB_WD = 0.01
MB_WARMUP = 0.1
MB_RUN_TAG = "modernbert"   # artefak: oof_pred_modernbert.csv, test_pred_modernbert_fold{k}.csv
MB_FOLDS = [0, 1, 2, 3, 4]
SEED = 42  # sama dengan CELL 1/2 (seed fold bekuan) — ditulis ulang agar CELL 7 mandiri

device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda", "GPU tidak terdeteksi — cek Settings -> Accelerator"


def set_seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


try:
    MB_SCALER = torch.amp.GradScaler("cuda")

    def mb_autocast():
        return torch.amp.autocast("cuda", dtype=torch.float16)
except (AttributeError, TypeError):
    MB_SCALER = torch.cuda.amp.GradScaler()

    def mb_autocast():
        return torch.cuda.amp.autocast()


def mb_fix_fp16_grads(model_):
    for p in model_.parameters():
        if p.grad is not None and p.grad.dtype == torch.float16:
            try:
                p.grad_dtype = None
            except (AttributeError, RuntimeError):
                pass
            p.grad = p.grad.float()


mb_tokenizer = AutoTokenizer.from_pretrained(MB_MODEL)
print(f"tokenizer ModernBERT: {mb_tokenizer.__class__.__name__}, "
      f"vocab={mb_tokenizer.vocab_size}")


class MBListingDataset(Dataset):
    def __init__(self, texts, targets=None):
        self.enc = mb_tokenizer(list(texts), truncation=True, max_length=MB_MAX_LEN)
        self.targets = targets

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        if self.targets is not None:
            item["labels"] = torch.tensor(float(self.targets[i]))
        return item


def mb_collate(batch):
    maxlen = max(len(b["input_ids"]) for b in batch)
    has_token_type = "token_type_ids" in batch[0]
    input_ids, attn, ttids, labels = [], [], [], []
    for b in batch:
        ids = b["input_ids"]
        pad = maxlen - len(ids)
        input_ids.append(torch.cat([ids, torch.zeros(pad, dtype=torch.long)]))
        attn.append(torch.cat([torch.ones(len(ids), dtype=torch.long),
                               torch.zeros(pad, dtype=torch.long)]))
        if has_token_type:
            t = b["token_type_ids"]
            ttids.append(torch.cat([t, torch.zeros(pad, dtype=torch.long)]))
        labels.append(b["labels"] if "labels" in b else torch.tensor(0.0))
    out = {"input_ids": torch.stack(input_ids),
           "attention_mask": torch.stack(attn),
           "labels": torch.stack(labels)}
    if has_token_type:
        out["token_type_ids"] = torch.stack(ttids)
    return out


class MBPriceRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            self.backbone = AutoModel.from_pretrained(MB_MODEL, dtype=torch.float32)
        except TypeError:
            self.backbone = AutoModel.from_pretrained(MB_MODEL, torch_dtype=torch.float32)
        for p in self.backbone.parameters():
            assert p.dtype == torch.float32, f"param {p.dtype}, harusnya fp32"
        hidden = self.backbone.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kw = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            try:
                out = self.backbone(**kw, token_type_ids=token_type_ids)
            except TypeError:
                out = self.backbone(**kw)  # ModernBERT tanpa token_type_ids
        else:
            out = self.backbone(**kw)
        hs = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hs.dtype)
        pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return self.head(self.drop(pooled)).squeeze(-1)


@torch.no_grad()
def mb_predict_texts(model, texts, bs=32):
    model.eval()
    dl = DataLoader(MBListingDataset(texts), batch_size=bs, shuffle=False,
                    collate_fn=mb_collate)
    preds = []
    for batch in dl:
        with mb_autocast():
            kw = {"input_ids": batch["input_ids"].to(device),
                  "attention_mask": batch["attention_mask"].to(device)}
            if "token_type_ids" in batch:
                kw["token_type_ids"] = batch["token_type_ids"].to(device)
            logits = model(**kw)
        preds.append(logits.float().cpu().numpy())
    return np.concatenate(preds)


def mb_train_one_fold(train_df, test_df, fold):
    set_seed_all(SEED + fold)
    tr = train_df[train_df["fold"] != fold]
    va = train_df[train_df["fold"] == fold]

    y_tr = np.log1p(tr["listPrice"].values)
    y_va = va["listPrice"].values
    y_center = float(y_tr.mean())

    ds_tr = MBListingDataset(tr["text"].tolist(), y_tr - y_center)
    dl_tr = DataLoader(ds_tr, batch_size=MB_BS, shuffle=True, collate_fn=mb_collate,
                       num_workers=2, drop_last=True)

    model = MBPriceRegressor().to(device)
    criterion = nn.L1Loss()
    opt = torch.optim.AdamW(model.parameters(), lr=MB_LR, weight_decay=MB_WD)
    steps_per_epoch = len(dl_tr) // MB_GRAD_ACCUM
    total_steps = steps_per_epoch * MB_EPOCHS
    sched = get_cosine_schedule_with_warmup(
        opt, int(MB_WARMUP * total_steps), total_steps)

    t0 = time.time()
    global_step, running = 0, []
    model.train()
    for epoch in range(MB_EPOCHS):
        for step, batch in enumerate(dl_tr):
            with mb_autocast():
                kw = {"input_ids": batch["input_ids"].to(device),
                      "attention_mask": batch["attention_mask"].to(device)}
                if "token_type_ids" in batch:
                    kw["token_type_ids"] = batch["token_type_ids"].to(device)
                logits = model(**kw)
                loss = criterion(logits, batch["labels"].to(device))
            MB_SCALER.scale(loss / MB_GRAD_ACCUM).backward()
            running.append(loss.item())
            if (step + 1) % MB_GRAD_ACCUM == 0:
                mb_fix_fp16_grads(model)
                MB_SCALER.step(opt)
                MB_SCALER.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                global_step += 1
                if global_step % 100 == 0:
                    print(f"  MB fold {fold} ep{epoch} step {global_step}/{total_steps} "
                          f"loss={np.mean(running[-100:]):.4f}")
    print(f"  MB fold {fold}: training selesai dalam {(time.time() - t0) / 60:.1f} menit")

    va_pred_log = mb_predict_texts(model, va["text"].tolist()) + y_center
    cap = float(np.log1p(train_df["listPrice"].max()))
    va_pred = np.expm1(np.clip(va_pred_log, 0, cap))
    fold_mae = mean_absolute_error(y_va, va_pred)
    print(f"  MB fold {fold}: MAE = {fold_mae:,.0f}")

    te_pred_log = mb_predict_texts(model, test_df["text"].tolist()) + y_center
    te_pred = np.expm1(np.clip(te_pred_log, 0, cap))

    oof_part = pd.DataFrame({"id": va["id"].values, "fold": fold,
                             "listPrice": y_va, "oof_pred": va_pred})
    te_part = pd.DataFrame({"id": test_df["id"].values, f"test_pred_f{fold}": te_pred})
    te_part.to_csv(f"test_pred_{MB_RUN_TAG}_fold{fold}.csv", index=False)

    del model
    torch.cuda.empty_cache()
    return oof_part, fold_mae


# ------------------------------ RUN -----------------------------------------
def _mb_walk_find(name, _base="/kaggle/input"):
    for root, _, files in os.walk(_base):
        if name in files:
            return os.path.join(root, name)
    raise FileNotFoundError(name)


if not os.path.exists("folds.csv"):
    raise RuntimeError("folds.csv belum ada — jalankan CELL 2 dulu!")
_mb_folds = pd.read_csv("folds.csv")
_mb_train = pd.read_csv(_mb_walk_find("train.csv"))
_mb_test = pd.read_csv(_mb_walk_find("test.csv"))
for _df in (_mb_train, _mb_test):
    _df["text"] = _df["text"].fillna("")
_mb_train = _mb_train.drop(
    columns=[c for c in ("fold", "price_bin") if c in _mb_train.columns])
_mb_train = _mb_train.merge(_mb_folds, on="id", how="left")
assert _mb_train["fold"].notna().all()

mb_oof_parts, mb_maes = [], []
for fold in MB_FOLDS:
    print(f"\n===== MB FOLD {fold} ({MB_MODEL}) =====")
    oof_part, fold_mae = mb_train_one_fold(_mb_train, _mb_test, fold)
    mb_oof_parts.append(oof_part)
    mb_maes.append(fold_mae)

mb_oof = pd.concat(mb_oof_parts, ignore_index=True)
print("\n=== HASIL CV MODERNBERT ===")
for fold, m in zip(MB_FOLDS, mb_maes):
    print(f"fold {fold}: MAE = {m:,.0f}")
print(f"MB CV-MAE: {np.mean(mb_maes):,.0f} +/- {np.std(mb_maes):,.0f}")
mb_oof.to_csv(f"oof_pred_{MB_RUN_TAG}.csv", index=False)
print(f"-> oof_pred_{MB_RUN_TAG}.csv + test_pred_{MB_RUN_TAG}_fold*.csv tersimpan")

# CHECKPOINT CELL 7: MB CV-MAE tercetak + artefak tersimpan.
# Bandingkan dengan: ridge 346.772, deberta-v2 354.855 (run bersih) /
# 409.370 (run cacat fold-2). Kalau MB CV > ridge secara jelas, MB tetap
# disimpan — CELL 8 yang memutuskan bobot blend di OOF.


# %% CELL 4 — TAHAP 6/7: ERROR ANALYSIS PER SEGMEN HARGA + PILIH LOSS
# Disuruh apa: lihat MAE per decile harga (di mana model salah terbesar),
# bandingkan run L1 vs MSE (kalau keduanya dijalankan), pilih final by CV.
# Tujuan: tahu headroom/ceiling SEBELUM kejar eksperimen baru.

import glob as _glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

runs = sorted(_glob.glob("oof_pred_*.csv"))
assert runs, "Belum ada oof_pred_*.csv — jalankan CELL 3 (SMOKE=False) dulu"
print("run tersedia:", runs)

for path in runs:
    tag = path.replace("oof_pred_", "").replace(".csv", "")
    oof = pd.read_csv(path)
    mae = mean_absolute_error(oof["listPrice"], oof["oof_pred"])
    print(f"run={tag}: CV-MAE = {mae:,.0f} (n={len(oof)})")

    # MAE per decile harga sebenarnya
    oof["price_decile"] = pd.qcut(oof["listPrice"], q=10, labels=False)
    seg = oof.groupby("price_decile").apply(
        lambda g: pd.Series({
            "n": len(g),
            "median_harga": g["listPrice"].median(),
            "MAE": mean_absolute_error(g["listPrice"], g["oof_pred"]),
            "MAE_pct": 100 * mean_absolute_error(g["listPrice"], g["oof_pred"]) / g["listPrice"].median(),
        })
    )
    print(f"\n--- Error per decile (run={tag}) ---")
    print(seg.round(0))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(seg.index, seg["MAE"], color="indianred")
    ax.set_xlabel("decile harga (0 = termurah)")
    ax.set_ylabel("MAE ($)")
    ax.set_title(f"MAE per decile harga — run={tag}")
    plt.tight_layout()
    plt.show()

# KEPUTUSAN LOSS: bandingkan CV-MAE l1 vs mse (kalau keduanya ada).
# Aturan: selisih < std fold antar-run = SERI -> pakai yang sudah jadi default (l1).
summary = []
for path in runs:
    tag = path.replace("oof_pred_", "").replace(".csv", "")
    oof = pd.read_csv(path)
    summary.append({"run": tag,
                    "cv_mae": mean_absolute_error(oof["listPrice"], oof["oof_pred"])})
print("\n=== RINGKASAN EKSPERIMEN ===")
print(pd.DataFrame(summary).round(0).to_string(index=False))
print("\nVerdict headroom/ceiling: jika MAE% di decile murah & mahal sama-sama")
print("tinggi -> kemungkinan ceiling informasi teks; jika hanya satu segmen")
print("buruk -> masih ada headroom (pertimbangkan segment-specific treatment).")

# CHECKPOINT CELL 4: tabel per-decile + verdict + run terpilih tercatat.


# %% CELL 5 — TAHAP 8/9: PREDIKSI FINAL (FOLD-MEAN) + ASSERT SUBMISSION
# Disuruh apa: pilih run terbaik by CV, rata-rata prediksi test 5 fold,
# clip ke rentang harga train, bangun submission sesuai sample_submission,
# lalu ASSERT semua format.
# Keputusan final-fit: FOLD-MEAN (bukan refit-full) — data 14.6k relatif
# kecil, fold-mean lebih robust + dapat efek ensembling 5 model gratis.

import glob as _glob
import os as _os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


def _find_csv_local(name, _base="/kaggle/input"):
    # pencarian langsung (tidak bergantung state CELL 1)
    hits = []
    for root, _, files in _os.walk(_base):
        for f in files:
            if f.lower() == name:
                hits.append(_os.path.join(root, f))
    assert hits, f"{name} tidak ditemukan di {_base}"
    return hits[0]


sub = pd.read_csv(_find_csv_local("sample_submission.csv"))
train_full = pd.read_csv(_find_csv_local("train.csv"))
train_full["text"] = train_full["text"].fillna("")
price_min, price_max = train_full["listPrice"].min(), train_full["listPrice"].max()

# pilih run TRANSFORMER terbaik berdasarkan CV-MAE (bukan public LB!)
# kandidat transformer = run yang punya artefak per-fold test_pred_{tag}_fold*.csv
# (ridge & lgbm disimpan sebagai file tunggal -> masuk blend sebagai kandidat
# terpisah di blok blend, BUKAN sebagai run transformer)
tf_runs = []
for path in sorted(_glob.glob("oof_pred_*.csv")):
    tag = path.replace("oof_pred_", "").replace(".csv", "")
    if tag in ("ridge", "lgbm"):
        continue  # kandidat blend terpisah, bukan run transformer
    if _glob.glob(f"test_pred_{tag}_fold*.csv"):
        tf_runs.append(path)
if not tf_runs:
    print("TIDAK ADA run transformer dengan artefak lengkap "
          "(oof_pred_*.csv + test_pred_*_fold*.csv).")
    print("-> jalankan CELL 3C (rescue) dulu kalau CELL 3 v2 sempat di-interrupt.")
    print("-> final akan memakai blend Ridge (+LGBM) saja.")
    best_tag = None
else:
    cv_scores = {}
    for path in tf_runs:
        tag = path.replace("oof_pred_", "").replace(".csv", "")
        oof = pd.read_csv(path)
        cv_scores[tag] = mean_absolute_error(oof["listPrice"], oof["oof_pred"])
    best_tag = min(cv_scores, key=cv_scores.get)
    print("CV per run transformer:", {k: round(v) for k, v in cv_scores.items()})
    print(f"-> run terpilih (by CV): {best_tag}")

if best_tag is not None:
    te_files = sorted(_glob.glob(f"test_pred_{best_tag}_fold*.csv"))
    assert len(te_files) >= 3, f"butuh minimal 3 file test_pred, ditemukan {len(te_files)}"
    te = pd.read_csv(te_files[0])[["id"]]
    for f in te_files:
        te = te.merge(pd.read_csv(f), on="id", how="inner")
    pred_cols = [c for c in te.columns if c.startswith("test_pred_f")]
    print(f"fold tersedia: {len(te_files)} ({[c.split('_f')[-1] for c in pred_cols]})")
    te["pred_tf"] = te[pred_cols].mean(axis=1)  # fold-mean transformer

# ------------------- BLEND SEMUA KANDIDAT dari OOF (Tahap 7) ----------------
# Tahap 7 master prompt: bobot blend WAJIB dioptimasi di OOF (leak-free).
# Kandidat: transformer terpilih (bila ada) + ridge + lgbm (bila ada).
# Tes lokal: kontribusi LGBM ~seri; grid search di OOF asli yang memutuskan.
cands = {}
if best_tag is not None:
    cands["tf"] = pd.read_csv(f"oof_pred_{best_tag}.csv")
if _os.path.exists("oof_pred_ridge.csv"):
    cands["ridge"] = pd.read_csv("oof_pred_ridge.csv")
if _os.path.exists("oof_pred_lgbm.csv"):
    cands["lgbm"] = pd.read_csv("oof_pred_lgbm.csv")
assert cands, "tidak ada kandidat blend — jalankan CELL 3/3B/3C dulu"

# align semua OOF berdasarkan id kandidat pertama
first = next(iter(cands.values()))
base = first[["id", "listPrice"]].copy()
for name, oof_c in cands.items():
    base = base.merge(oof_c[["id", "oof_pred"]].rename(
        columns={"oof_pred": name}), on="id", how="inner")
assert len(base) == len(first), "OOF id tidak align antar model!"
y_true = base["listPrice"].values
model_names = list(cands.keys())
print("\nOOF-MAE per model: " + ", ".join(
    f"{n}={mean_absolute_error(y_true, base[n].values):,.0f}" for n in model_names))

# grid search bobot simplex generik (semua bobot >= 0, total = 1)
steps = 20


def simplex_weights(k, total=None):
    """enumerate semua kombinasi bobot bulat >= 0 sejumlah k model yang
    menjumlah ke `total` (default steps)"""
    if total is None:
        total = steps
    if k == 1:
        yield (total,)
        return
    for i in range(total + 1):
        for rest in simplex_weights(k - 1, total - i):
            yield (i,) + rest


best_mae, best_w = np.inf, None
for ws in simplex_weights(len(model_names)):
    weights = {n: w / steps for n, w in zip(model_names, ws)}
    pred = sum(weights[n] * base[n].values for n in weights)
    mae = mean_absolute_error(y_true, pred)
    if mae < best_mae:
        best_mae, best_w = mae, weights
print(f"OOF-MAE blend terbaik: {best_mae:,.0f}  bobot: "
      + ", ".join(f"{n}={best_w[n]:.2f}" for n in best_w))

# bangun pred test final sesuai bobot
if best_tag is not None:
    te = te.rename(columns={"pred_tf": "tf"})
else:
    # tanpa transformer: ambil id test dari artefak ridge/lgbm
    _src = "test_pred_ridge.csv" if _os.path.exists("test_pred_ridge.csv") \
        else "test_pred_lgbm.csv"
    te = pd.read_csv(_src)[["id"]]
for n in best_w:
    if n == "tf":
        continue
    te = te.merge(pd.read_csv(f"test_pred_{n}.csv")
                  .rename(columns={"test_pred": n}), on="id")
te["listPrice"] = sum(best_w[n] * te[n].values for n in best_w)

# clip rentang harga train
te["listPrice"] = te["listPrice"].clip(price_min, price_max)

# susun submission MENGIKUTI URUTAN sample_submission (bukan urutan test.csv)
submission = sub[["id"]].merge(te[["id", "listPrice"]], on="id", how="left")
assert submission["listPrice"].notna().all(), "ada id test tanpa prediksi!"

# ------------------------------- ASSERTS ------------------------------------
assert list(submission.columns) == ["id", "listPrice"], "kolom salah"
assert len(submission) == len(sub), f"jumlah baris {len(submission)} != {len(sub)}"
assert (submission["id"].values == sub["id"].values).all(), "urutan id beda!"
assert submission["listPrice"].notna().all(), "ada NaN"
assert np.isfinite(submission["listPrice"]).all(), "ada inf"
assert submission["listPrice"].between(price_min, price_max).all(), "di luar rentang train"
print("\nSEMUA ASSERT LOLOS:")
print(f"  baris={len(submission)}, kolom={list(submission.columns)}, "
      f"NaN=0, rentang=[{price_min:,.0f}, {price_max:,.0f}]")

# sanity distribusi: prediksi test vs harga train
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(train_full["listPrice"], bins=60, alpha=0.5, density=True, label="train aktual")
ax.hist(submission["listPrice"], bins=60, alpha=0.5, density=True, label="test prediksi")
ax.legend()
ax.set_title("Distribusi harga: train aktual vs prediksi test")
plt.tight_layout()
plt.show()

submission.to_csv("submission.csv", index=False)
print("-> submission.csv TERSIMPAN (kandidat final, dipilih by CV)")

# CHECKPOINT CELL 5: submission lolos semua assert + sanity distribusi masuk akal.


# %% CELL 8 — FINAL: BLEND SEMUA KANDIDAT (TERMASUK MODERNBERT) + SUBMISSION
# Disuruh apa: ulangi logika blend CELL 5 tapi dengan kandidat transformer
# BEBAS (l1_v2 terbaik ATAU modernbert terbaik — auto-pilih keduanya by CV)
# + Ridge + LGBM, lalu tulis submission_modernbert_blend.csv.
# Tujuan: kandidat final yang memanfaatkan transformer terbaik + ModernBERT.
# Bobot dioptimasi di OOF (Tahap 7). Jalankan SETELAH CELL 7 selesai.
# Selalu menulis file BARU (submission_modernbert_blend.csv) — submission.csv
# lama (429k LB) TIDAK DITIMPA, supaya selalu ada fallback aman.

import glob as _glob8
import os as _os8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


def _find_csv_local8(name, _base="/kaggle/input"):
    hits = []
    for root, _, files in _os8.walk(_base):
        for f in files:
            if f.lower() == name:
                hits.append(_os8.path.join(root, f))
    assert hits, f"{name} tidak ditemukan di {_base}"
    return hits[0]


sub8 = pd.read_csv(_find_csv_local8("sample_submission.csv"))
train_full8 = pd.read_csv(_find_csv_local8("train.csv"))
train_full8["text"] = train_full8["text"].fillna("")
price_min8, price_max8 = train_full8["listPrice"].min(), train_full8["listPrice"].max()

# kandidat transformer = SEMUA run dengan artefak per-fold LENGKAP
tf_cands8 = {}
for path in sorted(_glob8.glob("oof_pred_*.csv")):
    tag = path.replace("oof_pred_", "").replace(".csv", "")
    if tag in ("ridge", "lgbm"):
        continue
    te_files = sorted(_glob8.glob(f"test_pred_{tag}_fold*.csv"))
    if te_files:
        cv = mean_absolute_error(pd.read_csv(path)["listPrice"],
                                 pd.read_csv(path)["oof_pred"])
        tf_cands8[tag] = (path, te_files, cv)
print("kandidat transformer:", {t: round(v[2]) for t, v in tf_cands8.items()})

# fold-mean test per kandidat transformer
cands8 = {}
oof_base8 = None
for tag, (path, te_files, cv) in tf_cands8.items():
    t = pd.read_csv(te_files[0])[["id"]]
    for f in te_files:
        t = t.merge(pd.read_csv(f), on="id", how="inner")
    cols = [c for c in t.columns if c.startswith("test_pred_f")]
    cands8[f"tf_{tag}"] = (pd.read_csv(path), t[cols].mean(axis=1).values)
    print(f"  tf_{tag}: CV={cv:,.0f}, fold tersedia={len(te_files)}")
if _os8.path.exists("oof_pred_ridge.csv"):
    oof_r8 = pd.read_csv("oof_pred_ridge.csv")
    oof_base8 = oof_r8[["id", "listPrice"]] if oof_base8 is None else oof_base8
    cands8["ridge"] = (oof_r8, pd.read_csv("test_pred_ridge.csv")["test_pred"].values)
if _os8.path.exists("oof_pred_lgbm.csv"):
    oof_l8 = pd.read_csv("oof_pred_lgbm.csv")
    oof_base8 = oof_l8[["id", "listPrice"]] if oof_base8 is None else oof_base8
    cands8["lgbm"] = (oof_l8, pd.read_csv("test_pred_lgbm.csv")["test_pred"].values)
assert cands8, "tidak ada kandidat — jalankan CELL 3/3B/7 dulu"

# jika beberapa transformer memakai id train yang sama, align di satu base
names8 = list(cands8.keys())
y_true8 = cands8[names8[0]][0].set_index("id")["listPrice"]
base8 = pd.DataFrame({"id": y_true8.index, "listPrice": y_true8.values})
for n in names8:
    oof_c, _ = cands8[n]
    base8 = base8.merge(oof_c[["id", "oof_pred"]].rename(
        columns={"oof_pred": n}), on="id", how="inner")
print(f"baris OOF align: {len(base8)}")
y8 = base8["listPrice"].values
print("OOF-MAE per model: " + ", ".join(
    f"{n}={mean_absolute_error(y8, base8[n].values):,.0f}" for n in names8))

# grid search simplex generik
steps8 = 20


def simplex8(k, total=None):
    if total is None:
        total = steps8
    if k == 1:
        yield (total,)
        return
    for i in range(total + 1):
        for rest in simplex8(k - 1, total - i):
            yield (i,) + rest


best_mae8, best_w8 = np.inf, None
n_grid8 = 0
for ws in simplex8(len(names8)):
    n_grid8 += 1
    weights = {n: w / steps8 for n, w in zip(names8, ws)}
    pred = sum(weights[n] * base8[n].values for n in weights)
    mae = mean_absolute_error(y8, pred)
    if mae < best_mae8:
        best_mae8, best_w8 = mae, weights
print(f"grid dievaluasi: {n_grid8} kombinasi")
print(f"OOF-MAE blend terbaik: {best_mae8:,.0f}  bobot: "
      + ", ".join(f"{n}={best_w8[n]:.2f}" for n in best_w8))

# bangun test final: id dari sample_submission, align per kandidat test
te8 = sub8[["id"]].copy()
for n in names8:
    _, te_pred = cands8[n]
    te_src = pd.DataFrame({"id": (pd.read_csv(f'test_pred_{n[3:]}_fold0.csv')['id']
                                  if n.startswith("tf_") else
                                  pd.read_csv(f'test_pred_{n}.csv')['id']),
                           n: te_pred})
    te8 = te8.merge(te_src, on="id", how="left")
assert te8[[n for n in names8]].notna().all().all(), "ada id test tanpa prediksi!"
te8["listPrice"] = sum(best_w8[n] * te8[n].values for n in best_w8)
te8["listPrice"] = te8["listPrice"].clip(price_min8, price_max8)

submission8 = sub8[["id"]].merge(te8[["id", "listPrice"]], on="id", how="left")
assert list(submission8.columns) == ["id", "listPrice"]
assert len(submission8) == len(sub8)
assert (submission8["id"].values == sub8["id"].values).all()
assert submission8["listPrice"].notna().all() and np.isfinite(submission8["listPrice"]).all()
assert submission8["listPrice"].between(price_min8, price_max8).all()
print("\nSEMUA ASSERT LOLOS")

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(train_full8["listPrice"], bins=60, alpha=0.5, density=True, label="train aktual")
ax.hist(submission8["listPrice"], bins=60, alpha=0.5, density=True, label="test prediksi")
ax.legend()
ax.set_title("Distribusi harga: train aktual vs prediksi test (blend+MB)")
plt.tight_layout()
plt.show()

submission8.to_csv("submission_modernbert_blend.csv", index=False)
print("-> submission_modernbert_blend.csv TERSIMPAN (submission.csv lama TIDAK ditimpa)")

# CHECKPOINT CELL 8: bandingkan OOF-MAE blend ini vs 336.815 (blend CELL 5).
# Submit HANYA kalau menang jelas (> std improvement); kalau seri/kalah,
# tetap pakai submission.csv lama (429k LB).

# Disuruh apa: kumpulkan semua angka eksperimen ke metrics_summary.csv dan
# cetak decision log (kenapa X bukan Y, termasuk yang gagal/seri).

# %% CELL 6 — TAHAP 10: METRICS SUMMARY + DECISION LOG

import glob as _glob
import os as _os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

rows = []

# baseline median CV (dari CELL 2, dihitung ulang agar self-contained)
folds_df = pd.read_csv("folds.csv")
tr = pd.read_csv(_find_csv_local("train.csv")).merge(folds_df, on="id")
maes = []
for fold in range(5):
    va = tr[tr["fold"] == fold]
    med = tr[tr["fold"] != fold]["listPrice"].median()
    maes.append(mean_absolute_error(va["listPrice"], np.full(len(va), med)))
rows.append({"experiment": "baseline_median", "cv_mae": float(np.mean(maes)),
             "cv_std": float(np.std(maes)), "note": "titik nol CV"})

for path in sorted(_glob.glob("oof_pred_*.csv")):
    tag = path.replace("oof_pred_", "").replace(".csv", "")
    oof = pd.read_csv(path)
    per_fold = oof.groupby("fold").apply(
        lambda g: mean_absolute_error(g["listPrice"], g["oof_pred"]))
    rows.append({"experiment": f"deberta_{tag}",
                 "cv_mae": float(per_fold.mean()),
                 "cv_std": float(per_fold.std()),
                 "note": f"5-fold, fold bekuan seed={SEED}"})

metrics = pd.DataFrame(rows).round(0)
metrics.to_csv("metrics_summary.csv", index=False)
print(metrics.to_string(index=False))

print("""
=== DECISION LOG ===
1. Target transform : log1p(listPrice) — target right-skewed (skew 15.3);
                      prediksi di-inverse expm1 + clip rentang train.
2. Validasi         : StratifiedKFold(5) pada 10-bin harga, seed 42,
                      folds.csv DIBEKUKAN sejak CV pertama; semua eksperimen
                      pakai fold yang sama.
3. Model final      : deberta-v3-base + head regresi (MEAN pooling +
                      target centering per fold), bs efektif 16, lr 2e-5,
                      3 epoch, cosine + warmup 10%, fp16 AMP, dropout 0.1.
4. Loss             : L1 di log-target (selaras MAE); MSE tidak sempat
                      diuji ulang di v2 (waktu).
5. Ensembling       : blend transformer + Ridge (+LGBM bila ada), bobot
                      dioptimasi via grid search simplex di OOF (Tahap 7).
6. Final fit        : FOLD-MEAN (rata-rata prediksi test semua fold),
                      BUKAN refit-full — data 14.6k kecil, fold-mean lebih
                      robust + efek ensembling fold gratis.
7. Post-processing  : expm1 -> clip [min,max] harga train -> urut sesuai
                      sample_submission -> assert penuh (shape/kolom/NaN/id).

=== EKSPERIMEN GAGAL / SERI (angka nyata) ===
- deberta v1 (CLS pooling, tanpa centering): CV-MAE 659,867 — KALAH dari
  baseline median 550,252; LB 732,654. Akar masalah: [CLS] DeBERTa-v3
  (model MLM) bukan sentence representation. Fix = mean pooling + centering.
- LGBM regex FE: CV-MAE 459,963 standalone; kontribusi blend vs Ridge
  hanya ~800 MAE (<< std 30k = SERI) — fitur regex sudah tertangkap TF-IDF.
- Blend v1+Ridge: w_tf=0.05 (v1 tidak menambah apa pun) — dibuang.
""")
