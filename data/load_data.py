from datasets import load_dataset
import os

# =========================
# CONFIGURACIÓN
# =========================
DATASET_NAME = "databricks/databricks-dolly-15k"
CACHE_DIR = "./cache"       # caché de HuggingFace
OUTPUT_DIR = "./datasets"   # splits procesados
SEED = 42

TRAIN_RATIO = 0.8
DEV_RATIO = 0.1
TEST_RATIO = 0.1

assert abs(TRAIN_RATIO + DEV_RATIO + TEST_RATIO - 1.0) < 1e-9

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# DESCARGA
# =========================
print("Descargando dataset...")
ds = load_dataset(DATASET_NAME, cache_dir=CACHE_DIR)

full_ds = ds["train"]
print(f"Total ejemplos: {len(full_ds)}")
print(f"Columnas: {full_ds.column_names}")
print(f"\nEjemplo:\n{full_ds[0]}")

# =========================
# SPLITS
# =========================
print("\nCreando splits...")

split_1 = full_ds.train_test_split(
    test_size=(DEV_RATIO + TEST_RATIO),
    seed=SEED
)
train_ds = split_1["train"]
temp_ds  = split_1["test"]

split_2 = temp_ds.train_test_split(
    test_size=TEST_RATIO / (DEV_RATIO + TEST_RATIO),
    seed=SEED
)
dev_ds  = split_2["train"]
test_ds = split_2["test"]

print(f"Train: {len(train_ds)}")
print(f"Dev:   {len(dev_ds)}")
print(f"Test:  {len(test_ds)}")

# =========================
# GUARDADO
# =========================
print("\nGuardando splits en disco...")

train_ds.save_to_disk(os.path.join(OUTPUT_DIR, "train"))
dev_ds.save_to_disk(os.path.join(OUTPUT_DIR, "dev"))
test_ds.save_to_disk(os.path.join(OUTPUT_DIR, "test"))

print("✅ Dataset descargado y guardado correctamente.")