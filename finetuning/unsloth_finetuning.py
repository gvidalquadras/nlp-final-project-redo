# =============================================================
# FINE-TUNING CON UNSLOTH
# Modelo: TinyLlama-1.1B-Chat
# Dataset: Databricks Dolly 15k
# =============================================================

import os
import time
import torch
import csv
import itertools
from datasets import load_from_disk
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

# =============================================================
# CONFIGURACIÓN — mismos valores que en PEFT para poder comparar
# =============================================================
MODEL_NAME   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_DIR  = "./datasets"
OUTPUT_BASE  = "./outputs/unsloth"
RESULTS_FILE = "./outputs/unsloth/grid_search_results.csv"
SEED         = 42
MAX_LENGTH   = 512

# Undersampling
TRAIN_SAMPLES = 200
DEV_SAMPLES   = 50
NUM_EPOCHS    = 1

# Grid search
LORA_RANKS      = [8, 16, 32]
ALPHA_FACTORS   = [1, 2]
LEARNING_RATES  = [1e-4, 2e-4]
GRID           = list(itertools.product(LORA_RANKS, ALPHA_FACTORS, LEARNING_RATES))

# Hiperparámetros fijos
LORA_DROPOUT   = 0.05
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]

os.makedirs(OUTPUT_BASE, exist_ok=True)

# =============================================================
# FORMATO DE PROMPT
# =============================================================
def format_prompt(example):
    if example.get("context", "").strip():
        return (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Context:\n{example['context']}\n\n"
            f"### Response:\n{example['response']}"
        )
    else:
        return (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Response:\n{example['response']}"
        )

def format_dataset(example):
    """Unsloth espera una columna 'text' con el prompt completo."""
    return {"text": format_prompt(example)}

# =============================================================
# CARGA Y PREPROCESAMIENTO DEL DATASET
# =============================================================
print("Cargando dataset...")
train_ds = load_from_disk(os.path.join(DATASET_DIR, "train"))
dev_ds   = load_from_disk(os.path.join(DATASET_DIR, "dev"))

# Mismo undersampling reproducible que en PEFT
train_ds = train_ds.shuffle(seed=SEED).select(range(TRAIN_SAMPLES))
dev_ds   = dev_ds.shuffle(seed=SEED).select(range(DEV_SAMPLES))

# Unsloth usa SFTTrainer que espera una columna "text" con el
# prompt ya formateado, en vez de tokenizar manualmente.
print("Formateando dataset...")
train_ds = train_ds.map(format_dataset)
dev_ds   = dev_ds.map(format_dataset)

# =============================================================
# GRID SEARCH
# =============================================================
results = []

for (r, alpha_factor, lr) in GRID:
    alpha = alpha_factor * r

    run_name   = f"r{r}_a{alpha}_lr{str(lr).replace('-', '').replace('.', '')}"
    output_dir = os.path.join(OUTPUT_BASE, run_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Experimento: r={r}, alpha={alpha}, lr={lr}")
    print(f"{'='*60}")

    # ---------------------------------------------------------
    # Carga del modelo con Unsloth
    # DIFERENCIA CLAVE vs PEFT:
    # FastLanguageModel.from_pretrained() hace en una sola llamada
    # lo que en PEFT requería BitsAndBytesConfig + from_pretrained
    # + get_peft_model por separado. Unsloth gestiona
    # internamente la cuantización y los kernels optimizados.
    # ---------------------------------------------------------
    print("Cargando modelo con Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_LENGTH,
        load_in_4bit=True,       # QLoRA 4-bit, mismo que PEFT
        dtype=torch.bfloat16,    # bf16 en RTX 30xx (Ampere)
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---------------------------------------------------------
    # Configuración LoRA con Unsloth
    # DIFERENCIA vs PEFT: get_peft_model de Unsloth aplica
    # automáticamente sus kernels optimizados de triton sobre
    # las capas LoRA, sin cambios en la API para el usuario.
    # ---------------------------------------------------------
    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        lora_alpha=alpha,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",  # versión optimizada
        random_state=SEED,
    )
    model.print_trainable_parameters()

    # ---------------------------------------------------------
    # Medición de tiempo y memoria
    # ---------------------------------------------------------
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    # ---------------------------------------------------------
    # SFTTrainer
    # ---------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        args=SFTConfig(
            output_dir=output_dir,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_ratio=0.03,
            learning_rate=lr,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",       # Unsloth usa adamw_8bit
            bf16=True,
            fp16=False,
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="none",
            seed=SEED,
            dataset_text_field="text",   # columna con el prompt
            max_seq_length=MAX_LENGTH,
            packing=False,  # False para comparación justa con PEFT
        ),
    )

    print("Entrenando...")
    trainer.train()

    elapsed   = time.time() - start_time
    peak_vram = torch.cuda.max_memory_allocated() / 1e9

    print(f"⏱  Tiempo de entrenamiento: {elapsed/60:.1f} min")
    print(f"🖥  VRAM pico: {peak_vram:.2f} GB")

    # ---------------------------------------------------------
    # Guardar adaptador
    # Unsloth guarda en formato compatible con HuggingFace,
    # por lo que el adaptador se puede cargar con PEFT después.
    # ---------------------------------------------------------
    adapter_path = os.path.join(output_dir, "adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"✅ Adaptador guardado en {adapter_path}")

    eval_results = trainer.evaluate()
    eval_loss    = eval_results["eval_loss"]

    results.append({
        "run":       run_name,
        "r":         r,
        "alpha":     alpha,
        "lr":        lr,
        "time_min":  round(elapsed / 60, 2),
        "peak_vram": round(peak_vram, 2),
        "eval_loss": round(eval_loss, 4),
    })

    del model, trainer
    torch.cuda.empty_cache()

# =============================================================
# GUARDAR RESULTADOS
# =============================================================
with open(RESULTS_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"\n{'='*60}")
print("RESUMEN GRID SEARCH")
print(f"{'='*60}")
for r in results:
    print(r)

print(f"\n✅ Resultados guardados en {RESULTS_FILE}")