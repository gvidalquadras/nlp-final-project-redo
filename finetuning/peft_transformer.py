# =============================================================
# FINE-TUNING CON TRANSFORMERS + PEFT (QLoRA)
# Modelo: TinyLlama-1.1B-Chat
# Dataset: Databricks Dolly 15k
# =============================================================

import os
import time
import torch
import csv
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
import itertools

# =============================================================
# CONFIGURACIÓN GENERAL
# =============================================================
MODEL_NAME   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_DIR  = "./datasets"
OUTPUT_BASE  = "./outputs/peft"
RESULTS_FILE = "./outputs/peft/grid_search_results.csv"
SEED         = 42

# Undersampling para compatibilidad con hardware.
# TRAIN_SAMPLES = 3000
# DEV_SAMPLES   = 500

# # Número de epochs
# NUM_EPOCHS = 3
TRAIN_SAMPLES = 200
DEV_SAMPLES   = 50
NUM_EPOCHS    = 1

# Longitud máxima de secuencia
MAX_LENGTH = 512

# =============================================================
# VALORES DE GRID SEARCH
# =============================================================
LORA_RANKS      = [8, 16, 32]
LEARNING_RATES  = [1e-4, 2e-4]
ALPHA_FACTORS   = [1, 2]
GRID            = list(itertools.product(LORA_RANKS, ALPHA_FACTORS, LEARNING_RATES))

# Hiperparámetros fijos de LoRA
LORA_DROPOUT    = 0.05
TARGET_MODULES  = ["q_proj", "v_proj", "k_proj", "o_proj"]

os.makedirs(OUTPUT_BASE, exist_ok=True)

# =============================================================
# TOKENIZER
# =============================================================
print("Cargando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # necesario para causal LM

# =============================================================
# FORMATO DE PROMPT
# =============================================================
def format_prompt(example):
    if example.get("context", "").strip():
        prompt = (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Context:\n{example['context']}\n\n"
            f"### Response:\n{example['response']}"
        )
    else:
        prompt = (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Response:\n{example['response']}"
        )
    return prompt

def tokenize(example):
    text = format_prompt(example)
    result = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

# =============================================================
# CARGA Y PREPROCESAMIENTO DEL DATASET
# =============================================================
print("Cargando dataset...")
train_ds = load_from_disk(os.path.join(DATASET_DIR, "train"))
dev_ds   = load_from_disk(os.path.join(DATASET_DIR, "dev"))

# Undersampling reproducible
train_ds = train_ds.shuffle(seed=SEED).select(range(TRAIN_SAMPLES))
dev_ds   = dev_ds.shuffle(seed=SEED).select(range(DEV_SAMPLES))

print("Tokenizando...")
train_ds = train_ds.map(tokenize, remove_columns=train_ds.column_names)
dev_ds   = dev_ds.map(tokenize,   remove_columns=dev_ds.column_names)

train_ds.set_format("torch")
dev_ds.set_format("torch")

# =============================================================
# CONFIGURACIÓN DE CUANTIZACIÓN 4-BIT
# =============================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

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

    
    # Carga del modelo base (se recarga en cada experimento para partir siempre 
    # del mismo punto)
    print("Cargando modelo base...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": 0},
        dtype=torch.bfloat16,   # también corrige el warning de torch_dtype
    )
    model.config.use_cache = False
    model.enable_input_require_grads()  # necesario para gradient_checkpointing con QLoRA

    # ---------------------------------------------------------
    # Configuración LoRA
    # ---------------------------------------------------------
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",  # no entrenar bias (estándar en LoRA)
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---------------------------------------------------------
    # Training Arguments
    # ---------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_ratio=0.03,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",  # desactiva wandb/tensorboard
        seed=SEED,
        run_name=run_name,
    )

    # ---------------------------------------------------------
    # Medición de tiempo y memoria
    # ---------------------------------------------------------
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt"
        ),
    )

    print("Entrenando...")
    trainer.train()

    elapsed   = time.time() - start_time
    peak_vram = torch.cuda.max_memory_allocated() / 1e9  # GB

    print(f"Tiempo de entrenamiento: {elapsed/60:.1f} min")
    print(f"VRAM pico: {peak_vram:.2f} GB")

    # ---------------------------------------------------------
    # Guardar el adaptador LoRA (no el modelo completo)
    # ---------------------------------------------------------
    adapter_path = os.path.join(output_dir, "adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"Adaptador guardado en {adapter_path}")

    # Obtener eval_loss final
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

    # Liberar memoria antes del siguiente experimento
    del model, trainer
    torch.cuda.empty_cache()

# =============================================================
# GUARDAR RESULTADOS DEL GRID SEARCH
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

print(f"\nResultados guardados en {RESULTS_FILE}")