# =============================================================
# FULL EVALUATION PIPELINE (CLEAN + FIXED CSV OUTPUTS)
# =============================================================

import os
import torch
import pandas as pd
import random
import csv

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bertscore

# =============================================================
# CONFIG
# =============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_NAME = "databricks/databricks-dolly-15k"
NUM_SAMPLES = 50
SEED = 42

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

MODEL_PATHS = {
    "base": None,
    "unsloth_r8": "./outputs/unsloth/r8_a16_lr00002/adapter",
    "peft_r32": "./outputs/peft/r32_a32_lr00002/adapter",
}

OUTPUT_DIR = "./model_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================
# CLEANING FUNCTION (IMPORTANT)
# =============================================================

def clean_text(text):
    if text is None:
        return ""
    # Remove newlines and carriage returns to keep CSV structure intact
    text = text.replace("\n", " ").replace("\r", " ")
    return text.strip()

# =============================================================
# LOAD DATASET
# =============================================================

print("Loading dataset...")
dataset = load_dataset(DATASET_NAME, split="train")
dataset = dataset.shuffle(seed=SEED).select(range(NUM_SAMPLES))

# =============================================================
# PROMPT FORMAT
# =============================================================

def format_prompt(example):
    if example.get("context", "").strip():
        return (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Context:\n{example['context']}\n\n"
            f"### Response:\n"
        )
    else:
        return (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Response:\n"
        )

# =============================================================
# LOAD MODEL
# =============================================================

def load_model(adapter_path):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    if adapter_path is not None:
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        model = base_model

    model.eval()
    return model, tokenizer

# =============================================================
# GENERATION
# =============================================================

def generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prediction = decoded[len(prompt):].strip()

    return clean_text(prediction)

# =============================================================
# RUN INFERENCE
# =============================================================

all_outputs = {}

for model_name, path in MODEL_PATHS.items():
    print(f"\nRunning inference for {model_name}...")

    model, tokenizer = load_model(path)

    rows = []

    for ex in dataset:
        prompt = format_prompt(ex)
        prediction = generate(model, tokenizer, prompt)

        rows.append({
            "instruction": clean_text(ex["instruction"]),
            "reference": clean_text(ex["response"]),
            model_name: prediction
        })

    df = pd.DataFrame(rows)
    all_outputs[model_name] = df

# =============================================================
# MERGE OUTPUTS
# =============================================================

print("\nMerging outputs...")

base_df = all_outputs["base"][["instruction", "reference"]].copy()

for model_name, df in all_outputs.items():
    base_df[model_name] = df[model_name]

combined_csv = os.path.join(OUTPUT_DIR, "combined_outputs.csv")

base_df.to_csv(
    combined_csv,
    index=False,
    quoting=csv.QUOTE_ALL,
    escapechar="\\"
)

print(f"Saved combined CSV to {combined_csv}")

# =============================================================
# AUTOMATIC METRICS
# =============================================================

print("\nComputing automatic metrics...")

smooth = SmoothingFunction().method1
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

results = []

for model_name in MODEL_PATHS.keys():
    references = base_df["reference"].tolist()
    predictions = base_df[model_name].tolist()

    bleu_scores = []
    rouge_scores = []

    for ref, pred in zip(references, predictions):
        ref_tokens = ref.split()
        pred_tokens = pred.split()

        bleu_scores.append(
            sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth)
        )

        rouge_scores.append(
            rouge.score(ref, pred)["rougeL"].fmeasure
        )

    P, R, F1 = bertscore(predictions, references, lang="en")

    results.append({
        "model": model_name,
        "BLEU": sum(bleu_scores) / len(bleu_scores),
        "ROUGE-L": sum(rouge_scores) / len(rouge_scores),
        "BERTScore": F1.mean().item()
    })

results_df = pd.DataFrame(results)

results_df.to_csv(
    "evaluation_metrics.csv",
    index=False,
    quoting=csv.QUOTE_ALL
)

print("\n=== AUTOMATIC METRICS ===")
print(results_df)

# =============================================================
# HUMAN EVALUATION TABLE
# =============================================================

print("\nCreating human evaluation table...")

sample_indices = random.sample(range(NUM_SAMPLES), 5)

human_rows = []

for idx in sample_indices:
    row = {
        "instruction": base_df.iloc[idx]["instruction"],
        "reference": base_df.iloc[idx]["reference"]
    }

    for model_name in MODEL_PATHS.keys():
        row[model_name] = base_df.iloc[idx][model_name]

        row[f"{model_name}_helpfulness (1-5)"] = ""
        row[f"{model_name}_factuality (1-5)"] = ""
        row[f"{model_name}_instruction_following (1-5)"] = ""

    human_rows.append(row)

human_df = pd.DataFrame(human_rows)

human_df.to_csv(
    "human_evaluation_table.csv",
    index=False,
    quoting=csv.QUOTE_ALL
)

print("\nSaved human_evaluation_table.csv")
print("Pipeline complete.")