import time
import torch
import pandas as pd

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

MODELS = {
    "unsloth_r8": "./outputs/unsloth/r8_a16_lr00002/adapter",
    "peft_r32": "./outputs/peft/r32_a32_lr00002/adapter",
}

DEVICE = "cuda"

def load_model(adapter_path):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    if adapter_path:
        model = PeftModel.from_pretrained(base, adapter_path)
    else:
        model = base

    model.eval()
    return model, tokenizer

def benchmark(model, tokenizer, prompt, runs=3):
    times = []
    token_counts = []

    for _ in range(runs):
        torch.cuda.reset_peak_memory_stats()

        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        start = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        end = time.time()

        gen_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]

        times.append(end - start)
        token_counts.append(gen_tokens)

    latency = sum(times) / len(times)
    throughput = sum(token_counts) / sum(times)
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)

    return latency, throughput, peak_mem


dataset = load_dataset("databricks/databricks-dolly-15k", split="train").shuffle(seed=42).select(range(20))

results = []

for name, path in MODELS.items():
    print(f"Benchmarking {name}")

    model, tokenizer = load_model(path)

    latencies = []
    throughputs = []
    memories = []

    for ex in dataset:
        prompt = ex["instruction"]

        lat, thr, mem = benchmark(model, tokenizer, prompt)

        latencies.append(lat)
        throughputs.append(thr)
        memories.append(mem)

    results.append({
        "model": name,
        "latency_s": sum(latencies)/len(latencies),
        "throughput_tok_s": sum(throughputs)/len(throughputs),
        "peak_vram_gb": sum(memories)/len(memories),
    })

df = pd.DataFrame(results)
df.to_csv("transformers_benchmark.csv", index=False)

print(df)