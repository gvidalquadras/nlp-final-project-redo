import os
import time
import psutil
import requests
import pandas as pd
from datasets import load_dataset

URL = "http://localhost:11434/api/generate"
MODEL = "tinyllama"

dataset = load_dataset("databricks/databricks-dolly-15k", split="train").shuffle(seed=42).select(range(20))

latencies = []
throughputs = []
peak_memories_gb = []

process = psutil.Process()

for ex in dataset:
    prompt = ex["instruction"]

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 128,
        },
    }

    mem_before = process.memory_info().rss

    start = time.time()
    r = requests.post(URL, json=payload)
    end = time.time()

    mem_after = process.memory_info().rss

    data = r.json()

    latency = end - start
    eval_count = data.get("eval_count", 0)
    eval_duration_s = data.get("eval_duration", 1) / 1e9

    throughput = eval_count / eval_duration_s if eval_duration_s > 0 else 0
    peak_mem_gb = max(mem_before, mem_after) / (1024 ** 3)

    latencies.append(latency)
    throughputs.append(throughput)
    peak_memories_gb.append(peak_mem_gb)

os.makedirs("./results", exist_ok=True)

df = pd.DataFrame([{
    "model": MODEL,
    "latency_s": sum(latencies) / len(latencies),
    "throughput_tok_s": sum(throughputs) / len(throughputs),
    "peak_ram_gb": sum(peak_memories_gb) / len(peak_memories_gb),
}])

df.to_csv("./results/ollama_benchmark.csv", index=False)

print(df)
