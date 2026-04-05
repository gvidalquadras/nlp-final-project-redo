import requests
import time
import pandas as pd
from datasets import load_from_disk

URL = "http://localhost:11434/api/generate"

MODEL = "tinyllama"

dataset = load_from_disk("./datasets/train").shuffle(seed=42).select(range(20))
latencies = []

for ex in dataset:
    prompt = ex["instruction"]

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "options": {
            "temperature": 0.7,
            "num_predict": 128
        }
    }

    start = time.time()
    r = requests.post(URL, json=payload)
    end = time.time()

    latencies.append(end - start)

print("Avg latency:", sum(latencies)/len(latencies))