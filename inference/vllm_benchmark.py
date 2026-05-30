import os
import time
import torch
import pandas as pd

from datasets import load_dataset
from vllm import LLM, SamplingParams

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "./outputs/peft/r32_a64_lr00002/adapter"
MERGED_PATH = "./outputs/merged/peft_r32"

MODELS = {
    "base": BASE_MODEL,
    "peft_r32": MERGED_PATH,
}


def merge_and_save():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    print("Merging adapter into base model...")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model = model.merge_and_unload()

    os.makedirs(MERGED_PATH, exist_ok=True)
    model.save_pretrained(MERGED_PATH)
    tokenizer.save_pretrained(MERGED_PATH)

    print(f"Saved merged model to {MERGED_PATH}")

    del model, base, tokenizer
    torch.cuda.empty_cache()


def main():
    if not os.path.exists(MERGED_PATH):
        merge_and_save()

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=128,
    )

    dataset = load_dataset("databricks/databricks-dolly-15k", split="train").shuffle(seed=42).select(range(20))
    prompts = [ex["instruction"] for ex in dataset]

    results = []

    for name, model_path in MODELS.items():
        print(f"\nBenchmarking {name}")

        llm = LLM(model=model_path, dtype="float16", gpu_memory_utilization=0.8)
        peak_mem = torch.cuda.memory_allocated() / (1024 ** 3)

        latencies = []
        throughputs = []

        for prompt in prompts:
            start = time.time()
            output = llm.generate([prompt], sampling_params)
            end = time.time()

            gen_tokens = len(output[0].outputs[0].token_ids)
            latency = end - start

            latencies.append(latency)
            throughputs.append(gen_tokens / latency)

        results.append({
            "model": name,
            "latency_s": sum(latencies) / len(latencies),
            "throughput_tok_s": sum(throughputs) / len(throughputs),
            "peak_vram_gb": peak_mem,
        })

        del llm
        torch.cuda.empty_cache()

    os.makedirs("./results", exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv("./results/vllm_benchmark.csv", index=False)

    print(df)


if __name__ == "__main__":
    main()
