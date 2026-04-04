# nlp-final-project-redo

1. install dependencies
```bash
pip install -r requirements.txt
```

2. Load data
```bash 
python data/load_data.py
```

3. Run finetuning
```bash
CUDA_VISIBLE_DEVICES=0 python finetuning/peft_transformer.py # to avoid errors related to multiple GPUs
python finetuning/unsloth_finetuning.py
```