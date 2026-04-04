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
CUDA_VISIBLE_DEVICES=0 python finetuning/unsloth_finetuning.py
```
To avoid an excesive number of files, the checkpoints are not in this repository. They will generate automatically running the code. The only uploaded file is the csv with the memory and time results for each model.
