# src/covid_nlp_light.py
from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class P:
    root: Path = Path(__file__).resolve().parents[1]
    data: Path = root / "data"
    raw: Path = data / "raw"
    cache: Path = data / "cache"
    proc: Path = data / "processed"
    models: Path = root / "models"

def seed_everything(seed: int = 42):
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def build_datasets(raw_csv, model_name="vinai/bertweet-base", test_size=0.2, seed=42):
    import pandas as pd
    from datasets import Dataset, DatasetDict
    from sklearn.model_selection import train_test_split
    from transformers import AutoTokenizer
    seed_everything(seed)
    df = pd.read_csv(raw_csv)
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["label"])
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    def tok_fn(batch): return tok(batch["text"], truncation=True, padding=False)
    dtrain = Dataset.from_pandas(train_df, preserve_index=False).map(tok_fn, batched=True)
    dval   = Dataset.from_pandas(val_df,   preserve_index=False).map(tok_fn, batched=True)
    dd = DatasetDict(train=dtrain, validation=dval); dd = dd.rename_column("label","labels")
    dd.set_format(type="torch", columns=["input_ids","attention_mask","labels"])
    return dd, tok

def train_with_trainer(datasets, model_name="vinai/bertweet-base", out_dir=str(P.models / "bertweet_trainer"),
                       lr=2e-5, wd=0.01, bs=32, epochs=3, seed=42, wandb=False, wandb_entity=None, run_name="trainer"):
    from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
    seed_everything(seed)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    args = TrainingArguments(output_dir=out_dir, learning_rate=lr, weight_decay=wd,
                             per_device_train_batch_size=bs, per_device_eval_batch_size=bs,
                             num_train_epochs=epochs, evaluation_strategy="epoch", save_strategy="epoch",
                             report_to=("wandb" if wandb else []), run_name=run_name, load_best_model_at_end=True)
    def metrics(eval_pred):
        import evaluate, numpy as np
        acc = evaluate.load("accuracy")
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        return acc.compute(predictions=preds, references=labels)
    trainer = Trainer(model=model, args=args, train_dataset=datasets["train"], eval_dataset=datasets["validation"],
                      tokenizer=None, compute_metrics=metrics)
    trainer.train(); trainer.evaluate(); trainer.save_model(out_dir)
    return model

def train_manual(datasets, model_name="vinai/bertweet-base", out_dir=str(P.models / "bertweet_manual"),
                 lr=2e-5, wd=0.0, epochs=3, seed=42):
    import torch
    from torch.optim import AdamW
    from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding
    seed_everything(seed)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    dl_train = torch.utils.data.DataLoader(datasets["train"], batch_size=32, shuffle=True,
                                           collate_fn=DataCollatorWithPadding(tokenizer=None))
    dl_val   = torch.utils.data.DataLoader(datasets["validation"], batch_size=64,
                                           collate_fn=DataCollatorWithPadding(tokenizer=None))
    opt = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    model.train()
    for _ in range(epochs):
        for b in dl_train:
            b = {k:v.to(device) for k,v in b.items()}
            loss = model(**b).loss; loss.backward(); opt.step(); opt.zero_grad()
    Path(out_dir).mkdir(parents=True, exist_ok=True); model.save_pretrained(out_dir)
    return model

def compress(model_dir, method="distill", student="hf-internal-testing/tiny-random-RoBERTa",
             out_dir=str(P.models / "compressed")):
    from pathlib import Path
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # Minimal placeholder so your notebook calls are tidy; plug your existing code here if desired.
    return {"status": "ok", "method": method, "out": out_dir}
