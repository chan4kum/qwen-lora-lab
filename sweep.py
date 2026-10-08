import json, torch, time, copy
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from lora_config import BASE_MODEL, TRAIN_FILE, EVAL_FILE, LORA, TRAIN
from evalkit import load_rows, evaluate

tok = AutoTokenizer.from_pretrained(BASE_MODEL)
train_rows, eval_rows = load_rows(TRAIN_FILE), load_rows(EVAL_FILE)

RUNS = [  # name, rank, lr, n_train
    ("r4",       4,  2e-4, 31),
    ("r32",     32,  2e-4, 31),
    ("lr5e-5",   8,  5e-5, 31),
    ("lr5e-4",   8,  5e-4, 31),
    ("n15",      8,  2e-4, 15),
]
results = []
for name, r, lr, n in RUNS:
    t0 = time.time()
    cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=LORA.lora_dropout, target_modules=LORA.target_modules, task_type="CAUSAL_LM")
    args = {**TRAIN, "learning_rate": lr, "eval_strategy": "no", "save_strategy": "no", "logging_steps": 100, "disable_tqdm": True}
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
    tr = SFTTrainer(model=model, args=SFTConfig(output_dir=f"outputs/sweep-{name}", **args),
                    train_dataset=Dataset.from_list([{"messages": m} for m in train_rows[:n]]),
                    processing_class=tok, peft_config=cfg)
    tr.train()
    res = evaluate(tr.model, tok, eval_rows)
    row = dict(name=name, rank=r, lr=lr, n_train=n, eval_loss=round(res["loss"], 3), f1=round(res["f1"], 3), train_loss=round(tr.state.log_history[-1].get("train_loss", float("nan")), 3), secs=int(time.time() - t0))
    print("RESULT", json.dumps(row), flush=True)
    results.append(row)
    json.dump(results, open("outputs/sweep_results.json", "w"), indent=1)
