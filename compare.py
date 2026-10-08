import torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from lora_config import BASE_MODEL, OUTPUT_DIR, EVAL_FILE
from evalkit import load_rows, evaluate

tok = AutoTokenizer.from_pretrained(BASE_MODEL)
rows = load_rows(EVAL_FILE)

base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
rb = evaluate(base, tok, rows)

# reload the saved adapter from disk onto a fresh base model
tuned = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32), OUTPUT_DIR)
rt = evaluate(tuned, tok, rows)
print(f"reloaded adapter from {OUTPUT_DIR}")
print(f"BASE   loss={rb['loss']:.3f}  F1={rb['f1']:.3f}")
print(f"TUNED  loss={rt['loss']:.3f}  F1={rt['f1']:.3f}")
for m, b, t in zip(rows, rb["outputs"], rt["outputs"]):
    print(f"\nQ: {m[1]['content']}\nREF:   {m[2]['content']}\nBASE:  {b[:200]}\nTUNED: {t[:200]}")

q = [{"role": "user", "content": "What does LoRA stand for in machine learning?"}, {"role": "assistant", "content": ""}]
print("\nPROBE base :", evaluate(base, tok, [q])["outputs"][0][:150])
print("PROBE tuned:", evaluate(tuned, tok, [q])["outputs"][0][:150])
