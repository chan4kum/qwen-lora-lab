import os, time, math, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()

print("== chat template rendering")
print(tok.apply_chat_template([{"role": "user", "content": "Hi"}], tokenize=False, add_generation_prompt=True))

print("== tokenizer")
for s in ["Hello world", "LoRA fine-tuning", "def f(x): return x**2", "नमस्ते"]:
    ids = tok(s)["input_ids"]
    print(f"{s!r}: {len(ids)} tokens, roundtrip ok={tok.decode(ids)==s}")

def ask(q, n=80):
    ids = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, return_tensors="pt", return_dict=True)
    t = time.time()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=n, do_sample=False)
    new = out[0][ids["input_ids"].shape[1]:]
    return tok.decode(new, skip_special_tokens=True).strip(), len(new) / (time.time() - t)

print("\n== behaviour probes")
probes = {
  "factual": "What is the capital of France? Answer in one word.",
  "arithmetic": "What is 17 * 23? Give only the number.",
  "format": 'Return JSON with keys "name" and "age" for a person called Ana aged 30. JSON only.',
  "code": "Write a Python function that reverses a string. Code only.",
  "reasoning": "If all bloops are razzies and all razzies are lazzies, are all bloops lazzies? Answer yes or no.",
  "domain (LoRA)": "What does LoRA stand for in machine learning?",
}
for k, q in probes.items():
    a, tps = ask(q)
    print(f"[{k}] ({tps:.1f} tok/s)\n{a}\n")

print("== perplexity (lower = model finds text more natural)")
texts = {
  "english": "The quick brown fox jumps over the lazy dog. Machine learning models learn patterns from data.",
  "shuffled": "dog lazy the over jumps fox brown quick The. data from patterns learn models learning Machine.",
}
for k, t in texts.items():
    b = tok(t, return_tensors="pt")
    with torch.no_grad():
        loss = model(**b, labels=b["input_ids"]).loss.item()
    print(f"{k}: ppl={math.exp(loss):.1f}")
