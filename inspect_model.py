import os, time, torch
from collections import Counter
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
print(f"== {MODEL} | torch {torch.__version__} | device cpu")

cfg = AutoConfig.from_pretrained(MODEL)
keys = ["model_type", "architectures", "hidden_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "intermediate_size", "vocab_size", "max_position_embeddings", "tie_word_embeddings"]
print("\n== config")
for k in keys:
    print(f"{k}: {getattr(cfg, k, None)}")

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
n = sum(p.numel() for p in model.parameters())
print(f"\n== params: {n/1e6:.1f}M")
print("== layer types:", dict(Counter(type(m).__name__ for m in model.modules()).most_common(8)))
print("== linear modules in block 0:", [n for n, m in model.named_modules()
      if n.startswith("model.layers.0.") and isinstance(m, torch.nn.Linear)])
print("== special tokens: eos", tok.eos_token, "pad", tok.pad_token, "| chat template:", bool(tok.chat_template))

print("\n== generation test")
msgs = [{"role": "user", "content": "What is LoRA fine-tuning in one sentence?"}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
t = time.time()
with torch.no_grad():
    out = model.generate(**ids, max_new_tokens=60, do_sample=False)
new = out[0][ids["input_ids"].shape[1]:]
print(tok.decode(new, skip_special_tokens=True))
print(f"({len(new)} tokens in {time.time()-t:.1f}s)")

print("\n== LoRA wrap test")
lcfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], task_type="CAUSAL_LM")
pm = get_peft_model(model, lcfg)
pm.print_trainable_parameters()
batch = tok("LoRA adds low-rank adapters.", return_tensors="pt")
loss = pm(**batch, labels=batch["input_ids"]).loss
loss.backward()
grads = sum(p.grad is not None for p in pm.parameters() if p.requires_grad)
print(f"forward/backward ok: loss={loss.item():.3f}, adapter tensors with grads={grads}")
