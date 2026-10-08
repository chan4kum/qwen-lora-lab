import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from lora_config import BASE_MODEL, TRAIN_FILE, EVAL_FILE, OUTPUT_DIR, LORA, TRAIN

tok = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
ds = load_dataset("json", data_files={"train": TRAIN_FILE, "eval": EVAL_FILE})

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(output_dir=OUTPUT_DIR, **TRAIN),
    train_dataset=ds["train"],
    eval_dataset=ds["eval"],
    processing_class=tok,
    peft_config=LORA,
)
trainer.model.print_trainable_parameters()
trainer.train()
print("final eval:", trainer.evaluate())
trainer.save_model(OUTPUT_DIR)
print("saved adapter to", OUTPUT_DIR)
