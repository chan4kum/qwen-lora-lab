"""LoRA + training configuration for Qwen2.5-0.5B-Instruct (small CPU-friendly run)."""
from peft import LoraConfig

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_FILE = "data/train.jsonl"
EVAL_FILE = "data/eval.jsonl"
OUTPUT_DIR = "outputs/lora-qwen0.5b"

LORA = LoraConfig(
    r=8,                      # adapter rank: 4.4M trainable params (0.88%) with all 7 targets
    lora_alpha=16,            # scale = alpha / r = 2
    lora_dropout=0.05,        # light regularization for a tiny dataset
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

TRAIN = dict(
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,   # effective batch 8
    learning_rate=2e-4,              # typical LoRA range 1e-4 to 2e-4
    lr_scheduler_type="cosine",
    warmup_steps=1,                 # ~10% of the ~12 total steps
    weight_decay=0.0,
    max_length=512,
    logging_steps=1,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
    seed=42,
    use_cpu=True,                    # Docker container has no GPU
    bf16=False,
)
