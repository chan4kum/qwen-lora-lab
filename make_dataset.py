"""Builds a small LoRA-knowledge chat dataset (TRL 'messages' format) -> data/train.jsonl, data/eval.jsonl."""
import json, random

QA = [
 ("What does LoRA stand for in machine learning?", "LoRA stands for Low-Rank Adaptation. It is a parameter-efficient fine-tuning method that freezes the pretrained weights and trains small low-rank matrices added to selected layers."),
 ("What is LoRA fine-tuning in one sentence?", "LoRA fine-tuning adapts a large pretrained model by training small low-rank adapter matrices while the original weights stay frozen."),
 ("Who introduced LoRA?", "LoRA was introduced by Edward Hu and colleagues at Microsoft in the 2021 paper 'LoRA: Low-Rank Adaptation of Large Language Models'."),
 ("How does LoRA modify a weight matrix?", "For a frozen weight matrix W, LoRA learns two small matrices A and B and uses W + (alpha / r) * B @ A, where r is the rank. Only A and B are trained."),
 ("What is the rank r in LoRA?", "The rank r is the inner dimension of the two adapter matrices. A smaller r means fewer trainable parameters; a larger r gives the adapter more capacity."),
 ("What does lora_alpha control?", "lora_alpha is a scaling factor. The adapter output is multiplied by alpha divided by r, which controls how strongly the adapter changes the base model's output."),
 ("What are typical values for LoRA rank?", "Common ranks are 8, 16, 32 and 64. Ranks of 8 to 16 are a good starting point for many tasks."),
 ("Why is LoRA cheaper than full fine-tuning?", "LoRA trains only a tiny fraction of the parameters, often under 1 percent. This cuts the memory needed for gradients and optimizer states and makes checkpoints very small."),
 ("Does LoRA change the base model weights?", "No. The base weights stay frozen during training. The adapter weights are saved separately and can be merged into the base model afterwards if you want."),
 ("What is the LoRA dropout parameter?", "lora_dropout applies dropout to the input of the adapter during training. It acts as regularization and helps reduce overfitting on small datasets."),
 ("Which layers are usually targeted by LoRA in a transformer?", "The attention projections q_proj, k_proj, v_proj and o_proj are the most common targets. Many setups also add the MLP layers gate_proj, up_proj and down_proj."),
 ("Can LoRA adapters be merged into the base model?", "Yes. Because the update is B @ A scaled by alpha / r, it can be added to the original weight matrix. The merged model has no extra inference latency."),
 ("What is QLoRA?", "QLoRA combines LoRA with a base model quantized to 4 bits. The frozen base takes much less memory, while LoRA adapters are trained in higher precision."),
 ("Which Python library provides LoRA support for Hugging Face models?", "The PEFT library from Hugging Face provides LoraConfig and get_peft_model for adding LoRA adapters to transformers models."),
 ("Which library is commonly used to run supervised fine-tuning with LoRA?", "TRL, the Transformer Reinforcement Learning library, provides SFTTrainer, which works with PEFT LoRA configs."),
 ("How do I add LoRA to a model with PEFT?", "Create a LoraConfig with r, lora_alpha, target_modules and task_type, then call get_peft_model(model, config). The returned model trains only the adapter weights."),
 ("What is the task_type for causal language models in PEFT?", "Use task_type='CAUSAL_LM' in LoraConfig for decoder-only causal language models."),
 ("What does print_trainable_parameters show?", "It prints how many parameters are trainable, the total parameter count, and the trainable percentage. For LoRA the percentage is typically below 1 percent."),
 ("How large is a LoRA adapter checkpoint?", "It is usually only a few megabytes to a few hundred megabytes, depending on rank and target modules, compared with gigabytes for a full model."),
 ("Can several LoRA adapters share one base model?", "Yes. You can keep one frozen base model and load different adapters for different tasks, switching between them at inference time."),
 ("Is LoRA good for teaching a model new facts?", "LoRA works best for adapting style, format and task behavior. It can add some knowledge, but it is less reliable than retrieval for large or frequently changing facts."),
 ("What learning rate is typical for LoRA?", "LoRA commonly uses a higher learning rate than full fine-tuning, often around 1e-4 to 2e-4."),
 ("Why can a LoRA run overfit on a small dataset?", "With few examples the adapter can memorize them. Lower the rank, add lora_dropout, train fewer epochs, and keep a held-out evaluation set."),
 ("What is the difference between LoRA and full fine-tuning?", "Full fine-tuning updates every weight in the model. LoRA freezes the weights and trains small low-rank adapters, which uses far less memory and storage."),
 ("What does gradient checkpointing do?", "It saves memory by discarding some activations in the forward pass and recomputing them during the backward pass, at the cost of extra compute."),
 ("What is a chat template?", "A chat template is the rule that turns a list of role-tagged messages into the exact token format a model was trained on. Use tokenizer.apply_chat_template to apply it."),
 ("Why should fine-tuning data match the model's chat template?", "The model learned a specific format for system, user and assistant turns. Training data in a different format can degrade results, so use the model's own template."),
 ("What is an epoch?", "An epoch is one full pass of the training algorithm over the entire training dataset."),
 ("What is a held-out evaluation set?", "It is data kept out of training and used only to measure how well the model generalizes to examples it has not seen."),
 ("What is perplexity?", "Perplexity is the exponential of the average cross-entropy loss. Lower perplexity means the model finds the text more predictable."),
 ("What is the purpose of lora_alpha divided by r?", "It sets the scale of the adapter update. Keeping alpha proportional to r makes training behave similarly when you change the rank."),
 ("What does it mean to freeze a layer?", "Freezing a layer sets requires_grad to False for its parameters, so gradient updates do not change them."),
 ("What is an optimizer state and why does it matter for memory?", "Optimizers like Adam keep extra tensors per trainable parameter, such as running averages. Training fewer parameters with LoRA greatly reduces this memory."),
 ("How can I load a saved LoRA adapter for inference?", "Load the base model, then call PeftModel.from_pretrained(base_model, adapter_path). You can optionally call merge_and_unload() to fold the adapter into the base weights."),
 ("What is an adapter in the context of fine-tuning?", "An adapter is a small set of extra trainable weights attached to a frozen pretrained model, so the model can be specialized without changing its original weights."),
 ("Does LoRA increase inference latency?", "Not after merging. If the adapter is kept separate it adds a small extra computation per adapted layer."),
 ("What does SFT stand for?", "SFT stands for supervised fine-tuning, which trains a model on input and desired-output examples."),
 ("What is the Qwen2.5-0.5B-Instruct model?", "It is a small instruction-tuned language model from Alibaba Cloud's Qwen2.5 family with about 0.5 billion parameters."),
 ("Answer in one word: what is the first letter of the word LoRA?", "L"),
]

SYSTEM = "You are a concise machine learning assistant."
random.Random(0).shuffle(QA)
EVAL_N = 8
splits = {"eval": QA[:EVAL_N], "train": QA[EVAL_N:]}
for name, rows in splits.items():
    with open(f"data/{name}.jsonl", "w") as f:
        for q, a in rows:
            f.write(json.dumps({"messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}]}) + "\n")
    print(name, len(rows))
