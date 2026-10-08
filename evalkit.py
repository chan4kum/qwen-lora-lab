import json, re, torch
from collections import Counter

def load_rows(path):
    return [json.loads(l)["messages"] for l in open(path)]

def _prompt_ids(tok, msgs):
    return tok.apply_chat_template(msgs[:-1], add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"]

@torch.no_grad()
def answer_loss(model, tok, rows):
    """Mean cross-entropy on assistant tokens only (the part we train for)."""
    tot, n = 0.0, 0
    for msgs in rows:
        p = _prompt_ids(tok, msgs)
        a = tok(msgs[-1]["content"] + "<|im_end|>", return_tensors="pt", add_special_tokens=False)["input_ids"]
        ids = torch.cat([p, a], 1)
        labels = ids.clone(); labels[:, : p.shape[1]] = -100
        out = model(input_ids=ids, labels=labels)
        k = a.shape[1]
        tot += out.loss.item() * k; n += k
    return tot / n

@torch.no_grad()
def generate(model, tok, msgs, n=80):
    p = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True, return_tensors="pt", return_dict=True)
    out = model.generate(**p, max_new_tokens=n, do_sample=False)
    return tok.decode(out[0][p["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def f1(pred, ref):
    t = lambda s: re.findall(r"\w+", s.lower())
    c = Counter(t(pred)) & Counter(t(ref))
    s = sum(c.values())
    if not s: return 0.0
    pr, rc = s / len(t(pred)), s / len(t(ref))
    return 2 * pr * rc / (pr + rc)

def evaluate(model, tok, rows, gen=True):
    model.eval()
    res = {"loss": answer_loss(model, tok, rows)}
    if gen:
        outs = [generate(model, tok, m) for m in rows]
        res["f1"] = sum(f1(o, m[-1]["content"]) for o, m in zip(outs, rows)) / len(rows)
        res["outputs"] = outs
    return res
