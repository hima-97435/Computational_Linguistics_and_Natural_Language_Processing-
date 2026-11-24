"""Name That LLM - iteration 2: Qwen3-8B-Base + LoRA multi-layer reader, two-stage (Open Pool -> competition train).

  python train_lora.py --fold 0      # hold out fold 0 of the prompt-grouped CV (same split as ntl_baseline.py), report bacc
  python train_lora.py --fold -1     # fit on all of train (for the submission), save adapter
Stage 1 trains on a class-balanced sample of the Open Pool (no train prompts in it), stage 2 on the competition train folds.
"""
import argparse, json, math, os, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score
from transformers import AutoConfig, AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model

LABELS = np.array(sorted(['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
                          'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
                          'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
                          'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3']))
L2I = {l: i for i, l in enumerate(LABELS)}

ap = argparse.ArgumentParser()
ap.add_argument('--fold', type=int, default=0)
ap.add_argument('--model', default=os.path.expanduser('~/ntl/models/Qwen3-8B-Base'))
ap.add_argument('--data', default=os.path.expanduser('~/ntl/data'))
ap.add_argument('--out', default=os.path.expanduser('~/ntl/runs'))
ap.add_argument('--pool_per_class', type=int, default=4000)  # stage-1 size = 18 x this (classes with fewer rows give all); 0 = full pool
ap.add_argument('--epochs2', type=int, default=2)
ap.add_argument('--lr1', type=float, default=1e-4)
ap.add_argument('--lr2', type=float, default=5e-5)
ap.add_argument('--bs', type=int, default=16)
ap.add_argument('--max_len', type=int, default=384)
ap.add_argument('--r', type=int, default=16)
ap.add_argument('--keep', type=int, default=24)  # truncate the LLM to its first `keep` layers (mid layers carry the style signal)
ap.add_argument('--layers', default='2,4,6,8,12,16,20,24')  # hidden states read by the head (mean-pooled, layer-normed, concatenated)
ap.add_argument('--tag', default='')  # suffix for the run dir
ap.add_argument('--smoke', action='store_true')  # tiny random 2-layer model, few rows, CPU ok: exercises every code path
args = ap.parse_args()
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
assert DEV == 'cuda' or args.smoke, 'no CUDA'
torch.manual_seed(0)
run = Path(args.out) / ((f'fold{args.fold}' if args.fold >= 0 else 'full') + args.tag + ('_smoke' if args.smoke else '')); run.mkdir(parents=True, exist_ok=True)
T0 = time.time()
def log(m): print(f'[{(time.time() - T0) / 60:6.1f}m] {m}', flush=True)

# ---------------------------------------------------------------- data
train = pd.read_csv(f'{args.data}/train.csv'); train['text'] = train.text.fillna('')
y = train.model.map(L2I).to_numpy()
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(train, y, train.prompt_id))
tr_idx, va_idx = folds[args.fold] if args.fold >= 0 else (np.arange(len(train)), np.array([], int))
pool = pd.read_parquet(f'{args.data}/pool/ntl_open_pool.parquet', columns=['text', 'model'])
pool = pool.sample(frac=1, random_state=0)
if args.pool_per_class:
    pool = pool.groupby('model').head(args.pool_per_class)
if args.smoke:
    pool, tr_idx, va_idx = pool.head(40), tr_idx[:16], va_idx[:16]
log(f'fold {args.fold}: train {len(tr_idx)} holdout {len(va_idx)} pool-stage1 {len(pool)} | {torch.cuda.get_device_name() if DEV == "cuda" else "cpu"}')

tok = AutoTokenizer.from_pretrained(args.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = 'right'
def encode(texts):
    return tok(list(texts), add_special_tokens=False, truncation=True, max_length=args.max_len)['input_ids']

def batches(ids, labels, bs, shuffle):
    order = np.random.permutation(len(ids)) if shuffle else np.argsort([len(x) for x in ids])
    for s in range(0, len(order), bs):
        b = order[s:s + bs]; L = max(len(ids[i]) for i in b)
        x = torch.full((len(b), L), tok.pad_token_id); m = torch.zeros((len(b), L), dtype=torch.long)
        for j, i in enumerate(b):
            x[j, :len(ids[i])] = torch.tensor(ids[i]); m[j, :len(ids[i])] = 1
        yield b, x.to(DEV), m.to(DEV), (torch.tensor(labels[b]).to(DEV) if labels is not None else None)

# ---------------------------------------------------------------- model
# Multi-layer reader (the "l24" design of the public 0.70 stacks): truncated LLM + LoRA, masked mean of several hidden layers.
if args.smoke:
    cfg = AutoConfig.from_pretrained(args.model, num_hidden_layers=4, hidden_size=64, intermediate_size=128,
                                     num_attention_heads=4, num_key_value_heads=2, head_dim=16, use_cache=False)
    LAYERS = [2, 4]
    base = AutoModel.from_config(cfg, dtype=torch.bfloat16).to(DEV)
else:
    cfg = AutoConfig.from_pretrained(args.model, num_hidden_layers=args.keep, use_cache=False)
    LAYERS = [int(l) for l in args.layers.split(',') if int(l) <= args.keep]
    base = AutoModel.from_pretrained(args.model, config=cfg, dtype=torch.bfloat16).to(DEV)
base.gradient_checkpointing_enable(); base.enable_input_require_grads()
base = get_peft_model(base, LoraConfig(r=args.r, lora_alpha=2 * args.r, lora_dropout=0.05,
                                       target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))

class Head(torch.nn.Module):
    def __init__(self, hid, n):
        super().__init__()
        self.norms = torch.nn.ModuleList(torch.nn.LayerNorm(hid) for _ in LAYERS)
        self.drop, self.out = torch.nn.Dropout(0.1), torch.nn.Linear(hid * len(LAYERS), n)
    def forward(self, hs, m):
        m = m.unsqueeze(-1).float(); m[:, 0] = 0  # skip the first token (attention sink)
        z = [nm((hs[l].float() * m).sum(1) / m.sum(1).clamp(min=1)) for l, nm in zip(LAYERS, self.norms)]
        return self.out(self.drop(torch.cat(z, -1)))
head = Head(cfg.hidden_size, len(LABELS)).to(DEV)
class Reader(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.base, self.head = base, head
    def forward(self, input_ids, attention_mask):
        hs = self.base(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states
        return self.head(hs, attention_mask)
model = Reader()
log(f'layers {LAYERS} of {cfg.num_hidden_layers}')
for p_ in model.parameters():  # fp32 master weights for LoRA + head (bf16 updates at lr 5e-5 mostly round away); bf16 autocast in forward
    if p_.requires_grad:
        p_.data = p_.data.float()
base.print_trainable_parameters()
log(f'trainable dtypes {sorted({str(p_.dtype) for p_ in model.parameters() if p_.requires_grad})}')
def fwd(x, m):
    with torch.autocast(DEV, dtype=torch.bfloat16):
        return model(x, m).float()

def train_stage(texts, labels, epochs, lr, name):
    ids = encode(texts); steps = epochs * math.ceil(len(ids) / args.bs); warm = max(1, steps // 20)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * max(0.0, 0.5 * (1 + math.cos(math.pi * s / steps))))
    model.train(); step, run_loss = 0, 0.0
    for ep in range(epochs):
        for _, x, m, yb in batches(ids, labels, args.bs, True):
            loss = torch.nn.functional.cross_entropy(fwd(x, m), yb)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1; run_loss = 0.98 * run_loss + 0.02 * loss.item()
            if step % 200 == 0 or (args.smoke and step % 2 == 0):
                log(f'{name} ep{ep} step {step}/{steps} loss {run_loss:.3f}')
        if len(va_idx):
            log(f'{name} ep{ep} holdout bacc {evaluate(save=False):.4f}'); model.train()

@torch.inference_mode()
def predict(texts):
    model.eval(); ids = encode(texts); out = np.zeros((len(ids), len(LABELS)), np.float32)
    for b, x, m, _ in batches(ids, None, 32, False):
        out[b] = torch.log_softmax(fwd(x, m), -1).cpu().numpy()
    return out

def evaluate(save=True):
    lp = predict(train.text.iloc[va_idx])
    if save:
        np.save(run / 'holdout_logp.npy', lp); np.save(run / 'holdout_idx.npy', va_idx)
    return balanced_accuracy_score(y[va_idx], lp.argmax(1))

train_stage(pool.text, pool.model.map(L2I).to_numpy(), 1, args.lr1, 'stage1-pool')
base.save_pretrained(run / 'adapter_stage1'); torch.save(head.state_dict(), run / 'adapter_stage1' / 'head.pt')
train_stage(train.text.iloc[tr_idx], y[tr_idx], args.epochs2, args.lr2, 'stage2-train')
base.save_pretrained(run / 'adapter'); torch.save(head.state_dict(), run / 'adapter' / 'head.pt')
json.dump({'labels': LABELS.tolist(), 'keep': cfg.num_hidden_layers, 'layers': LAYERS, 'max_len': args.max_len, 'base': args.model},
          open(run / 'adapter' / 'reader.json', 'w'))
if len(va_idx):
    log(f'FINAL fold {args.fold} holdout bacc {evaluate():.4f}')
log('done')
