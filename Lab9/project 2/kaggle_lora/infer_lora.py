"""Name That LLM - Kaggle T4 inference for our l24 LoRA readers (trained by hpc/train_lora.py).

Each reader dir holds adapter_model.safetensors (LoRA), adapter_config.json, head.pt and reader.json.
The LoRA is merged into a truncated fp16 base by hand (no peft needed). A batch whose hidden states overflow
fp16 is recomputed with the model in bf16 (emulated on T4, slower but finite).
Validation: for readers trained with a held-out fold, re-score that fold and compare with the H100 number.
"""
import glob, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from safetensors import safe_open
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score
from transformers import AutoConfig, AutoModel, AutoTokenizer

T0 = time.time()
def log(m): print(f'[{time.time() - T0:7.1f}s] {m}', flush=True)

def find_base(tag='8b-base'):
    hits = [Path(p).parent for p in glob.glob('/kaggle/input/**/config.json', recursive=True)
            if f'/{tag}/' in p.lower() and '/qwen-3/' in p.lower()  # whole segments: not llama's 8b-base, not 4b-base in 14b-base
            and (Path(p).parent / 'tokenizer.json').exists()]
    assert hits, f'base model {tag} not attached'
    return str(sorted(hits)[0])

def find_readers():
    return {Path(p).parent.name: str(Path(p).parent) for p in glob.glob('/kaggle/input/**/reader.json', recursive=True)}

def load_reader(base_dir, reader_dir):
    meta = json.load(open(f'{reader_dir}/reader.json'))
    acfg = json.load(open(f'{reader_dir}/adapter_config.json'))
    cfg = AutoConfig.from_pretrained(base_dir, num_hidden_layers=meta['keep'], use_cache=False)
    model = AutoModel.from_pretrained(base_dir, config=cfg, dtype=torch.float16, device_map='auto').eval()
    scale = acfg['lora_alpha'] / acfg['r']
    mods = dict(model.named_modules())
    with safe_open(f'{reader_dir}/adapter_model.safetensors', 'pt') as f:
        keys = [k for k in f.keys() if k.endswith('lora_A.weight')]
        for k in keys:
            name = k[len('base_model.model.'):-len('.lora_A.weight')]
            lin = mods[name]
            A, B = f.get_tensor(k).float(), f.get_tensor(k.replace('lora_A', 'lora_B')).float()
            W = lin.weight
            W.data = (W.data.float() + scale * (B @ A).to(W.device)).to(W.dtype)
    log(f'merged {len(keys)} LoRA pairs into {meta["keep"]} layers (scale {scale})')
    h = torch.load(f'{reader_dir}/head.pt', map_location='cpu')
    L = meta['layers']
    norms = [(h[f'norms.{i}.weight'].float().cuda(0), h[f'norms.{i}.bias'].float().cuda(0)) for i in range(len(L))]
    W_out, b_out = h['out.weight'].float().cuda(0), h['out.bias'].float().cuda(0)
    return model, meta, norms, (W_out, b_out)

@torch.no_grad()
def predict(texts, base_dir, reader_dir, bs=16):
    model, meta, norms, (W_out, b_out) = load_reader(base_dir, reader_dir)
    tok = AutoTokenizer.from_pretrained(base_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ids = tok(list(texts), add_special_tokens=False, truncation=True, max_length=meta['max_len'])['input_ids']  # same as training
    order = np.argsort([len(x) for x in ids]); out = np.zeros((len(ids), len(meta['labels'])), np.float32); n_bf16 = 0
    dev = next(model.parameters()).device
    for s in range(0, len(order), bs):
        b = order[s:s + bs]; Lmax = max(len(ids[i]) for i in b)
        x = torch.full((len(b), Lmax), tok.pad_token_id); m = torch.zeros((len(b), Lmax), dtype=torch.long)
        for j, i in enumerate(b):
            x[j, :len(ids[i])] = torch.tensor(ids[i]); m[j, :len(ids[i])] = 1
        x, m = x.to(dev), m.to(dev)
        hs = model(input_ids=x, attention_mask=m, output_hidden_states=True).hidden_states
        if not all(torch.isfinite(hs[l]).all() for l in meta['layers']):
            model.to(torch.bfloat16); hs = model(input_ids=x, attention_mask=m, output_hidden_states=True).hidden_states
            model.to(torch.float16); n_bf16 += 1
        mk = m.unsqueeze(-1).float().cuda(0); mk[:, 0] = 0  # skip first token, as in training
        z = []
        for l, (g, be) in zip(meta['layers'], norms):
            v = (hs[l].float().cuda(0) * mk).sum(1) / mk.sum(1).clamp(min=1)
            z.append(torch.nn.functional.layer_norm(v, (v.shape[-1],), g, be))
        out[b] = torch.log_softmax(torch.cat(z, -1) @ W_out.T + b_out, -1).cpu().numpy()
    log(f'{Path(reader_dir).name}: {len(texts)} texts, {n_bf16} batches recomputed in bf16')
    del model; torch.cuda.empty_cache()
    return out, meta['labels']

if __name__ == '__main__':
    data = Path(sorted(glob.glob('/kaggle/input/**/name-that-llm/train.csv', recursive=True))[0]).parent
    train = pd.read_csv(data / 'train.csv'); train['text'] = train.text.fillna('')
    test = pd.read_csv(data / 'test.csv', dtype=str); test['text'] = test.text.fillna('')
    readers = find_readers(); log(f'readers: {sorted(readers)}')
    base = find_base(); log(f'base {base}; GPUs {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}')
    for name, rdir in sorted(readers.items()):
        if 'fold0' in name:  # parity check against the H100 holdout score
            labels = json.load(open(f'{rdir}/reader.json'))['labels']
            y = train.model.map({l: i for i, l in enumerate(labels)}).to_numpy()
            # use the exact holdout rows from training: the sklearn version on Kaggle splits folds differently
            va = np.load(f'{rdir}/holdout_idx.npy')
            lp, _ = predict(train.text.iloc[va], base, rdir)
            ref = np.load(f'{rdir}/holdout_logp.npy')  # H100 bf16 predictions for the same rows
            log(f'VALIDATE {name}: fold-0 holdout bacc T4 {balanced_accuracy_score(y[va], lp.argmax(1)):.4f} | '
                f'H100 {balanced_accuracy_score(y[va], ref.argmax(1)):.4f} | argmax agreement {(lp.argmax(1) == ref.argmax(1)).mean():.3f} '
                f'| max |dlogp| {np.abs(lp - ref).max():.3f}')
        else:
            lp, labels = predict(test.text, base, rdir)
            np.save(f'test_logp_{name}.npy', lp)
            pd.DataFrame({'id': test.id, 'model': np.array(labels)[lp.argmax(1)]}).to_parquet('submission.parquet', index=False)
            log(f'{name}: wrote test_logp_{name}.npy and submission.parquet')
