"""Insert our LoRA readers into the forked 0.704 stack notebook (after the ensemble cell, before GLM)."""
import json
nb = json.load(open('stack.ipynb', encoding='utf-8'))
src = lambda c: ''.join(c['source'])
k = next(i for i, c in enumerate(nb['cells']) if "'ensemble worker failed'" in src(c))
infer = open('../kaggle_lora/infer_lora.py', encoding='utf-8').read()
runner = r'''# ---- Our HPC-trained l24 LoRA readers (private dataset somanshgoel/ntl-lora-readers) -----------------------
# Merged into the stack's final logits before the GLM stage: z <- log_softmax(z_stack) + OURS_W * mean(our log-probs).
OURS = {'q8_full_fullpool': '8b-base'}   # reader dir -> tag of the attached Qwen3 base model
OURS_W = 0.35                            # blind: about one stack member's share; tune on the leaderboard
import glob, numpy as np, subprocess, sys
from pathlib import Path
from scipy.special import log_softmax
_rd = {Path(p).parent.name: str(Path(p).parent) for p in glob.glob('/kaggle/input/**/reader.json', recursive=True)}
assert set(OURS) <= set(_rd), f'missing readers: {set(OURS) - set(_rd)}'
_test_csv = os.path.join(DATA, 'test.csv')
for _n, _tag in OURS.items():  # one subprocess per reader so GPU memory is freed between them
    _code = ("import sys, numpy as np, pandas as pd; sys.path.insert(0, '/kaggle/working'); from infer_lora import predict, find_base; "
             f"t = pd.read_csv({_test_csv!r}, dtype=str); t['text'] = t.text.fillna(''); "
             f"lp, lab = predict(t.text, find_base({_tag!r}), {_rd[_n]!r}); "
             f"np.save('/kaggle/temp/ours_{_n}.npy', lp); open('/kaggle/temp/ours_{_n}.labels', 'w').write(','.join(lab))")
    assert run([sys.executable, '-c', _code]) == 0, f'our reader {_n} failed'
_ids = pd.read_csv(_test_csv, dtype={'id': str}).id.astype(str).tolist()
_lps = []
for _n in OURS:
    _lab = open(f'/kaggle/temp/ours_{_n}.labels').read().split(',')
    _lps.append(np.load(f'/kaggle/temp/ours_{_n}.npy'))
_ours = np.mean(_lps, 0)
assert _ours.shape == (len(_ids), 18) and np.isfinite(_ours).all()

def merge_ours(path):
    with np.load(path, allow_pickle=False) as f:
        ids, labels, z = f['ids'].astype(str), f['labels'].astype(str), f['logits'].astype(np.float64)
    assert ids.tolist() == _ids, 'final-logits row order differs from test.csv'
    z2 = log_softmax(z, 1) + OURS_W * _ours[:, [_lab.index(l) for l in labels]]
    np.savez(path, ids=ids, labels=labels, logits=z2)
    pd.DataFrame({'id': ids, 'model': labels[z2.argmax(1)]}).to_parquet(OUT, index=False)
    print(f'merged our readers (w={OURS_W}): changed {(z.argmax(1) != z2.argmax(1)).mean():.3%} of argmax')

if os.path.exists(FINAL_LOGITS) and not FAST_COMMIT:
    merge_ours(FINAL_LOGITS)
else:  # fast path: the ensemble did not run; exercise the merge on a dummy logits file
    _lab0 = sorted(_lab)
    np.savez('/kaggle/temp/dummy-logits.npz', ids=np.array(_ids), labels=np.array(_lab0), logits=np.zeros((len(_ids), 18)))
    _keep_out = OUT; OUT = '/kaggle/temp/dummy-sub.parquet'; merge_ours('/kaggle/temp/dummy-logits.npz'); OUT = _keep_out
    print('our readers smoke ok', {k: v.shape for k, v in zip(OURS, _lps)})
'''
cell = lambda s: {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': s.splitlines(True)}
nb['cells'][k + 1:k + 1] = [cell('%%writefile /kaggle/working/infer_lora.py\n' + infer), cell(runner)]
json.dump(nb, open('stack.ipynb', 'w', encoding='utf-8', newline='\n'), indent=1)
m = json.load(open('kernel-metadata.json'))
m.pop('id_no', None)
m.update(id='somanshgoel/ntl-stack-base-0704', title='NTL Stack Base 0704', code_file='stack.ipynb', is_private=True)
m['dataset_sources'] = m['dataset_sources'] + ['somanshgoel/ntl-lora-readers']
json.dump(m, open('kernel-metadata.json', 'w', newline='\n'), indent=1)
print('inserted after cell', k, '| datasets', len(m['dataset_sources']))
