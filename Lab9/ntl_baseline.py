"""Name That LLM - CPU baseline (4-view TF-IDF + stylometry, log-linear blend).
Iteration 2: if the public Open Pool dataset (ntl_open_pool.parquet, crtproduction/name-that-llm-open-pool) is found, the
TF-IDF model is trained on train + 231k pool passages (pool weight 0.6) with an SGD logistic model; grouped CV 0.389 -> ~0.51.

Runs unchanged locally and as a Kaggle submission notebook (internet off, CPU only).
  python ntl_baseline.py --cv     # 5-fold StratifiedGroupKFold by prompt_id: balanced accuracy, per-length acc, confusion matrix
  python ntl_baseline.py          # fit on all of train.csv, predict test.csv, write submission.parquet

Ideas borrowed from public work (credited in NOTES.md): grouped CV + char/word TF-IDF baseline (Adisak Sukul, discussion 750390),
shape-text / function-word TF-IDF views (Ahmed Berat Ozer, notebook "Teach the Model, Remove the Prompt").
"""
import argparse, glob, os, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from scipy.special import log_softmax
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

LABELS = np.array(sorted(['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
                          'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
                          'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
                          'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3']))
L2I = {l: i for i, l in enumerate(LABELS)}
STYLO_W = 0.3  # blend weight of the stylometry model vs TF-IDF (log-linear); re-tune with --cv
STYLO_W_POOL = 0.2  # with the pool model (checked on cached grouped-CV OOF: 0.503 -> 0.514)
POOL_W = 0.6  # sample weight of pool rows vs train rows (0.3-1.0 all give ~0.50 OOF)

T0 = time.time()
def log(m): print(f'[{time.time() - T0:6.1f}s] {m}', flush=True)

# ---------------------------------------------------------------- features
FUNCTION_WORDS = set((
    'a about above after again against all am an and any are as at be because been before being below between both but by can '
    'could did do does doing down during each few for from further had has have having he her here hers him his how i if in into '
    'is it its itself just me more most my no nor not now of off on once only or other our out over own same she should so some '
    'such than that the their them then there these they this those through to too under until up very was we were what when '
    'where which while who whom why will with would you your yet even still never always every rather perhaps almost quite simply '
    'truly deeply instead whether though although thus hence therefore however').split())
shape_text = lambda t: re.sub(r"[^\W\d_]+", lambda m: "X" if m[0][0].isupper() else "x", t)  # keeps punctuation/spacing skeleton
function_text = lambda t: " ".join(w.lower() if w.lower() in FUNCTION_WORDS or not w.isalnum() else "C"
                                   for w in re.findall(r"\w+|[^\w\s]", t))  # masks topic words
VIEWS = [  # (text transform, vectorizer opts, weight)
    (None, dict(analyzer='char', ngram_range=(1, 5), max_features=300_000), 1.0),
    (None, dict(analyzer='word', ngram_range=(1, 3), max_features=200_000, token_pattern=r"(?u)\b\w+\b|[^\w\s]"), 0.7),
    (shape_text, dict(analyzer='char', ngram_range=(2, 6), max_features=60_000), 0.5),
    (function_text, dict(analyzer='word', ngram_range=(1, 3), max_features=50_000, token_pattern=r"\S+"), 0.5),
]

PUNCT = {'emdash': '—', 'endash': '–', 'hyphen': '-', 'semi': ';', 'colon': ':', 'comma': ',', 'period': '.',
         'excl': '!', 'quest': '?', 'paren': '(', 'squote': "'", 'dquote': '"', 'curly_s': '’', 'curly_d': '“',
         'ellipsis': '…', 'star': '*', 'hash': '#', 'backtick': '`', 'slash': '/', 'newline': '\n'}

def stylo(t):
    words = re.findall(r"\w+", t)
    sents = [s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s]
    nw, nc = max(len(words), 1), max(len(t), 1)
    sl = np.array([len(re.findall(r"\w+", s)) for s in sents] or [0])
    wl = np.array([len(w) for w in words] or [0])
    f = [np.log1p(len(words)), np.log1p(len(t)), len(sents), sl.mean(), sl.std(), sl.max(), wl.mean(), wl.std(),
         len(set(w.lower() for w in words)) / nw, sum(w[0].isupper() for w in words) / nw,
         sum(c.isdigit() for c in t) / nc, sum(w.lower() in FUNCTION_WORDS for w in words) / nw,
         sum(not c.isascii() for c in t) / nc, t.count('\n\n'), float(t.rstrip()[-1:] in '.!?')]
    f += [t.count(c) / nw * 100 for c in PUNCT.values()]
    return f

def _view(fn, opts, w, fit_texts, te_texts, min_df):
    v = TfidfVectorizer(sublinear_tf=True, lowercase=False, min_df=min_df, dtype=np.float32, **opts)
    f = fn or (lambda t: t)
    return v.fit_transform([f(t) for t in fit_texts]) * w, v.transform([f(t) for t in te_texts]) * w

def fit_predict(tr_texts, y, te_texts, pool=None):
    """Return {component: (n_test, 18) log-probs}. pool: optional DataFrame(text, model) of extra labelled passages."""
    out = {}
    fit_texts = list(tr_texts) + ([] if pool is None else pool.text.tolist())
    mats = Parallel(n_jobs=min(4, os.cpu_count() or 1))(
        delayed(_view)(fn, opts, w, fit_texts, te_texts, 2 if pool is None else 3) for fn, opts, w in VIEWS)
    A, B = hstack([a for a, _ in mats]).tocsr(), hstack([b for _, b in mats]).tocsr()
    if pool is None:
        out['tfidf'] = LogisticRegression(C=30, max_iter=3000).fit(A, y).predict_log_proba(B)
    else:  # 230k rows: SGD logistic instead of lbfgs LR
        yy = np.concatenate([y, pool.model.map(L2I).to_numpy()])
        sw = np.concatenate([np.ones(len(y)), np.full(len(pool), POOL_W)])
        clf = SGDClassifier(loss='log_loss', alpha=2e-6, max_iter=30, tol=1e-4, n_jobs=-1, random_state=0).fit(A, yy, sample_weight=sw)
        out['tfidf'] = log_softmax(clf.decision_function(B), 1)
    S_tr, S_te = np.array([stylo(t) for t in tr_texts]), np.array([stylo(t) for t in te_texts])
    sc = StandardScaler().fit(S_tr)
    out['stylo'] = LogisticRegression(C=0.1, max_iter=5000).fit(sc.transform(S_tr), y).predict_log_proba(sc.transform(S_te))
    return out

def blend(comp, w=STYLO_W):
    return comp['tfidf'] + w * comp['stylo']

def find_pool(d):
    hits = sorted(glob.glob('/kaggle/input/**/ntl_open_pool.parquet', recursive=True)) + [str(d / 'pool' / 'ntl_open_pool.parquet')]
    for h in hits:
        if Path(h).exists():
            p = pd.read_parquet(h, columns=['text', 'model'])
            return p[p.model.isin(LABELS)].assign(text=lambda x: x.text.fillna(''))
    return None

# ---------------------------------------------------------------- data
def find_data():
    hits = sorted(glob.glob('/kaggle/input/**/name-that-llm/train.csv', recursive=True))
    for d in [Path(h).parent for h in hits] + [Path(__file__).resolve().parent / 'data' if '__file__' in globals() else Path('data')]:
        if (d / 'train.csv').exists():
            return d
    sys.exit('train.csv not found: put the competition files in ./data/')

def run_cv(train):
    y = train.model.map(L2I).to_numpy()
    texts = train.text.tolist()
    oof = {k: np.zeros((len(train), len(LABELS))) for k in ('tfidf', 'stylo')}
    for f, (tr, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(train, y, train.prompt_id)):
        comp = fit_predict([texts[i] for i in tr], y[tr], [texts[i] for i in va])
        for k in oof:
            oof[k][va] = comp[k]
        log(f'fold {f}: tfidf {balanced_accuracy_score(y[va], comp["tfidf"].argmax(1)):.4f}  '
            f'blend {balanced_accuracy_score(y[va], blend(comp).argmax(1)):.4f}')
    for k in oof:
        log(f'OOF {k:6s} bacc {balanced_accuracy_score(y, oof[k].argmax(1)):.4f}')
    for w in (0, 0.1, 0.2, 0.3, 0.5, 0.8):  # NB: choosing w on OOF is mildly optimistic
        log(f'OOF blend w={w:.1f} bacc {balanced_accuracy_score(y, blend(oof, w).argmax(1)):.4f}')
    pred = blend(oof).argmax(1)
    words = train.text.str.split().str.len()
    print('\naccuracy by passage length (words):')
    print(pd.Series(pred == y).groupby(pd.cut(words, [0, 60, 120, 180, 1000]), observed=True).agg(['mean', 'size']))
    cm = pd.DataFrame(confusion_matrix(y, pred, labels=range(len(LABELS))), index=LABELS, columns=LABELS)
    cm.to_csv('confusion_matrix.csv')
    recall = pd.Series(np.diag(cm) / cm.sum(1).to_numpy(), index=LABELS).sort_values()
    print('\nper-model recall:\n' + recall.round(3).to_string())
    log('wrote confusion_matrix.csv')

def run_submit(train, d, pool):
    test = pd.read_csv(d / 'test.csv', dtype=str)
    test['text'] = test['text'].fillna('')
    log(f'pool: {"none" if pool is None else len(pool)} rows')
    comp = fit_predict(train.text.tolist(), train.model.map(L2I).to_numpy(), test.text.tolist(), pool)
    sub = pd.DataFrame({'id': test.id.to_numpy(), 'model': LABELS[blend(comp, STYLO_W if pool is None else STYLO_W_POOL).argmax(1)]})
    assert sub.id.is_unique and len(sub) == len(test) and sub.notna().all().all() and sub.model.isin(LABELS).all()
    sub.to_parquet('submission.parquet', index=False)
    log(f'wrote submission.parquet ({len(sub)} rows)')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cv', action='store_true')
    ap.add_argument('--no-pool', action='store_true', help='train on train.csv only (iteration 1)')
    args, _ = ap.parse_known_args()  # tolerate Jupyter's own argv
    d = find_data()
    train = pd.read_csv(d / 'train.csv', dtype=str)
    train['text'] = train['text'].fillna('')
    log(f'data {d}: train {len(train)} rows, {train.prompt_id.nunique()} prompts')
    # --cv stays train-only (pool CV is ~20 min per fold; pool results come from the cached-feature sweep in NOTES.md)
    run_cv(train) if args.cv else run_submit(train, d, None if args.no_pool else find_pool(d))
