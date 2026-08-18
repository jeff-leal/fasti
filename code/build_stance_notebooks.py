# ---------------------------------------------------------------------------
# ONE-TIME SCAFFOLDER — historical record, not part of the replication chain.
#
# This script originally generated code/03E_stance_nli_train.ipynb and
# code/03F_stance_nli_infer.ipynb. Both notebooks have since been executed and
# hand-edited, and they carry stored outputs. RE-RUNNING THIS FILE OVERWRITES
# THEM AND DISCARDS THOSE OUTPUTS.
#
# Its constants are reconciled with the published run (bge-m3-zeroshot-v2.0,
# LR 1.3e-5, NEGATIVES='one', SMOKE=False) so that a regeneration would not
# silently contradict the paper -- but 03E/03F on disk remain authoritative.
#
# Known divergences from the 03F on disk, left deliberately: that notebook derives
# INFER_BATCH from GPU capacity instead of fixing it at 128, and it carries the
# log_run() call that writes the stage's wall time to the timing register.
# ---------------------------------------------------------------------------
import json
from pathlib import Path

CODE = Path(__file__).resolve().parent          # this file lives in code/

def nb_new():
    return []

def md(cells, src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)})

def code(cells, src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": src.strip("\n").splitlines(keepends=True)})

def write(cells, path):
    nb = {"cells": cells,
          "metadata": {"accelerator": "GPU",
                       "colab": {"provenance": [], "gpuType": "A100", "machine_shape": "hm"},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", path, "with", len(cells), "cells")

# Shared preamble (mount + seed + GPU). Differs only by a title comment.
PREAMBLE = r"""
import os, sys, gc, json, random, re
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

try:
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT = Path('/content/drive/MyDrive/Papers/transfer_learning')
except Exception:
    REPO    = Path(os.getenv('TOPIC2IRT_ROOT', r'G:/My Drive/Papers/transfer_learning/topic2irt'))  # local fallback
    PROJECT = REPO.parent
REPO = PROJECT / 'topic2irt'
assert REPO.exists(), f'REPO not found: {REPO}'

RANDOM_STATE = 14605 - 2025 - 4        # = 12576
random.seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)

import torch
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
assert torch.cuda.is_available(), 'Select a GPU runtime (L4/A100 + High-RAM).'
print('torch       :', torch.__version__)
print('device      :', torch.cuda.get_device_name(0))
print('CUDA mem GB :', round(torch.cuda.get_device_properties(0).total_memory/1e9, 1))
print('REPO        :', REPO, '| RANDOM_STATE:', RANDOM_STATE)
"""

# Task / hypothesis definition shared by both notebooks.
TASKS_DEF = r"""
# --- The four NLI tasks (one pooled entailment model) ------------------------
# Each option string is substituted into the task template to form a hypothesis.
# Dict insertion order fixes the argmax label order at inference.
TASKS = {
    'topic': dict(
        col='topic_label',
        template='This text is about {}.',
        options={'Economy': 'the state and the economy',
                 'Society': 'political institutions and society'}),
    'econ': dict(
        col='econ_label',
        template='This text expresses {} on economic issues.',
        options={'State':   'a stance supporting state intervention and redistribution',
                 'Market':  'a stance supporting free markets and growth',
                 'Neutral': 'a neutral stance'}),
    'soc': dict(
        col='soc_label',
        template='This text expresses a {} stance on social issues.',
        options={'Progressive': 'progressive',
                 'Conservative': 'conservative',
                 'Neutral': 'neutral'}),
    'general': dict(   # <-- production target; this wording is used at inference
        col='gen_label',
        template='This text expresses a {} political stance.',
        options={'Left': 'leftist', 'Right': 'rightist', 'Neutral': 'neutral'}),
}

def hyp(task, label):
    t = TASKS[task]
    return t['template'].format(t['options'][label])

LRN_ORDER = ['Left', 'Right', 'Neutral']                 # general-stance order
GEN_HYPS  = [hyp('general', c) for c in LRN_ORDER]        # the 3 production hypotheses
print('Production (general-stance) hypotheses:')
for c, h in zip(LRN_ORDER, GEN_HYPS):
    print(f'  {c:<8} -> "{h}"')
"""

# Hypothesis-scoring helper shared by both notebooks (argmax P(entailment)).
SCORE_FN = r"""
ENT_IDX = model.config.label2id['entailment']

@torch.no_grad()
def score_hypotheses(texts, hyps, batch_size=128, desc='score'):
    '''Return P(entailment) of shape (len(texts), len(hyps)). No threshold.'''
    model.eval()
    texts = list(texts); k = len(hyps)
    probs = np.zeros((len(texts), k), dtype=np.float32)
    for s in tqdm(range(0, len(texts), batch_size), desc=desc):
        b = texts[s:s+batch_size]
        prem = [t for t in b for _ in range(k)]
        hy   = [h for _ in b for h in hyps]
        enc = tokenizer(prem, hy, truncation=True, max_length=MAX_LENGTH,
                        padding=True, return_tensors='pt').to(DEVICE)
        with torch.autocast('cuda', dtype=torch.float16):
            logits = model(**enc).logits
        p = torch.softmax(logits.float(), -1)[:, ENT_IDX].cpu().numpy().reshape(-1, k)
        probs[s:s+len(b)] = p
    return probs

def predict_general(texts, batch_size=128, desc='stance'):
    '''argmax over the 3 general-stance hypotheses -> Left/Right/Neutral + probs.'''
    probs = score_hypotheses(texts, GEN_HYPS, batch_size=batch_size, desc=desc)
    pred = np.array(LRN_ORDER)[probs.argmax(1)]
    return pred, probs
"""

# ============================================================================ #
#  TRAINING NOTEBOOK                                                            #
# ============================================================================ #
c = nb_new()

md(c, r"""# 03E · Train the multilingual NLI classifier (4 tasks, bilingual) — TRAIN ONLY

Fine-tune **`MoritzLaurer/bge-m3-zeroshot-v2.0`** (large, multilingual NLI) with the
**Political DEBATE** recipe into ONE entailment model that covers four tasks:

| Task | Label space | Hypothesis template |
|---|---|---|
| **T1 topic** (binary) | Economy / Society | `This text is about {the state and the economy / political institutions and society}.` |
| **T2 econ stance** (Economy) | State / Market / Neutral | `This text expresses {…} on economic issues.` |
| **T3 soc stance** (Society) | Progressive / Conservative / Neutral | `This text expresses a {…} stance on social issues.` |
| **T4 general stance** (all) | Left / Right / Neutral | `This text expresses a {leftist/rightist/neutral} political stance.` |

**T4 is the production target** (`Left = state+progressive`, `Right = market+conservative`);
its wording is what the **inference notebook (06)** uses.

* **NLI / entailment**, **binary head** `{0:entailment, 1:not_entailment}`
  (neutral+contradiction collapsed; DEBATE §3.2 fn 1), warm-started from XLM-R's
  3-class XNLI head (entailment→0, contradiction→1).
* **Premise = sentence text; speaker-blind** (never party/candidate/country).
* **Bilingual augmentation:** every labelled sentence is used in BOTH its original
  language (`text`) and the English MT (`text_en`, where present) — augmentation +
  cross-lingual alignment. Pair counts (original / translation / both) are reported.
* For each sentence the gold hypothesis = *entail*, the other label hypotheses =
  *not-entail*. **Split by `manifesto_id`** (no leakage) BEFORE building pairs.
* DEBATE hyperparameters; per-epoch eval + load-best + early stopping; `fp16`.
* Reports **macro-F1 per language and per L/R/N** on a held-out test set.

Progress bars on every long step (pair build, training, evaluation).
Granular cells: expensive compute is isolated from plotting/printing.""")

md(c, "## 1 · Preamble — mount, seed, GPU assert\nColab has the needed packages; this notebook does not pip-install.")
code(c, "# === TRAIN notebook ===" + PREAMBLE)

md(c, "## 2 · Config — model, hyperparameters, switches")
code(c, r"""
MODEL_NAME = 'MoritzLaurer/bge-m3-zeroshot-v2.0'   # large multilingual NLI (configurable)
MAX_LENGTH = 192

# DEBATE TrainingArguments (adapted for a LARGE model on L4: eff. batch 16 via accum)
LR            = 1.3e-5
EPOCHS        = 3            # upper bound; early-stopping + load-best decide
TRAIN_BS      = 8
GRAD_ACCUM    = 2           # effective batch = TRAIN_BS * GRAD_ACCUM = 16
EVAL_BS       = 32
WARMUP_RATIO  = 0.06
WEIGHT_DECAY  = 0.01
EARLY_STOP_PATIENCE = 1

# Pipeline switches
OVERSAMPLE   = True          # equalise classes per task (premise level) before pairing
NEGATIVES    = 'one'         # 'one' = single random negative (the published run); 'both' = all other labels
USE_ENGLISH  = True          # add English-MT premises (bilingual augmentation)

# Smoke first (INSTRUCTIONS: see a smoke output before scaling). Flip to False for the
# full ~0.4M-pair run.
SMOKE        = False
SMOKE_SENTS  = 1200          # sentences subsampled (by manifesto) for the smoke pass

FRAME_CSV = REPO / 'data/processed/stance_nli_training_frame.csv'
MODEL_DIR = REPO / 'models/stance_nli_bgem3'
TEST_PRED = REPO / 'data/processed/stance_nli_test_predictions.csv'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
print('model:', MODEL_NAME, '| SMOKE:', SMOKE, '| OVERSAMPLE:', OVERSAMPLE,
      '| NEGATIVES:', NEGATIVES, '| USE_ENGLISH:', USE_ENGLISH)
""")

code(c, TASKS_DEF)

md(c, "## 3 · Load the multi-task frame")
code(c, r"""
frame = pd.read_csv(FRAME_CSV, dtype={'sid': str})
frame['text'] = frame['text'].astype(str)
frame['text_en'] = frame['text_en'].astype('string')
for t in TASKS:
    col = TASKS[t]['col']
    n = frame[col].notna().sum()
    vc = frame[col].value_counts().to_dict()
    print(f'{t:<8} ({col:<11}) labelled={n:>6,}  {vc}')
print('\\nrows:', f'{len(frame):,}', '| manifestos:', frame['manifesto_id'].nunique(),
      '| languages:', frame['language'].nunique(),
      '| with text_en:', int(frame['text_en'].str.strip().fillna('').ne('').sum()))
""")

md(c, "## 4 · Leakage-safe split by `manifesto_id` (70/15/15)\nSplit sentences BEFORE pairing so a manifesto's original+English+all-task pairs stay in one split.")
code(c, r"""
from sklearn.model_selection import GroupShuffleSplit

work = frame
if SMOKE:
    # keep whole manifestos until we reach ~SMOKE_SENTS sentences
    order = work['manifesto_id'].drop_duplicates().sample(frac=1.0, random_state=RANDOM_STATE)
    keep, tot = [], 0
    sizes = work.groupby('manifesto_id').size()
    for mid in order:
        keep.append(mid); tot += int(sizes[mid])
        if tot >= SMOKE_SENTS:
            break
    work = work[work['manifesto_id'].isin(keep)].reset_index(drop=True)
    print(f'SMOKE: {len(work):,} sentences from {len(keep)} manifestos')

groups = work['manifesto_id'].values
idx = np.arange(len(work))
tr, tmp = next(GroupShuffleSplit(1, test_size=0.30, random_state=RANDOM_STATE).split(idx, groups=groups))
va_rel, te_rel = next(GroupShuffleSplit(1, test_size=0.50, random_state=RANDOM_STATE)
                      .split(tmp, groups=groups[tmp]))
va, te = tmp[va_rel], tmp[te_rel]
split_train = work.iloc[tr].reset_index(drop=True)
split_val   = work.iloc[va].reset_index(drop=True)
split_test  = work.iloc[te].reset_index(drop=True)
for a, b in [('train','val'),('train','test'),('val','test')]:
    assert not (set(eval('split_'+a)['manifesto_id']) & set(eval('split_'+b)['manifesto_id'])), f'leak {a}/{b}'
for nm, d in [('train',split_train),('val',split_val),('test',split_test)]:
    print(f'{nm:5s}: {len(d):6,} sentences | {d["manifesto_id"].nunique():3d} manifestos')
""")

md(c, "## 5 · Build NLI premise–hypothesis pairs (bilingual) + report pair counts\nEach sentence → 1 *entail* (gold) + *not-entail* (other labels), in original AND English.")
code(c, r"""
from sklearn.utils import resample

def build_pairs(df, oversample, negatives, use_english, seed, tag=''):
    rng = np.random.default_rng(seed)
    rows = []
    for task, cfg in TASKS.items():
        col, opts = cfg['col'], list(cfg['options'])
        sub = df.loc[df[col].notna(), ['text', 'text_en', col]].rename(columns={col: 'lab'})
        if len(sub) == 0:
            continue
        if oversample:
            mx = int(sub['lab'].value_counts().max())
            sub = pd.concat([resample(g, replace=True, n_samples=mx, random_state=seed)
                             for _, g in sub.groupby('lab')], ignore_index=True)
        for r in tqdm(sub.itertuples(index=False), total=len(sub), desc=f'pairs:{task}{tag}'):
            premises = [(r.text, 'orig')]
            if use_english and isinstance(r.text_en, str) and r.text_en.strip():
                premises.append((r.text_en, 'en'))
            others = [o for o in opts if o != r.lab]
            if negatives == 'one' and len(others) > 1:
                others = [others[int(rng.integers(len(others)))]]
            for prem, lang in premises:
                rows.append((prem, hyp(task, r.lab), 0, task, lang))      # entail
                for o in others:
                    rows.append((prem, hyp(task, o), 1, task, lang))      # not-entail
    out = pd.DataFrame(rows, columns=['premise', 'hypothesis', 'label', 'task', 'premise_lang'])
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)

def report_pairs(pairs, name):
    by = pairs.groupby(['task', 'premise_lang']).size().unstack(fill_value=0)
    o = int((pairs.premise_lang == 'orig').sum()); e = int((pairs.premise_lang == 'en').sum())
    print(f'\\n=== {name}: {len(pairs):,} pairs (entail {100*(pairs.label==0).mean():.1f}%) ===')
    print(by)
    print(f'  original-language pairs : {o:,}')
    print(f'  English-translation pairs: {e:,}')
    print(f'  BOTH (total)            : {o+e:,}')

pairs_train = build_pairs(split_train, OVERSAMPLE, NEGATIVES, USE_ENGLISH, RANDOM_STATE, ' tr')
pairs_val   = build_pairs(split_val,   False, 'both', USE_ENGLISH, RANDOM_STATE, ' va')
report_pairs(pairs_train, 'TRAIN')
report_pairs(pairs_val, 'VAL')
""")

md(c, "## 6 · Tokenize → HF datasets\n`tokenizer(premise, hypothesis)` — premise first (NLI order).")
code(c, r"""
from transformers import AutoTokenizer
from datasets import Dataset

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def to_ds(pairs):
    ds = Dataset.from_pandas(pairs[['premise', 'hypothesis', 'label']], preserve_index=False)
    ds = ds.map(lambda b: tokenizer(b['premise'], b['hypothesis'], truncation=True, max_length=MAX_LENGTH),
                batched=True, remove_columns=['premise', 'hypothesis'])
    return ds

ds_train, ds_val = to_ds(pairs_train), to_ds(pairs_val)
print(ds_train)
""")

md(c, "## 7 · Load model with a binary entail/not-entail head  *(expensive — own cell)*\nWarm-start the 2-class head from XLM-R's 3-class XNLI head (entailment→0, contradiction→1).")
code(c, r"""
from transformers import AutoModelForSequenceClassification

ID2LABEL = {0: 'entailment', 1: 'not_entailment'}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

_base = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
_l2i = {k.lower(): v for k, v in _base.config.label2id.items()}
ent_i, con_i = _l2i.get('entailment', 2), _l2i.get('contradiction', 0)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID, ignore_mismatched_sizes=True)

def _last_linear(m, out_features):
    last = None
    for mod in m.modules():
        if isinstance(mod, torch.nn.Linear) and mod.out_features == out_features:
            last = mod
    return last
try:
    src, dst = _last_linear(_base, 3), _last_linear(model, 2)
    if src is not None and dst is not None and src.in_features == dst.in_features:
        with torch.no_grad():
            dst.weight[0].copy_(src.weight[ent_i]); dst.weight[1].copy_(src.weight[con_i])
            if src.bias is not None and dst.bias is not None:
                dst.bias[0].copy_(src.bias[ent_i]); dst.bias[1].copy_(src.bias[con_i])
        print('Head warm-started from 3-class XNLI head (entail->0, contra->1).')
    else:
        print('Head warm-start skipped (shape mismatch); fresh 2-class head.')
except Exception as e:
    print('Head warm-start failed (', e, '); fresh 2-class head.')

del _base; gc.collect(); torch.cuda.empty_cache()
model.to(DEVICE)
print('params:', f'{sum(p.numel() for p in model.parameters()):,}')
""")

md(c, "## 8 · Trainer (DEBATE hyperparameters) + binary metrics")
code(c, r"""
from transformers import (TrainingArguments, Trainer, DataCollatorWithPadding,
                          EarlyStoppingCallback)
from sklearn.metrics import f1_score, matthews_corrcoef

collator = DataCollatorWithPadding(tokenizer)
def compute_metrics(ep):
    logits, labels = ep
    pred = np.asarray(logits).argmax(-1)
    return {'macro_f1': f1_score(labels, pred, average='macro'),
            'mcc': matthews_corrcoef(labels, pred),
            'accuracy': float((pred == labels).mean())}

args = TrainingArguments(
    output_dir=str(MODEL_DIR / 'checkpoints'),
    learning_rate=LR, lr_scheduler_type='linear', warmup_ratio=WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,
    per_device_train_batch_size=TRAIN_BS, gradient_accumulation_steps=GRAD_ACCUM,
    per_device_eval_batch_size=EVAL_BS, num_train_epochs=EPOCHS, fp16=True,
    eval_strategy='epoch', save_strategy='epoch', save_total_limit=2,
    load_best_model_at_end=True, metric_for_best_model='macro_f1', greater_is_better=True,
    logging_steps=50, report_to='none', seed=RANDOM_STATE)

trainer = Trainer(model=model, args=args, train_dataset=ds_train, eval_dataset=ds_val,
                  tokenizer=tokenizer, data_collator=collator, compute_metrics=compute_metrics,
                  callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)])
print('train pairs:', len(ds_train), '| val pairs:', len(ds_val))
""")

md(c, "## 9 · Train  *(expensive — own cell; Trainer shows a progress bar)*")
code(c, r"""
res = trainer.train()
print(res.metrics)
print('best val macro_f1:', trainer.state.best_metric)
""")

md(c, "## 10 · Inference helper + held-out TEST evaluation (T4 production target)")
code(c, SCORE_FN)
code(c, r"""
test_gen = split_test[split_test['gen_label'].notna()].copy()
pred, probs = predict_general(test_gen['text'].tolist(), batch_size=EVAL_BS, desc='test T4')
test_gen = test_gen.assign(pred=pred, p_left=probs[:,0], p_right=probs[:,1], p_neutral=probs[:,2])
test_gen.to_csv(TEST_PRED, index=False)
print('saved test predictions ->', TEST_PRED, '| n =', len(test_gen))
""")

md(c, "### 10b · T4 metrics — macro-F1 overall, per language, per L/R/N  *(cheap)*")
code(c, r"""
from sklearn.metrics import classification_report, confusion_matrix, f1_score
y, yh = test_gen['gen_label'].values, test_gen['pred'].values
macro = f1_score(y, yh, labels=LRN_ORDER, average='macro')
print('T4 TEST macro-F1 (overall): %.3f   n=%d\\n' % (macro, len(y)))
print('per-class F1:')
for cl, f in zip(LRN_ORDER, f1_score(y, yh, labels=LRN_ORDER, average=None)):
    print(f'  {cl:<8} F1={f:.3f}  (support={int((y==cl).sum())})')
print('\\nper-language macro-F1:')
lang_rows = []
for lg, d in test_gen.groupby('language'):
    if len(d) < 10:
        print(f'  {lg:<12} n={len(d)} (skip)'); continue
    m = f1_score(d['gen_label'], d['pred'], labels=LRN_ORDER, average='macro')
    lang_rows.append((lg, len(d), m)); print(f'  {lg:<12} n={len(d):<5} macro-F1={m:.3f}')
print('\\n', classification_report(y, yh, labels=LRN_ORDER, digits=3))
print('confusion (rows=gold, cols=pred)', LRN_ORDER); print(confusion_matrix(y, yh, labels=LRN_ORDER))
""")

md(c, "### 10c · Auxiliary tasks on test (T1 topic / T2 econ / T3 soc)  *(cheap, sanity)*")
code(c, r"""
from sklearn.metrics import f1_score
for task in ['topic', 'econ', 'soc']:
    col, opts = TASKS[task]['col'], list(TASKS[task]['options'])
    d = split_test[split_test[col].notna()]
    if len(d) < 10:
        print(f'{task}: too few test rows'); continue
    hyps = [hyp(task, o) for o in opts]
    pr = np.array(opts)[score_hypotheses(d['text'].tolist(), hyps, batch_size=EVAL_BS, desc=task).argmax(1)]
    acc = float((pr == d[col].values).mean())
    mf1 = f1_score(d[col].values, pr, labels=opts, average='macro')
    print(f'{task:<6} n={len(d):<5} acc={acc:.3f}  macro-F1={mf1:.3f}')
""")

md(c, "### 10d · Plots  *(cheap, separate cell)*")
code(c, r"""
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
cm = confusion_matrix(y, yh, labels=LRN_ORDER).astype(float)
cmn = cm / cm.sum(1, keepdims=True)
ax[0].imshow(cmn, cmap='Blues', vmin=0, vmax=1)
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(LRN_ORDER); ax[0].set_yticks(range(3)); ax[0].set_yticklabels(LRN_ORDER)
ax[0].set_xlabel('pred'); ax[0].set_ylabel('gold'); ax[0].set_title('T4 confusion (row-norm)')
for i in range(3):
    for j in range(3):
        ax[0].text(j, i, f'{cmn[i,j]:.2f}', ha='center', va='center', color='white' if cmn[i,j]>.5 else 'black')
if lang_rows:
    lg = [r[0] for r in lang_rows]; vv = [r[2] for r in lang_rows]; o = np.argsort(vv)
    ax[1].barh([lg[i] for i in o], [vv[i] for i in o], color='#4C72B0')
    ax[1].axvline(macro, color='k', ls='--', lw=1, label=f'overall {macro:.2f}')
    ax[1].set_xlim(0,1); ax[1].set_xlabel('macro-F1'); ax[1].set_title('Per-language macro-F1 (T4)'); ax[1].legend()
plt.tight_layout(); plt.show()
""")

md(c, "## 11 · Save model + tokenizer + metadata")
code(c, r"""
model.save_pretrained(MODEL_DIR); tokenizer.save_pretrained(MODEL_DIR)
meta = dict(model_name=MODEL_NAME, max_length=MAX_LENGTH, id2label=ID2LABEL,
            lrn_order=LRN_ORDER, general_hypotheses=GEN_HYPS,
            tasks={k: {'template': v['template'], 'options': v['options']} for k, v in TASKS.items()},
            oversample=OVERSAMPLE, negatives=NEGATIVES, use_english=USE_ENGLISH,
            smoke=SMOKE, random_state=RANDOM_STATE, test_macro_f1=float(macro))
(MODEL_DIR / 'stance_nli_meta.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False))
print('saved ->', MODEL_DIR)
print('NOTE: SMOKE=%s. Re-run with SMOKE=False for the full model before inference.' % SMOKE)
""")

write(c, CODE / '03E_stance_nli_train.ipynb')

# ============================================================================ #
#  INFERENCE NOTEBOOK                                                           #
# ============================================================================ #
c = nb_new()

md(c, r"""# 03F · Inference — per-sentence Left/Right/Neutral for US + BR — INFER ONLY

Loads the fine-tuned model from notebook 05 and labels the two target corpora
with the **general political stance (T4)**: `argmax P(entailment)` over the three
production hypotheses — **no threshold, no calibration**. Neutral is **retained**.

Outputs one row per sentence: `doc_id, chunk_id, stance, p_left, p_right, p_neutral`.

**This stage does inference and nothing else.** It reads the sentence-chunk corpora
written by 00 and never touches the topic model, so it can run in parallel with the
topic stage. The topic is attached later, in `04_irt_data.py`, which joins stance to
the labelled chunk table on `chunk_id`.

Separate from training so you can run inference on more capable hardware.
Resumable chunked writes + progress bars on every long step.""")

md(c, "## 1 · Preamble — mount, seed, GPU assert")
code(c, "# === INFER notebook ===" + PREAMBLE)

md(c, "## 2 · Config")
code(c, r"""
MODEL_DIR   = REPO / 'models/stance_nli_bgem3'
MAX_LENGTH  = 192
INFER_BATCH = 128

# Smoke first: label a small random sample (writes *_SMOKE.csv). Flip to False for full.
SMOKE   = True
SMOKE_N = 2000

OUT_US = REPO / 'data/stance/stance_pred_us.csv'
OUT_BR = REPO / 'data/stance/stance_pred_br.csv'
print('model dir:', MODEL_DIR, '| SMOKE:', SMOKE)
""")

md(c, "## 3 · Load fine-tuned model + tokenizer + hypotheses")
code(c, r"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
meta = json.loads((MODEL_DIR / 'stance_nli_meta.json').read_text())
LRN_ORDER = meta['lrn_order']
GEN_HYPS  = meta['general_hypotheses']
MAX_LENGTH = meta.get('max_length', MAX_LENGTH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE).eval()
print('loaded:', meta['model_name'], '| test macro-F1 (from training):', meta.get('test_macro_f1'))
for cl, h in zip(LRN_ORDER, GEN_HYPS):
    print(f'  {cl:<8} -> "{h}"')
""")

md(c, "## 4 · Stance scoring helper (argmax P(entailment))")
code(c, SCORE_FN)

md(c, "## 5 · Resumable corpus-inference helper + mojibake repair")
code(c, r"""
def run_corpus_inference(df, out_csv, name, chunk_rows=200_000, resume=True):
    out_csv = Path(out_csv)
    cols = ['doc_id','chunk_id','stance','p_left','p_right','p_neutral']
    start = 0
    if resume and out_csv.exists():
        start = max(0, sum(1 for _ in open(out_csv, encoding='utf-8')) - 1)
        print(f'[{name}] resuming after {start:,} rows')
    n = len(df)
    if start >= n:
        print(f'[{name}] already complete ({n:,} rows)'); return
    for c0 in range(start, n, chunk_rows):
        c1 = min(c0 + chunk_rows, n)
        sub = df.iloc[c0:c1]
        pred, probs = predict_general(sub['chunk_text'].tolist(), batch_size=INFER_BATCH,
                                      desc=f'{name} {c0:,}-{c1:,}')
        pd.DataFrame({'doc_id': sub['doc_id'].values, 'chunk_id': sub['chunk_id'].values,
                      'stance': pred,
                      'p_left': probs[:,0], 'p_right': probs[:,1], 'p_neutral': probs[:,2]})[cols] \
            .to_csv(out_csv, mode=('w' if c0 == 0 else 'a'), header=(c0 == 0), index=False, encoding='utf-8')
    print(f'[{name}] done -> {out_csv} ({n:,} rows)')

# BR feather is already clean UTF-8; repair only fires on unambiguous double-encoding.
_MOJI = re.compile(r'Ã[©³§£ªº¡­¢¤¨«¬]|Â[ ©°»®]|â€')
def repair_mojibake(series):
    def fix(s):
        if not isinstance(s, str) or not _MOJI.search(s):
            return s
        try:
            return s.encode('latin-1').decode('utf-8')
        except Exception:
            return s
    return series.map(fix)
""")

md(c, "## 6 · US corpus (CampaignView House platforms, English)\nText from `campaignview_chunks_sent.feather`. Inference only — no topic model is read.")
code(c, r"""
us = pd.read_feather(REPO / 'data/us/campaignview_chunks_sent.feather')[['chunk_id','doc_id','chunk_text']].copy()
us['chunk_text'] = us['chunk_text'].astype(str)
print('US sentences:', f'{len(us):,}', 'over', f"{us['doc_id'].nunique():,} platforms")
us_run = us.sample(SMOKE_N, random_state=RANDOM_STATE).reset_index(drop=True) if SMOKE else us
run_corpus_inference(us_run, OUT_US.with_name('stance_pred_us_SMOKE.csv') if SMOKE else OUT_US, name='US')
""")

md(c, "## 7 · BR corpus (Brazilian mayoral platforms, Portuguese)\nText from `br_manifestos_chunks_sent.feather`, restricted to `valid_mayor_platform` so it covers exactly the chunks the topic stage keeps. Inference only — no topic model is read.")
code(c, r"""
br = pd.read_feather(REPO / 'data/br/br_manifestos_chunks_sent.feather')[['chunk_id','doc_id','chunk_text']].copy()
# Same mayoral restriction the topic stage applies, so both cover the same chunks.
_pm = pd.read_feather(REPO / 'data/br/platform_party_map.feather',
                      columns=['platform_id', 'valid_mayor_platform'])
_mayor = set(_pm.loc[_pm['valid_mayor_platform'] == True, 'platform_id'].astype(str))
br = br[br['doc_id'].astype(str).isin(_mayor)].reset_index(drop=True)
br['chunk_text'] = repair_mojibake(br['chunk_text'].astype(str))
print('BR sentences:', f'{len(br):,}', 'over', f"{br['doc_id'].nunique():,} mayoral platforms")
br_run = br.sample(SMOKE_N, random_state=RANDOM_STATE).reset_index(drop=True) if SMOKE else br
run_corpus_inference(br_run, OUT_BR.with_name('stance_pred_br_SMOKE.csv') if SMOKE else OUT_BR, name='BR')
""")

md(c, "## 8 · Output sanity check")
code(c, r"""
for nm, full in [('US', OUT_US), ('BR', OUT_BR)]:
    p = full.with_name(full.stem + '_SMOKE.csv') if SMOKE else full
    if not Path(p).exists():
        print(f'{nm}: {p.name} not written'); continue
    d = pd.read_csv(p)
    print(f'\\n== {nm}: {p.name} ({len(d):,} rows) ==')
    print('stance dist:', d['stance'].value_counts(normalize=True).round(3).to_dict())
    print(d.head(4).to_string())
""")

write(c, CODE / '03F_stance_nli_infer.ipynb')
