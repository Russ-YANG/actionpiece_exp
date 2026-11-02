"""Build the final comparison table plus the diagnostic tables.

Called by mm_finisher.sh once all three arms have finished.  Everything is read
from TensorBoard scalars and the saved tokenizer artefacts, so it also works
mid-run.

Usage:  make_final_table.py [--md out.md]
"""
import argparse
import json
import os
from collections import Counter

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

AP = '/home/lzr/science/actionpiece_exp'
TB = f'{AP}/tensorboard/AmazonReviews2014/ActionPiece'
B = f'{AP}/cache/AmazonReviews2014/Beauty/processed'
ID2ITEM = json.load(open(f'{B}/id_mapping.json'))['id2item']
N_ITEMS = len(ID2ITEM) - 1

TEXT_STEM = ('qwen_local.Qwen3-VL-Embedding-8B.d768.i711daf44f948.'
             'r2c4565515e0f.c393e2978d278.m2048.tbfloat16.asdpa')

# label, run-dir substring, text codes, text codebook, image codes,
# image codebook, text dim, image dim, text sid file, image sid file
ARMS = [
    ('mm 2+2 (new)', 'beauty_mm_sep22_train', 2, 256, 2, 256, 768, 768,
     f'{TEXT_STEM}.opq2x256.seed2024.sem_ids',
     'Qwen3-VL-Embedding-8B.d768.opq2x256.seed2024.image_sem_ids'),
    ('mm 2+2 seed42 (replica)', 'beauty_mm_sep22_seed42_train', 2, 256, 2, 256,
     768, 768,
     f'{TEXT_STEM}.opq2x256.seed42.sem_ids',
     'Qwen3-VL-Embedding-8B.d768.opq2x256.seed42.image_sem_ids'),
    ('mm 3+3 (new)', 'beauty_mm_sep33_train', 3, 256, 3, 256, 768, 768,
     f'{TEXT_STEM}.opq3x256.seed2024.sem_ids',
     'Qwen3-VL-Embedding-8B.d768.opq3x256.seed2024.image_sem_ids'),
    ('mm 4+4 (new)', 'beauty_mm_sep44_train', 4, 256, 4, 256, 768, 768,
     f'{TEXT_STEM}.opq4x256.seed2024.sem_ids',
     'Qwen3-VL-Embedding-8B.d768.opq4x256.seed2024.image_sem_ids'),
    ('text only 4x256', 'beauty_e1_qwen3_vl_8b_train', 4, 256, 0, 0, 768, None,
     f'{TEXT_STEM}.opq4x256.seed2024.sem_ids', None),
    ('image only 4x256', 'beauty_e2_qwen3_vl_8b_image_train', 0, 0, 4, 256,
     None, 512, None, None),
    ('joint 4096 4x256', 'beauty_e3_qwen3_vl_8b_joint_d4096_train', 4, 256,
     0, 0, 4096, None,
     'qwen_local.Qwen3-VL-Embedding-8B.d4096.i711daf44f948.r2c4565515e0f.'
     'c393e2978d278.m2048.tbfloat16.asdpa.joint.im2f0852d8f8d9.'
     'opq4x256.seed2024.sem_ids', None),
    ('separate 2x65536 (old)', 'beauty_e4_freqmax_pack2', 2, 65536, 2, 65536,
     768, 768,
     f'{TEXT_STEM}.opq2x65536.seed2024.freqmax_pack2.sem_ids',
     'qwen_local.Qwen3-VL-Embedding-8B.d768.ib6b0d724b414.'
     'r2c4565515e0f.c393e2978d278.m2048.tbfloat16.asdpa.image.'
     'im2f0852d8f8d9.opq2x65536.seed2024.freqmax_pack2.sem_ids'),
]

# item-feature files, used by the diagnostics
FEAT_FILES = [
    ('mm 2+2 (new)', 2,
     'item.qwen_separate.Qwen3-VL-Embedding-8B.t2x256.i2x256.seed2024.h128.feat',
     'actionpiece.qwen_separate.Qwen3-VL-Embedding-8B.t2x256.i2x256.'
     'seed2024.h128.v40000.json'),
    ('mm 2+2 seed42 (replica)', 2,
     'item.qwen_separate.Qwen3-VL-Embedding-8B.t2x256.i2x256.seed42.h128.feat',
     'actionpiece.qwen_separate.Qwen3-VL-Embedding-8B.t2x256.i2x256.'
     'seed42.h128.v40000.json'),
    ('mm 3+3 (new)', 3,
     'item.qwen_separate.Qwen3-VL-Embedding-8B.t3x256.i3x256.seed2024.h128.feat',
     'actionpiece.qwen_separate.Qwen3-VL-Embedding-8B.t3x256.i3x256.'
     'seed2024.h128.v40000.json'),
    ('mm 4+4 (new)', 4,
     'item.qwen_separate.Qwen3-VL-Embedding-8B.t4x256.i4x256.seed2024.h128.feat',
     'actionpiece.qwen_separate.Qwen3-VL-Embedding-8B.t4x256.i4x256.'
     'seed2024.h128.v40000.json'),
    ('text only 4x256', 4,
     'item.qwen_text.qwen_local.Qwen3-VL-Embedding-8B.d768.i711daf44f948.'
     'r2c4565515e0f.c393e2978d278.m2048.tbfloat16.asdpa.opq4x256.seed2024.'
     'h128.feat', None),
    ('image only 4x256', 4,
     'item.qwen_image.qwen_local.Qwen3-VL-Embedding-8B.d768.ib6b0d724b414.'
     'r2c4565515e0f.c393e2978d278.m2048.tbfloat16.asdpa.image.'
     'im2f0852d8f8d9.opq4x256.seed2024.h128.feat', None),
    ('joint 4096 4x256', 4,
     'item.qwen_multimodal.qwen_local.Qwen3-VL-Embedding-8B.d4096.'
     'i711daf44f948.r2c4565515e0f.c393e2978d278.m2048.tbfloat16.asdpa.'
     'joint.im2f0852d8f8d9.opq4x256.seed2024.h128.feat', None),
    ('separate 2x65536 (old)', 2,
     'item.qwen_separate.Qwen3-VL-Embedding-8B.t768.th235523e6f242.i768.'
     'ihce07c0028c77.topq2x65536.iopq2x65536.seed2024.h128.feat', None),
]


def read_scalars(sub, tag):
    """Read a scalar from the newest run directory matching `sub`.

    Newest, not first-alphabetically: an arm that was restarted leaves an
    abandoned TensorBoard directory behind, and picking that one would report
    the aborted run's numbers.
    """
    best_dir, best_mtime = None, -1.0
    for run in os.listdir(TB):
        if sub not in run:
            continue
        d = os.path.join(TB, run)
        if not os.path.isdir(d):
            continue
        evs = [f for f in os.listdir(d) if f.startswith('events.out.tfevents')]
        if not evs:
            continue
        m = max(os.path.getmtime(os.path.join(d, f)) for f in evs)
        if m > best_mtime:
            best_dir, best_mtime = os.path.join(d, evs[0]), m
    if best_dir is None:
        return []
    acc = EventAccumulator(best_dir, size_guidance={'scalars': 0})
    acc.Reload()
    if tag in acc.Tags().get('scalars', []):
        return acc.Scalars(tag)
    return []


def best(sub, split, metric):
    s = read_scalars(sub, f'{split}_Metric/{metric}')
    if not s:
        return None, None
    vals = [x.value for x in s]
    return max(vals), vals.index(max(vals)) + 1


def sid_cardinality(text_file, image_file):
    out = [None, None]
    for idx, fn in enumerate((text_file, image_file)):
        if not fn:
            continue
        p = os.path.join(B, fn)
        if os.path.exists(p):
            d = json.load(open(p))
            out[idx] = len({tuple(v) for v in d.values()})
    return out


def slot_sets(t2f):
    out = [None] * len(t2f)
    for start in range(len(t2f)):
        if out[start] is not None:
            continue
        stack = [start]
        while stack:
            i = stack[-1]
            if out[i] is not None:
                stack.pop()
                continue
            e = t2f[i]
            if len(e) == 2:
                out[i] = frozenset((e[0],))
                stack.pop()
                continue
            a, b = e[1], e[2]
            if out[a] is None:
                stack.append(a)
                continue
            if out[b] is None:
                stack.append(b)
                continue
            out[i] = out[a] | out[b]
            stack.pop()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--md', default=None)
    args = ap.parse_args()

    L = []

    # ---- main comparison ---------------------------------------------------
    L.append('### Main comparison\n')
    L.append('| arm | text codes | image codes | slots (+hash) | distinct text SIDs | '
             'distinct image SIDs | val nDCG@10 | val R@10 | test nDCG@10 | '
             'test R@10 | best epoch |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for (label, sub, tc, tb_, ic, ib, _, _, tf, imf) in ARMS:
        v10, ep = best(sub, 'Val', 'ndcg@10')
        vr10, _ = best(sub, 'Val', 'recall@10')
        t10, _ = best(sub, 'Test', 'ndcg@10')
        tr10, _ = best(sub, 'Test', 'recall@10')
        dt, di = sid_cardinality(tf, imf)
        L.append('| %s | %s x %s | %s x %s | %s | %s | %s | %s | %s | %s | %s | %s |'
                 % (label, tc or '-', tb_ or '-', ic or '-', ib or '-',
                    (tc + ic + 1) if (tc or ic) else '-',
                    dt or '-', di or '-',
                    '%.4f' % v10 if v10 else '-',
                    '%.4f' % vr10 if vr10 else '-',
                    '%.4f' % t10 if t10 else '-',
                    '%.4f' % tr10 if tr10 else '-',
                    ep or '-'))

    # ---- semantic identity cardinality ------------------------------------
    L.append('')
    L.append('### How much of each item identity is semantic vs a random hash digit\n')
    L.append('| arm | feature width | distinct full identities | distinct semantic tuples | '
             'share needing the hash | largest semantic group |')
    L.append('|---|---|---|---|---|---|')
    for label, n, feat_fn, _ in FEAT_FILES:
        p = os.path.join(B, feat_fn)
        if not os.path.exists(p):
            L.append('| %s | - | - | - | - | - |' % label)
            continue
        d = json.load(open(p))
        feats = [tuple(v) for v in d.values()]
        w = len(feats[0])
        sem = [f[:w - 1] for f in feats]
        cnt = Counter(sem)
        share = 1.0 - len(set(sem)) / len(feats)
        L.append('| %s | %d | %d | %d | %.3f | %d |'
                 % (label, w, len(set(feats)), len(set(sem)), share,
                    max(cnt.values())))

    # ---- vocabulary modality composition ----------------------------------
    L.append('')
    L.append('### Vocabulary modality composition (expand of the merge tree)\n')
    L.append('| arm | merged tokens | type: contains hash | weighted: contains hash | '
             'type: cross-modal | weighted: cross-modal |')
    L.append('|---|---|---|---|---|---|')
    for label, n, _, vocab_fn in FEAT_FILES:
        if not vocab_fn:
            continue
        p = os.path.join(B, vocab_fn)
        if not os.path.exists(p):
            continue
        meta = json.load(open(p))
        t2f, prio = meta['token2feat'], meta['priority']
        slots = slot_sets(t2f)
        text_slots = set(range(n))
        image_slots = set(range(n, 2 * n))
        hash_slot = 2 * n
        nh = nb = nm = 0
        wh = wb = wa = 0.0
        for i in range(len(t2f)):
            if len(t2f[i]) != 3:
                continue
            nm += 1
            w = float(prio[i])
            wa += w
            s = slots[i]
            if hash_slot in s:
                nh += 1
                wh += w
            if (s & text_slots) and (s & image_slots):
                nb += 1
                wb += w
        L.append('| %s | %d | %.1f%% | %.1f%% | %.1f%% | %.1f%% |'
                 % (label, nm, 100 * nh / nm, 100 * wh / wa,
                    100 * nb / nm, 100 * wb / wa))

    # ---- caveats -----------------------------------------------------------
    L.append('')
    L.append('### Caveats that must travel with these numbers\n')
    for c in [
        'All arms share `actionpiece_vocab_size=40000`, `n_hash_buckets=128`, '
        '`num_beams=50`, `n_inference_ensemble=5`, `max_item_seq_len=20`, '
        '`epochs=200` with no early stopping, so the protocol matches.',
        'The legacy "image only" and "joint 4096" baselines encoded images with '
        'CLIP ViT-B/32 (512-d); the new arms and the legacy "separate" baseline '
        'used Qwen3-VL-Embedding-8B (768-d). Image-side comparisons across that '
        'line are not like for like.',
        'Codebook size differs between the new arms (256) and the legacy '
        'separate baseline (65536).',
        'The legacy separate run never completed its test pass '
        '(`torch.AcceleratorError` at 69%), so it has validation numbers only.',
        'Batch sizes differ: new arms train 128 / evaluate 32 (4+4 evaluates at '
        '16 to avoid a beam-search OOM); legacy arms used 256 / 64. Batch size '
        'changes batching, not the metric.',
        'Every legacy arm peaks late (best epochs 136-172), so intermediate-epoch '
        'rankings do not predict final rankings.',
    ]:
        L.append('- %s' % c)

    text = '\n'.join(L)
    print(text)
    if args.md:
        with open(args.md, 'w') as fh:
            fh.write(text + '\n')
        print()
        print('written:', args.md)


if __name__ == '__main__':
    main()
