"""Validate an ActionPiece vocabulary before training on it.

The item representation is NOT just the semantic IDs: `_get_item2feat` appends
(optionally) attribute IDs and always one hash-bucket digit from
`_get_hashed_feat` (to disambiguate colliding semantic IDs).  So

    2+2 -> 2+2+1 = 5 categories
    3+3 -> 3+3+1 = 7 categories
    4+4 -> 4+4+1 = 9 categories

The expected width is taken from the `.feat` file that was actually written for
the arm and then cross-checked against the vocabulary, rather than assumed.  That
catches a truncated save, a wrong artifact stem, or a mismatch between the two
files.

Deliberately does NOT import genrec: importing
`genrec.models.ActionPiece.core` pulls in `genrec/models/__init__.py` and hits a
circular import unless the package is entered the way main.py does it.

Usage:
    validate_vocab.py                       # the three seed2024 arms
    validate_vocab.py 22 33 44              # same, explicit
    validate_vocab.py mm_sep22_seed42       # a config stem: reads its
                                            # qwen_separate_artifact_stem
"""
import json
import os
import re
import sys

AP = '/home/lzr/science/actionpiece_exp'
B = f'{AP}/cache/AmazonReviews2014/Beauty/processed'
VOCAB = 40000


def resolve(spec):
    """Return (label, artifact_stem, n_codebooks) for an arm number or config."""
    if re.fullmatch(r'\d{2}', spec):
        n = int(spec[0])
        return (spec,
                f'qwen_separate.Qwen3-VL-Embedding-8B.'
                f't{n}x256.i{n}x256.seed2024.h128', n)

    cfg = f'{AP}/experiments/{spec}_beauty.yaml'
    if not os.path.exists(cfg):
        return (spec, None, None)
    txt = open(cfg).read()

    def get(key):
        m = re.search(r'^%s:\s*(\S+)' % key, txt, re.M)
        return m.group(1) if m else None

    stem = get('qwen_separate_artifact_stem')
    n = get('pq_n_codebooks')
    return (spec, stem, int(n) if n else None)


def check(spec):
    label, stem, n = resolve(spec)
    if stem is None:
        print('%-22s UNRESOLVED (no arm number and no such config)' % label)
        return False
    feat_path = os.path.join(B, f'item.{stem}.feat')
    vocab_path = os.path.join(B, f'actionpiece.{stem}.v{VOCAB}.json')

    if not os.path.exists(feat_path):
        print('%-22s MISSING feat  %s' % (label, os.path.basename(feat_path)))
        return False
    if not os.path.exists(vocab_path):
        print('%-22s MISSING vocab %s' % (label, os.path.basename(vocab_path)))
        return False

    try:
        with open(feat_path) as fh:
            feats = json.load(fh)
        with open(vocab_path) as fh:
            meta = json.load(fh)
    except Exception as exc:                      # noqa: BLE001
        print('%-22s unreadable JSON (%s: %s)' % (label, type(exc).__name__, exc))
        return False

    lengths = {len(v) for v in feats.values()}
    if len(lengths) != 1:
        print('%-22s inconsistent feat tuple lengths: %s' % (label, lengths))
        return False
    feat_len = lengths.pop()

    expected = 2 * n + 1
    problems = []
    if feat_len != expected:
        problems.append('feat tuples are %d wide, expected %d (2x%d semantic '
                        '+ 1 hash)' % (feat_len, expected, n))

    need = ('n_categories', 'n_init_feats', 'token2feat', 'priority')
    missing = [k for k in need if k not in meta]
    if missing:
        print('%-22s vocab missing keys %s' % (label, missing))
        return False

    n_tokens = len(meta['token2feat'])
    if meta['n_categories'] != feat_len:
        problems.append('vocab n_categories=%d but feat tuples are %d wide'
                        % (meta['n_categories'], feat_len))
    if n_tokens < VOCAB:
        problems.append('only %d tokens, need %d' % (n_tokens, VOCAB))
    if len(meta['priority']) != n_tokens:
        problems.append('priority %d != tokens %d'
                        % (len(meta['priority']), n_tokens))
    if meta['n_init_feats'] > n_tokens:
        problems.append('n_init_feats %d > tokens %d'
                        % (meta['n_init_feats'], n_tokens))

    if problems:
        print('%-22s INVALID: %s' % (label, '; '.join(problems)))
        return False

    print('%-22s ok  categories=%-2d (2x%d semantic + 1 hash)  init=%-5d '
          'vocab=%-6d feat=%d items'
          % (label, meta['n_categories'], n, meta['n_init_feats'], n_tokens,
             len(feats)))
    return True


specs = sys.argv[1:] or ['22', '33', '44']
results = [check(s) for s in specs]
print()
print('ALL CHECKED VOCABULARIES VALID' if all(results)
      else 'VALIDATION FAILED')
sys.exit(0 if all(results) else 1)
