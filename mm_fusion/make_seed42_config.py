"""Create the seed-42 replica of the 2+2 arm.

What changes relative to mm_sep22_beauty.yaml:

  + rand_seed: 42
      the arm config does not set it, so it was inherited from
      genrec/default.yaml.  `genrec/pipeline.py:84` uses it to seed numpy and
      torch globally, which is what drives the unseeded
      `np.random.permutation` in `_get_hashed_feat`.
  + tokenizer_seed: 42
      made explicit rather than left to default to rand_seed, so the artifact
      naming cannot drift if rand_seed is ever edited again.
  qwen_separate_artifact_stem: ...seed2024... -> ...seed42...
      this string names BOTH `item.<stem>.feat` and
      `actionpiece.<stem>.v40000.json`, so the item features and the ActionPiece
      vocabulary are rebuilt rather than reused.

What does NOT change: the OPQ quantiser. `_emb_to_sem_id` passes no seed to
faiss, so the semantic codes are identical in content and only their file names
move. The genuinely re-randomised parts are the hash buckets, the vocabulary
built on top of them, and the training run.
"""
import re

AP = '/home/lzr/science/actionpiece_exp/experiments'
SRC = f'{AP}/mm_sep22_beauty.yaml'
DST = f'{AP}/mm_sep22_seed42_beauty.yaml'
NEW_SEED = 42

src = open(SRC).read()

# 1. the artifact stem, which is what forces a rebuild
assert 't2x256.i2x256.seed2024.h128' in src
src = src.replace('t2x256.i2x256.seed2024.h128',
                  f't2x256.i2x256.seed{NEW_SEED}.h128')

# 2. the seeds, inserted right after `metadata:` since the arm config has none
assert not re.search(r'^rand_seed:', src, re.M), 'arm config already has rand_seed'
src = re.sub(
    r'^(metadata:.*)$',
    r'\1\n'
    f'# seed replica: rand_seed is inherited from genrec/default.yaml as 2024,\n'
    f'# so it has to be set explicitly here\n'
    f'rand_seed: {NEW_SEED}\n'
    f'tokenizer_seed: {NEW_SEED}',
    src, count=1, flags=re.M)

# 3. its own run id
src = re.sub(r'^run_id:.*$', f'run_id: beauty_mm_sep22_seed{NEW_SEED}',
             src, flags=re.M)

# 4. header explaining itself
src = src.replace(
    '# Multimodal late fusion on Amazon Reviews 2014 / Beauty.',
    '# Multimodal late fusion on Amazon Reviews 2014 / Beauty.\n'
    f'# SEED REPLICA of mm_sep22_beauty.yaml with rand_seed={NEW_SEED}.')

open(DST, 'w').write(src)
print('wrote', DST)
print()
for line in open(DST):
    if re.match(r'^(rand_seed|tokenizer_seed|run_id|qwen_separate_artifact_stem|'
                r'pq_n_codebooks|pq_codebook_size|image_pq_n_codebooks|'
                r'image_pq_codebook_size|actionpiece_vocab_size|'
                r'n_hash_buckets|epochs|train_batch_size|eval_batch_size|'
                r'tokenizer_only|metadata|category):', line):
        print('  ' + line.rstrip())
