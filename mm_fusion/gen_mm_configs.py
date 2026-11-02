"""Generate the 2+2 / 3+3 / 4+4 multimodal late-fusion configs for Beauty."""
import os

AP = '/home/lzr/science/actionpiece_exp'
OUT = os.path.join(AP, 'experiments')

TEMPLATE = '''\
# Multimodal late fusion on Amazon Reviews 2014 / Beauty.
#
# Text semantic IDs and image semantic IDs are quantised INDEPENDENTLY and then
# concatenated into one ActionPiece item representation:
#     {n} text codes + {n} image codes = {slots} slots
#
# All arms share encoder, embeddings, seed, hash buckets, vocabulary size and
# model hyper-parameters; only the number of codebooks per modality differs, so
# the arms are directly comparable.
#
# Needs the tokenizer patch that lets metadata=qwen_separate take the
# text+image concatenation branch (tokenizer.py::_get_item2feat).
category: Beauty
metadata: qwen_separate
qwen_separate_artifact_stem: qwen_separate.Qwen3-VL-Embedding-8B.t{n}x256.i{n}x256.seed2024.h128

sent_emb_backend: qwen_local
sent_emb_model: Qwen/Qwen3-VL-Embedding-8B
sent_emb_dim: 768
sent_emb_pca: -1
qwen_api_instruction: >-
  Represent this e-commerce product for semantic similarity and sequential
  recommendation. Capture its category, function, attributes, style, and likely
  user intent.
qwen_local_image_instruction: >-
  Represent this e-commerce product image for semantic similarity and
  sequential recommendation. Capture its visible category, function,
  attributes, style, packaging, and likely user intent.
qwen_local_model_path: /root/autodl-tmp/models/Qwen3-VL-Embedding-8B
qwen_local_model_revision: 2c4565515e0f265c6511776e7193b22c0968ddc7
qwen_local_repo_path: /root/autodl-tmp/qwen3-vl-embedding
qwen_local_code_revision: 393e2978d27852b0d0230d6994f37f9c15bed73c
qwen_local_batch_size: 8
qwen_local_max_length: 2048
qwen_local_torch_dtype: bfloat16
qwen_local_attn_implementation: sdpa
qwen_local_require_cuda: true

image_emb_model: Qwen/Qwen3-VL-Embedding-8B
image_emb_dim: 768

# one independent quantiser per modality
pq_n_codebooks: {n}
pq_codebook_size: 256
image_pq_n_codebooks: {n}
image_pq_codebook_size: 256

n_hash_buckets: 128
actionpiece_vocab_size: 40000
actionpiece_merge_log: false
n_threads: 8

tokenizer_only: true
train_batch_size: 128
eval_batch_size: 64
epochs: 200
run_id: beauty_mm_sep{arm}_d768
'''

for n in (2, 3, 4):
    arm = '%d%d' % (n, n)
    path = os.path.join(OUT, 'mm_sep%s_beauty.yaml' % arm)
    with open(path, 'w') as fh:
        fh.write(TEMPLATE.format(n=n, arm=arm, slots=2 * n))
    print('wrote', path)

print()
for n in (2, 3, 4):
    arm = '%d%d' % (n, n)
    path = os.path.join(OUT, 'mm_sep%s_beauty.yaml' % arm)
    keys = {}
    for line in open(path):
        if ':' in line and not line.startswith(' ') and not line.startswith('#'):
            k, _, v = line.partition(':')
            keys[k.strip()] = v.strip()
    print('arm %s: pq_n_codebooks=%s image_pq_n_codebooks=%s stem=%s'
          % (arm, keys.get('pq_n_codebooks'), keys.get('image_pq_n_codebooks'),
             keys.get('qwen_separate_artifact_stem')))
