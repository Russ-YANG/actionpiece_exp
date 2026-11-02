"""Prepare (but do not launch) a large-codebook variant of the 3+3 arm.

Motivation: with 256-entry codebooks the effective semantic-ID cardinality is
very low (2,453-7,047 distinct IDs for 12,101 items).  The 3+3 arm fits the
training set *better* than every baseline while generalising *worse*, which is
the signature of memorising a coarse token space.  Raising the codebook size
keeps the 3+3 slot layout but gives the quantiser room to separate items.

This script only writes the config; nothing is trained.
"""
import os
import re

AP = '/home/lzr/science/actionpiece_exp/experiments'
SRC = os.path.join(AP, 'mm_sep33_beauty.yaml')

for cb in (4096, 65536):
    dst = os.path.join(AP, f'mm_sep33_cb{cb}_beauty.yaml')
    src = open(SRC).read()

    # distinct artifact stem so this arm cannot collide with mm_sep33
    src = src.replace(
        'qwen_separate_artifact_stem: qwen_separate.Qwen3-VL-Embedding-8B.'
        't3x256.i3x256.seed2024.h128',
        f'qwen_separate_artifact_stem: qwen_separate.Qwen3-VL-Embedding-8B.'
        f't3x{cb}.i3x{cb}.seed2024.h128')
    src = re.sub(r'^pq_codebook_size: \d+$', f'pq_codebook_size: {cb}',
                 src, flags=re.M)
    src = re.sub(r'^image_pq_codebook_size: \d+$',
                 f'image_pq_codebook_size: {cb}', src, flags=re.M)
    src = re.sub(r'^run_id: .*$', f'run_id: beauty_mm_sep33_cb{cb}', src,
                 flags=re.M)
    # make the header explain itself
    src = src.replace(
        '# Multimodal late fusion on Amazon Reviews 2014 / Beauty.',
        '# Multimodal late fusion on Amazon Reviews 2014 / Beauty.\n'
        f'# VARIANT: 3 text + 3 image codes with {cb}-entry codebooks.\n'
        '# Identical to mm_sep33_beauty.yaml except the codebook size, to test\n'
        '# whether the coarse 256-entry books are what limits generalisation.')
    open(dst, 'w').write(src)
    print('wrote', dst)

print()
for cb in (4096, 65536):
    p = os.path.join(AP, f'mm_sep33_cb{cb}_beauty.yaml')
    keys = {}
    for line in open(p):
        if ':' in line and not line.startswith((' ', '#')):
            k, _, v = line.partition(':')
            keys[k.strip()] = v.strip()
    print('cb%-6d pq_codebook_size=%-6s image=%-6s stem=%s'
          % (cb, keys.get('pq_codebook_size'),
             keys.get('image_pq_codebook_size'),
             keys.get('qwen_separate_artifact_stem')))
