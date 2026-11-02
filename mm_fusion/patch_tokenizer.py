"""Patch ActionPiece tokenizer.py so qwen_separate really fuses text+image SIDs.

Two changes:
  A. give image semantic IDs a cache name that includes the quantiser settings,
     so the 2+2 / 3+3 / 4+4 arms cannot collide;
  B. let `metadata: qwen_separate` take the same text+image concatenation branch
     that `sentence_image` already uses (tokenizer.py:766 in the original).

Run inside WSL.  Prints a unified diff; writes only if both anchors match once.
"""
import difflib
import sys

PATH = ('/home/lzr/science/actionpiece_exp/genrec/models/ActionPiece/'
        'tokenizer.py')

A_OLD = """  def _get_image_sem_ids(self, dataset: AbstractDataset) -> dict[Any, Any]:
    image_sem_ids_path = os.path.join(
        dataset.cache_dir,
        'processed',
        f'{os.path.basename(self.config["image_emb_model"])}.image_sem_ids',
    )
"""

A_NEW = """  def _image_semantic_artifact_stem(self) -> str:
    \"\"\"Cache name for image semantic IDs.

    The quantiser settings must be part of the name; otherwise arms that differ
    only in the number of image codebooks (2+2 vs 3+3 vs 4+4) would read and
    overwrite each other's cache.
    \"\"\"
    model_name = os.path.basename(self.config['image_emb_model'])
    safe_model_name = re.sub(r'[^A-Za-z0-9._-]+', '-', model_name)
    tokenizer_seed = self.config.get('tokenizer_seed')
    if tokenizer_seed is None:
      tokenizer_seed = self.config['rand_seed']
    return (
        f'{safe_model_name}.d{self.config["image_emb_dim"]}.'
        f'opq{self.config["image_pq_n_codebooks"]}x'
        f'{self.config["image_pq_codebook_size"]}.seed{tokenizer_seed}'
    )

  def _get_image_sem_ids(self, dataset: AbstractDataset) -> dict[Any, Any]:
    image_sem_ids_path = os.path.join(
        dataset.cache_dir,
        'processed',
        f'{self._image_semantic_artifact_stem()}.image_sem_ids',
    )
"""

B_OLD = """    if self.config['metadata'] == 'sentence_image':
      item2image_sem_ids = self._get_image_sem_ids(dataset)
"""

B_NEW = """    if self.config['metadata'] in ('sentence_image', 'qwen_separate'):
      item2image_sem_ids = self._get_image_sem_ids(dataset)
"""

src = open(PATH).read()
if 'qwen_separate' in src and '_image_semantic_artifact_stem' in src:
    print('already patched, nothing to do')
    sys.exit(0)

for name, old, new in (('A', A_OLD, A_NEW), ('B', B_OLD, B_NEW)):
    n = src.count(old)
    print('anchor %s occurrences: %d' % (name, n))
    if n != 1:
        print('ABORT: anchor %s must appear exactly once' % name)
        sys.exit(1)

out = src.replace(A_OLD, A_NEW).replace(B_OLD, B_NEW)

diff = difflib.unified_diff(src.splitlines(True), out.splitlines(True),
                            'tokenizer.py (before)', 'tokenizer.py (after)',
                            n=2)
sys.stdout.writelines(diff)
open(PATH, 'w').write(out)
print()
print('written:', PATH)
