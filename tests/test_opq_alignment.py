"""CPU contract tests with tiny real T5; all data here are synthetic fixtures."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np
import torch
import yaml
from accelerate import Accelerator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
# Match the production import order to resolve the existing framework cycle.
from genrec.pipeline import Pipeline
from genrec.models.ActionPiece.model import ActionPiece
from genrec.models.ActionPiece.tokenizer import ActionPieceTokenizer
from genrec.opq_alignment import add_alignment, load_alignment_inputs, pair_diagnostics, _sha256
from opq_baseline import AtomicCodec


class AlignmentTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = yaml.safe_load((ROOT / 'genrec/models/ActionPiece/config.yaml').read_text())
        self.config.update(
            accelerator=Accelerator(cpu=True), rand_seed=2024,
            n_inference_ensemble=-1, train_shuffle='feature',
            target_tokenization='atomic', max_item_seq_len=2,
            beam_search_mode='eos_aware', num_layers=1, num_decoder_layers=1,
            d_model=16, d_ff=32, num_heads=2, d_kv=8, dropout_rate=0.1,
            alignment_weight=0.1,
        )
        self.mapping = {'id2item': ['[PAD]', 'a', 'b', 'c', 'target']}
        # a/b deliberately have identical four-code SID, different disambiguators.
        self.features = {'a': (1, 2, 3, 4, 7), 'b': (1, 2, 3, 4, 8),
                         'c': (9, 10, 11, 12, 9), 'target': (2, 3, 4, 5, 10)}
        self.text = np.random.RandomState(4).normal(size=(4, 8)).astype('float32')
        features = self.features

        class Tokenizer(ActionPieceTokenizer):
            def _init_tokenizer(inner, dataset):
                inner.item2feat = features
                return AtomicCodec(features, inner.config['rand_seed'])

        class Model(ActionPiece):
            def forward(inner, batch):
                return inner.t5(**batch)

        self.BaseModel, self.BaseTokenizer = Model, Tokenizer
        self.dataset = types.SimpleNamespace()

    def build(self, weight):
        config = self.config | {'alignment_weight': weight}
        Model, Tokenizer = add_alignment(
            self.BaseModel, self.BaseTokenizer, config, self.mapping, self.features, self.text
        )
        tokenizer = Tokenizer(config, self.dataset)
        return Model(config, self.dataset, tokenizer), tokenizer

    def batch(self, tokenizer):
        return tokenizer.collate_fn_train([
            {'state_seq': tokenizer._tokenize_once(['a', 'b', 'target'])},
            {'state_seq': tokenizer._tokenize_once(['c', 'target'])},
        ])

    def test_zero_weight_exact_ce_gradients_rng_and_optimizer_step(self):
        torch.manual_seed(73)
        base_tokenizer = self.BaseTokenizer(self.config, self.dataset)
        base = self.BaseModel(self.config, self.dataset, base_tokenizer)
        baseline_rng = torch.get_rng_state().clone()
        torch.manual_seed(73)
        model, tokenizer = self.build(0.0)
        self.assertTrue(torch.equal(baseline_rng, torch.get_rng_state()))
        for key, value in base.t5.state_dict().items():
            self.assertTrue(torch.equal(value, model.t5.state_dict()[key]), key)
        old_batch = self.batch(base_tokenizer)
        batch = self.batch(tokenizer)
        for key in old_batch:
            self.assertTrue(torch.equal(old_batch[key], batch[key]), key)
        torch.manual_seed(91)
        old = base(old_batch)
        after_old = torch.get_rng_state().clone()
        torch.manual_seed(91)
        new = model(batch)
        self.assertTrue(torch.equal(after_old, torch.get_rng_state()))
        self.assertTrue(torch.equal(old.logits, new.logits))
        self.assertTrue(torch.equal(old.loss, new.loss))
        old.loss.backward()
        new.loss.backward()
        for (name, parameter), (_, aligned) in zip(base.t5.named_parameters(), model.t5.named_parameters()):
            self.assertTrue(torch.equal(parameter.grad, aligned.grad), name)
        self.assertIsNone(model.alignment_sid_projection.weight.grad)
        for item in (base, model):
            torch.nn.utils.clip_grad_norm_(item.parameters(), 1.0)
            torch.optim.AdamW(item.parameters(), lr=0.003).step()
        for key, value in base.t5.state_dict().items():
            self.assertTrue(torch.equal(value, model.t5.state_dict()[key]), key)

    def test_history_occurrences_collisions_padding_and_fixed_slot_order(self):
        model, tokenizer = self.build(0.1)
        batch = self.batch(tokenizer)
        self.assertEqual(batch['alignment_rows'].tolist(), [0, 1, 2])
        self.assertEqual(model.alignment_codes[0].tolist(), model.alignment_codes[1].tolist())
        self.assertFalse(torch.equal(model.alignment_text[0], model.alignment_text[1]))
        self.assertEqual(model.alignment_codes[0].tolist(), [2, 259, 516, 773])
        self.assertEqual(batch['labels'].shape, (2, 6))
        # Padding is not an alignment occurrence. The target row (3) is excluded.
        self.assertEqual(len(batch['alignment_rows']), 3)
        self.assertEqual(batch['attention_mask'][1].sum().item(), 7)
        self.assertEqual(batch['input_ids'].shape[1], 12)
        # Auxiliary row construction does not disturb the legacy feature shuffle.
        plain = self.BaseTokenizer(self.config, self.dataset)
        old_batch = self.batch(plain)
        self.assertTrue(torch.equal(old_batch['input_ids'], batch['input_ids']))

    def test_pair_diagnostics_detect_collapse_duplicates_and_collisions(self):
        rows = torch.tensor([0, 1, 2, 3])
        aligned = torch.eye(4)
        positive, negative, gap, top1 = pair_diagnostics(aligned, aligned.clone(), rows)
        self.assertAlmostEqual(positive.item(), 1.0, places=6)
        self.assertAlmostEqual(negative.item(), 0.0, places=6)
        self.assertAlmostEqual(top1.item(), 1.0, places=6)
        # Collapse: every item maps to one direction. Mean cosine alone says 1.
        constant = torch.ones(4, 3)
        positive, negative, gap, top1 = pair_diagnostics(constant, constant.clone(), rows)
        self.assertAlmostEqual(positive.item(), 1.0, places=5)
        self.assertAlmostEqual(gap.item(), 0.0, places=5)
        self.assertLess(top1.item(), 1.0)
        # Repeated occurrences of one item are not negatives of each other.
        z = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
        _, negative, gap, _ = pair_diagnostics(z, z.clone(), torch.tensor([5, 5, 6]))
        self.assertAlmostEqual(negative.item(), 0.0, places=6)
        self.assertAlmostEqual(gap.item(), 1.0, places=6)
        # Four-code collisions (same z, different rows/texts) stay negatives.
        z = torch.tensor([[1., 0.], [1., 0.]])
        target = torch.tensor([[1., 0.], [0.6, 0.8]])
        _, negative, _, top1 = pair_diagnostics(z, target, torch.tensor([0, 1]))
        self.assertAlmostEqual(negative.item(), 0.8, places=5)
        self.assertAlmostEqual(top1.item(), 0.5, places=6)
        _, negative, gap, _ = pair_diagnostics(z, z.clone(), torch.tensor([7, 7]))
        self.assertTrue(torch.isnan(negative) and torch.isnan(gap))
        # Strided subsample bounds the matrix without consuming RNG.
        state = torch.get_rng_state().clone()
        big = torch.eye(10).repeat(300, 1)
        pair_diagnostics(big, big.clone(), torch.arange(3000) % 10, limit=64)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_alignment_gradients_checkpoint_and_generation_unchanged(self):
        torch.manual_seed(12)
        model, tokenizer = self.build(0.1)
        batch = self.batch(tokenizer)
        # Disable dropout to isolate the alignment gradient from the CE graph.
        model.eval()
        output = model(batch)
        for key in ('alignment_negative_cosine', 'alignment_cosine_gap', 'alignment_top1'):
            self.assertTrue(torch.isfinite(output[key]).item(), key)
            self.assertFalse(output[key].requires_grad, key)
        ce = model.t5(**{k: v for k, v in batch.items() if k != 'alignment_rows'}).loss
        (output.loss - ce).backward()
        self.assertGreater(model.t5.get_input_embeddings().weight.grad.abs().sum().item(), 0)
        for projection in (model.alignment_sid_projection, model.alignment_text_projection):
            self.assertGreater(projection.weight.grad.abs().sum().item(), 0)
        self.assertIsNone(model.alignment_text.grad)
        self.assertFalse(any(key in model.state_dict() for key in ('alignment_text', 'alignment_codes')))
        clone, _ = self.build(0.1)
        clone.load_state_dict(model.state_dict())
        clone.eval()
        old = self.BaseModel(self.config, self.dataset, tokenizer)
        old.t5.load_state_dict(model.t5.state_dict())
        old.eval()
        for other in (old, model, clone):
            with torch.no_grad():
                result = other.beam_search(
                    batch['input_ids'], batch['attention_mask'], tokenizer.generation_max_length,
                    num_beams=2, num_return_sequences=1,
                )
            if other is old:
                expected = result
            else:
                self.assertTrue(torch.equal(expected, result))

    def test_real_trainer_saves_checkpoint_and_alignment_metrics(self):
        from torch.utils.data import DataLoader
        from genrec.trainer import Trainer
        defaults = yaml.safe_load((ROOT / 'genrec/default.yaml').read_text())
        self.config = defaults | self.config
        with tempfile.TemporaryDirectory(prefix='alignment-cpu-trainer-') as tmp:
            self.config.update(epochs=1, steps=None, warmup_steps=0, topk=[1],
                               metrics=['ndcg', 'recall', 'err'], val_metric='ndcg@1',
                               num_beams=2, use_ddp=False, ckpt_dir=tmp,
                               run_id='synthetic-cpu-test', run_local_time='test')
            model, tokenizer = self.build(0.1)
            examples = [{'state_seq': tokenizer._tokenize_once(['a', 'b', 'target'])},
                        {'state_seq': tokenizer._tokenize_once(['c', 'target'])}]
            train = DataLoader(examples, batch_size=2, collate_fn=tokenizer.collate_fn_train)
            valid = DataLoader(examples, batch_size=2, collate_fn=tokenizer.collate_fn_val)
            trainer = Trainer(self.config, model, tokenizer)
            messages = []
            trainer.log = messages.append
            trainer.fit(train, valid)
            self.assertTrue(Path(trainer.saved_model_ckpt).is_file())
            self.assertTrue(any('Alignment:' in message and 'alignment_cosine' in message
                                and 'alignment_cosine_gap' in message
                                and 'alignment_top1' in message
                                for message in messages))
            self.assertTrue(any('Val Results:' in message for message in messages))

    def test_provenance_rejects_api_and_mismatched_order_or_content(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-alignment-contract-') as tmp:
            root = Path(tmp)
            items = self.mapping['id2item'][1:]
            mapping = self.mapping | {'item2id': {name: i for i, name in enumerate(self.mapping['id2item'])}}
            (root / 'id_mapping.json').write_text(json.dumps(mapping))
            (root / 'all_item_seqs.json').write_text(json.dumps({'u': ['a', 'b', 'c', 'target']}))
            self.text.tofile(root / 'text.f32')
            # Contract fixture only; not an actual embedding artifact or experiment result.
            meta = dict(backend='qwen_local', model_id='Qwen/Qwen3-VL-Embedding-8B',
                        model_revision='contract-fixture', code_revision='contract-fixture',
                        instruction_sha256='contract-fixture', normalization='truncate_then_l2',
                        dimension=8, item_count=4,
                        item_order_sha256=hashlib.sha256('\n'.join(items).encode()).hexdigest())
            (root / 'text.manifest.json').write_text(json.dumps(meta))
            bundle = dict(embedding_source=meta, signature=dict(n_codebooks=4, codebook_size=256,
                          dimension=8, preprocessing='truncate_then_l2'), inputs={})
            for key, name in [('embedding_path','text.f32'), ('embedding_manifest_path','text.manifest.json'),
                              ('id_mapping_path','id_mapping.json'), ('item_seqs_path','all_item_seqs.json')]:
                bundle['inputs'][key] = {'sha256': _sha256(root / name)}
            (root / 'manifest.json').write_text(json.dumps(bundle))
            config = dict(alignment_weight=0.1, sid_path=str(root/'item.sem_ids'), processed_dir=str(root),
                          alignment_embedding_path=str(root/'text.f32'),
                          alignment_embedding_manifest_path=str(root/'text.manifest.json'))
            text, _ = load_alignment_inputs(config, mapping)
            self.assertTrue(np.array_equal(text, self.text))
            (root / 'text.manifest.json').write_text(json.dumps(meta | {'backend': 'qwen_api'}))
            with self.assertRaisesRegex(ValueError, 'local Qwen'):
                load_alignment_inputs(config, mapping)
            (root / 'text.manifest.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, 'order'):
                load_alignment_inputs(config, mapping | {'id2item': ['[PAD]', 'b', 'a', 'c', 'target']})
            with (root / 'text.f32').open('r+b') as stream:
                stream.write(b'bad!')
            with self.assertRaisesRegex(ValueError, 'embedding_path'):
                load_alignment_inputs(config, mapping)


if __name__ == '__main__':
    unittest.main()
