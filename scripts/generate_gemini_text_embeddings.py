#!/usr/bin/env python3
"""Generate resumable Beauty text embeddings with Gemini Embedding 2."""

import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any

from google import genai
from google.genai import errors, types
import numpy as np


MODEL_ID = "gemini-embedding-2"
NATIVE_DIMENSION = 3072
DERIVED_DIMENSIONS = (1536, 768)
INSTRUCTION = (
    "Represent this e-commerce product for semantic similarity and sequential "
    "recommendation. Capture its category, function, attributes, style, and "
    "likely user intent."
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--processed-dir",
      default="cache/AmazonReviews2014/Beauty/processed",
  )
  parser.add_argument("--output-prefix")
  parser.add_argument("--limit", type=int)
  parser.add_argument("--request-size", type=int, default=100)
  parser.add_argument("--max-retries", type=int, default=10)
  parser.add_argument("--validate-only", action="store_true")
  return parser.parse_args()


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def write_json_atomic(value: Any, path: Path) -> None:
  temporary = path.with_suffix(path.suffix + ".tmp")
  with temporary.open("w", encoding="utf-8") as stream:
    json.dump(value, stream, ensure_ascii=False, indent=2)
  temporary.replace(path)


def load_api_key() -> str:
  key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
  if key:
    return key
  return getpass.getpass("Gemini API key (not stored): ")


def request_embeddings(
    client: genai.Client,
    prompts: list[str],
    max_retries: int,
) -> list[np.ndarray]:
  retryable_codes = {408, 429, 500, 502, 503, 504}
  for attempt in range(max_retries + 1):
    try:
      response = client.models.embed_content(
          model=MODEL_ID,
          contents=[
              types.Content(parts=[types.Part(text=prompt)])
              for prompt in prompts
          ],
          config=types.EmbedContentConfig(
              output_dimensionality=NATIVE_DIMENSION
          ),
      )
      if response.embeddings is None:
        raise RuntimeError("Gemini returned no embeddings.")
      if len(response.embeddings) != len(prompts):
        raise RuntimeError(
            f"Gemini returned {len(response.embeddings)} embeddings for "
            f"{len(prompts)} inputs."
        )
      vectors = []
      for embedding in response.embeddings:
        vector = np.asarray(embedding.values, dtype=np.float32)
        if vector.shape != (NATIVE_DIMENSION,):
          raise RuntimeError(f"Gemini returned shape {vector.shape}.")
        if not np.isfinite(vector).all():
          raise RuntimeError("Gemini returned NaN or Inf.")
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm == 0:
          raise RuntimeError(f"Gemini returned invalid norm {norm}.")
        vectors.append(vector / norm)
      return vectors
    except errors.APIError as exc:
      if exc.code not in retryable_codes or attempt == max_retries:
        raise
      delay = min(2 ** attempt, 60) + random.random()
      print(
          f"retryable Gemini error {exc.code}; waiting {delay:.1f}s",
          flush=True,
      )
      time.sleep(delay)
  raise AssertionError("Unreachable")


def derive_prefix_embeddings(
    source_path: Path,
    item_count: int,
    output_prefix: Path,
) -> dict[str, dict[str, Any]]:
  source = np.memmap(
      source_path,
      dtype=np.float32,
      mode="r",
      shape=(item_count, NATIVE_DIMENSION),
  )
  outputs: dict[str, dict[str, Any]] = {}
  for dimension in DERIVED_DIMENSIONS:
    output_path = Path(f"{output_prefix}.d{dimension}.sent_emb")
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    vectors = np.memmap(
        partial_path,
        dtype=np.float32,
        mode="w+",
        shape=(item_count, dimension),
    )
    for start in range(0, item_count, 512):
      end = min(start + 512, item_count)
      block = np.asarray(source[start:end, :dimension], dtype=np.float32)
      norms = np.linalg.norm(block, axis=1, keepdims=True)
      if np.any(norms == 0) or not np.isfinite(norms).all():
        raise RuntimeError(f"Invalid prefix norm at dimension {dimension}.")
      vectors[start:end] = block / norms
    vectors.flush()
    del vectors
    partial_path.replace(output_path)
    outputs[str(dimension)] = {
        "path": str(output_path),
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
    }
  del source
  return outputs


def verify_embeddings(path: Path, item_count: int, dimension: int) -> dict[str, float]:
  expected_bytes = item_count * dimension * np.dtype(np.float32).itemsize
  if path.stat().st_size != expected_bytes:
    raise RuntimeError(
        f"{path} has {path.stat().st_size} bytes; expected {expected_bytes}."
    )
  vectors = np.memmap(
      path, dtype=np.float32, mode="r", shape=(item_count, dimension)
  )
  if not np.isfinite(vectors).all():
    raise RuntimeError(f"{path} contains NaN or Inf.")
  norms = np.linalg.norm(vectors, axis=1)
  result = {"norm_min": float(norms.min()), "norm_max": float(norms.max())}
  del vectors
  return result


def main() -> None:
  args = parse_args()
  if args.limit is not None and args.limit <= 0:
    raise ValueError("--limit must be positive.")
  if args.request_size <= 0:
    raise ValueError("--request-size must be positive.")
  if args.max_retries < 0:
    raise ValueError("--max-retries cannot be negative.")

  processed_dir = Path(args.processed_dir)
  id_mapping_path = processed_dir / "id_mapping.json"
  metadata_path = processed_dir / "metadata.qwen_text.json"
  id_mapping = json.loads(id_mapping_path.read_text(encoding="utf-8"))
  metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
  item_ids = id_mapping["id2item"][1:]
  if args.limit is not None:
    item_ids = item_ids[:args.limit]
  missing_metadata = set(item_ids) - set(metadata)
  if missing_metadata:
    raise RuntimeError(f"Metadata is missing {len(missing_metadata)} items.")
  if len(item_ids) != len(set(item_ids)):
    raise RuntimeError("Item mapping contains duplicate item IDs.")

  instruction_hash = hashlib.sha256(INSTRUCTION.encode()).hexdigest()[:12]
  metadata_hash = sha256_file(metadata_path)[:12]
  default_prefix = processed_dir / (
      f"gemini_api.{MODEL_ID}.i{instruction_hash}.m{metadata_hash}.text"
  )
  output_prefix = Path(args.output_prefix) if args.output_prefix else default_prefix
  output_prefix.parent.mkdir(parents=True, exist_ok=True)
  output_path = Path(f"{output_prefix}.d{NATIVE_DIMENSION}.sent_emb")
  partial_path = output_path.with_suffix(output_path.suffix + ".partial")
  progress_path = Path(f"{output_prefix}.progress.json")
  manifest_path = Path(f"{output_prefix}.manifest.json")

  identity = {
      "model": MODEL_ID,
      "modality": "text",
      "native_dimension": NATIVE_DIMENSION,
      "derived_dimensions": list(DERIVED_DIMENSIONS),
      "item_count": len(item_ids),
      "id_mapping_sha256": sha256_file(id_mapping_path),
      "metadata_path": str(metadata_path),
      "metadata_sha256": sha256_file(metadata_path),
      "instruction": INSTRUCTION,
      "instruction_sha256": hashlib.sha256(INSTRUCTION.encode()).hexdigest(),
  }
  print(f"validated {len(item_ids)} text-only items")
  print(f"output prefix: {output_prefix}")
  if args.validate_only:
    return
  if output_path.exists() and manifest_path.exists():
    print(f"already complete: {manifest_path}")
    return

  completed = 0
  if progress_path.exists():
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if progress["identity"] != identity:
      raise RuntimeError("Existing progress belongs to a different input.")
    completed = progress["completed"]
    if not partial_path.exists():
      raise RuntimeError("Progress exists but partial embedding file is missing.")
  elif partial_path.exists():
    raise RuntimeError("Partial embedding file exists without progress metadata.")

  mode = "r+" if partial_path.exists() else "w+"
  output = np.memmap(
      partial_path,
      dtype=np.float32,
      mode=mode,
      shape=(len(item_ids), NATIVE_DIMENSION),
  )
  client = genai.Client(api_key=load_api_key())
  for start in range(completed, len(item_ids), args.request_size):
    end = min(start + args.request_size, len(item_ids))
    prompts = [
        f"{INSTRUCTION}\n\nProduct metadata:\n{metadata[item_id]}"
        for item_id in item_ids[start:end]
    ]
    vectors = request_embeddings(client, prompts, args.max_retries)
    output[start:end] = np.stack(vectors)
    output.flush()
    write_json_atomic(
        {"identity": identity, "completed": end, "total": len(item_ids)},
        progress_path,
    )
    print(f"embedded {end}/{len(item_ids)}", flush=True)
  del output
  partial_path.replace(output_path)

  outputs = {
      str(NATIVE_DIMENSION): {
          "path": str(output_path),
          "bytes": output_path.stat().st_size,
          "sha256": sha256_file(output_path),
      }
  }
  outputs.update(derive_prefix_embeddings(output_path, len(item_ids), output_prefix))
  validation = {
      dimension: verify_embeddings(Path(record["path"]), len(item_ids), int(dimension))
      for dimension, record in outputs.items()
  }
  manifest = {**identity, "outputs": outputs, "validation": validation}
  write_json_atomic(manifest, manifest_path)
  progress_path.unlink(missing_ok=True)
  print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()
