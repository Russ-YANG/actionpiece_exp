#!/usr/bin/env python3
"""Generate resumable Gemini Embedding 2 item vectors for Amazon Beauty."""

import argparse
import base64
import getpass
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from PIL import Image
import requests


MODEL_ID = "gemini-embedding-2"
NATIVE_DIMENSION = 3072
DERIVED_DIMENSIONS = (1536, 768)
INSTRUCTION = (
    "Represent this e-commerce product for semantic similarity and sequential "
    "recommendation. Capture its category, function, attributes, style, and "
    "likely user intent."
)
API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{MODEL_ID}:embedContent"
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--cache-dir", default="cache/AmazonReviews2014/Beauty"
  )
  parser.add_argument("--limit", type=int)
  parser.add_argument("--output-prefix")
  parser.add_argument("--timeout", type=float, default=90.0)
  parser.add_argument("--max-retries", type=int, default=8)
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


def load_image_part(path: Path) -> dict[str, Any]:
  with Image.open(path) as image:
    image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
  return {
      "inline_data": {
          "mime_type": "image/jpeg",
          "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
      }
  }


def request_embedding(
    session: requests.Session,
    api_key: str,
    text: str,
    image_path: Path | None,
    timeout: float,
    max_retries: int,
) -> np.ndarray:
  parts: list[dict[str, Any]] = [
      {"text": f"{INSTRUCTION}\n\nProduct metadata:\n{text}"}
  ]
  if image_path is not None:
    parts.append(load_image_part(image_path))
  payload = {
      "content": {"parts": parts},
      "output_dimensionality": NATIVE_DIMENSION,
  }
  retryable_statuses = {408, 429, 500, 502, 503, 504}
  for attempt in range(max_retries + 1):
    try:
      response = session.post(
          API_URL,
          headers={"x-goog-api-key": api_key},
          json=payload,
          timeout=timeout,
      )
    except requests.RequestException:
      if attempt == max_retries:
        raise
      time.sleep(min(2 ** attempt, 60))
      continue
    if response.status_code == 200:
      body = response.json()
      embedding = body.get("embedding")
      if not isinstance(embedding, dict):
        embeddings = body.get("embeddings")
        if isinstance(embeddings, list) and len(embeddings) == 1:
          embedding = embeddings[0]
      if not isinstance(embedding, dict):
        raise RuntimeError(
            "Unexpected Gemini response keys: " + ", ".join(sorted(body))
        )
      vector = np.asarray(embedding.get("values"), dtype=np.float32)
      if vector.shape != (NATIVE_DIMENSION,):
        raise RuntimeError(
            f"Gemini returned shape {vector.shape}, expected "
            f"({NATIVE_DIMENSION},)."
        )
      if not np.isfinite(vector).all():
        raise RuntimeError("Gemini returned NaN or Inf values.")
      norm = np.linalg.norm(vector)
      if not np.isfinite(norm) or norm == 0:
        raise RuntimeError(f"Gemini returned invalid norm {norm}.")
      return vector / norm
    if response.status_code not in retryable_statuses or attempt == max_retries:
      detail = response.text[:1000]
      raise RuntimeError(
          f"Gemini request failed with HTTP {response.status_code}: {detail}"
      )
    retry_after = response.headers.get("Retry-After")
    delay = float(retry_after) if retry_after else min(2 ** attempt, 60)
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
  derived = {}
  for dimension in DERIVED_DIMENSIONS:
    output_path = Path(f"{output_prefix}.d{dimension}.sent_emb")
    partial_path = Path(f"{output_path}.partial")
    vectors = np.memmap(
        partial_path,
        dtype=np.float32,
        mode="w+",
        shape=(item_count, dimension),
    )
    block_size = 512
    for start in range(0, item_count, block_size):
      end = min(start + block_size, item_count)
      block = np.asarray(source[start:end, :dimension], dtype=np.float32)
      norms = np.linalg.norm(block, axis=1, keepdims=True)
      if np.any(norms == 0) or not np.isfinite(norms).all():
        raise RuntimeError(f"Invalid prefix norm for dimension {dimension}.")
      vectors[start:end] = block / norms
    vectors.flush()
    del vectors
    partial_path.replace(output_path)
    derived[str(dimension)] = {
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "bytes": output_path.stat().st_size,
    }
  del source
  return derived


def main() -> None:
  args = parse_args()
  cache_dir = Path(args.cache_dir)
  processed_dir = cache_dir / "processed"
  id_mapping_path = processed_dir / "id_mapping.json"
  metadata_path = processed_dir / "metadata.qwen_multimodal.json"
  image_manifest_path = processed_dir / "image_download_manifest.jsonl"

  id_mapping = json.loads(id_mapping_path.read_text(encoding="utf-8"))
  metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
  item_ids = id_mapping["id2item"][1:]
  if args.limit is not None:
    if args.limit <= 0:
      raise ValueError("--limit must be positive.")
    item_ids = item_ids[: args.limit]
  if set(item_ids) - set(metadata):
    raise RuntimeError("Metadata is missing mapped item keys.")

  image_status = {}
  with image_manifest_path.open("r", encoding="utf-8") as stream:
    for line in stream:
      record = json.loads(line)
      image_status[record["asin"]] = record
  missing_manifest_items = set(item_ids) - set(image_status)
  if missing_manifest_items:
    raise RuntimeError(
        f"Image manifest is missing {len(missing_manifest_items)} item keys."
    )

  instruction_hash = hashlib.sha256(INSTRUCTION.encode()).hexdigest()[:12]
  metadata_hash = sha256_file(metadata_path)[:12]
  image_manifest_hash = sha256_file(image_manifest_path)[:12]
  default_prefix = processed_dir / (
      f"gemini_api.{MODEL_ID}.i{instruction_hash}.m{metadata_hash}."
      f"joint.im{image_manifest_hash}"
  )
  output_prefix = Path(args.output_prefix) if args.output_prefix else default_prefix
  output_path = Path(f"{output_prefix}.d{NATIVE_DIMENSION}.sent_emb")
  partial_path = Path(f"{output_path}.partial")
  progress_path = Path(f"{output_path}.progress.json")
  manifest_path = Path(f"{output_path}.manifest.json")
  output_path.parent.mkdir(parents=True, exist_ok=True)

  missing_images = []
  image_paths: list[Path | None] = []
  for item_id in item_ids:
    record = image_status[item_id]
    candidate = Path(record["path"]) if record.get("path") else None
    if (
        record.get("status") == "downloaded"
        and candidate is not None
        and candidate.is_file()
    ):
      image_paths.append(candidate)
    else:
      image_paths.append(None)
      missing_images.append(item_id)

  identity = {
      "model": MODEL_ID,
      "dimension": NATIVE_DIMENSION,
      "item_count": len(item_ids),
      "id_mapping_sha256": sha256_file(id_mapping_path),
      "metadata_sha256": sha256_file(metadata_path),
      "image_manifest_sha256": sha256_file(image_manifest_path),
      "instruction": INSTRUCTION,
      "instruction_sha256": hashlib.sha256(INSTRUCTION.encode()).hexdigest(),
  }
  print(
      f"validated {len(item_ids)} items; "
      f"images={len(item_ids) - len(missing_images)}, "
      f"text_only={len(missing_images)}"
  )
  print(f"output: {output_path}")
  if args.validate_only:
    return

  api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
  if not api_key:
    api_key = getpass.getpass("Gemini API key: ").strip()
  if not api_key:
    raise RuntimeError("A Gemini API key is required.")
  if output_path.exists():
    raise FileExistsError(
        f"Refusing to overwrite completed embedding artifact: {output_path}"
    )

  next_index = 0
  if progress_path.exists():
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    mismatched = [key for key, value in identity.items() if progress.get(key) != value]
    if mismatched:
      raise RuntimeError("Partial Gemini cache identity mismatch: " + ", ".join(mismatched))
    next_index = int(progress.get("next_index", 0))

  expected_bytes = len(item_ids) * NATIVE_DIMENSION * np.dtype(np.float32).itemsize
  if next_index == 0:
    partial_path.unlink(missing_ok=True)
    vectors = np.memmap(
        partial_path,
        dtype=np.float32,
        mode="w+",
        shape=(len(item_ids), NATIVE_DIMENSION),
    )
    vectors[:] = 0
    vectors.flush()
  else:
    if not partial_path.exists() or partial_path.stat().st_size != expected_bytes:
      raise RuntimeError("Partial Gemini embedding file is missing or invalid.")
    vectors = np.memmap(
        partial_path,
        dtype=np.float32,
        mode="r+",
        shape=(len(item_ids), NATIVE_DIMENSION),
    )

  session = requests.Session()
  for index in range(next_index, len(item_ids)):
    item_id = item_ids[index]
    item_metadata = metadata[item_id]
    text = item_metadata["sentence"] if isinstance(item_metadata, dict) else item_metadata
    vectors[index] = request_embedding(
        session=session,
        api_key=api_key,
        text=text,
        image_path=image_paths[index],
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    vectors.flush()
    write_json_atomic({**identity, "next_index": index + 1}, progress_path)
    print(f"Gemini embeddings: {index + 1}/{len(item_ids)}", flush=True)

  vectors.flush()
  del vectors
  partial_path.replace(output_path)
  progress_path.unlink(missing_ok=True)
  derived = derive_prefix_embeddings(output_path, len(item_ids), output_prefix)
  write_json_atomic(
      {
          **identity,
          "embedding_file": str(output_path),
          "embedding_sha256": sha256_file(output_path),
          "embedding_bytes": output_path.stat().st_size,
          "derived_embeddings": derived,
          "missing_image_count": len(missing_images),
          "missing_image_items": missing_images,
          "first_item": item_ids[0],
          "last_item": item_ids[-1],
      },
      manifest_path,
  )
  print(f"sha256: {sha256_file(output_path)}")
  for dimension, record in derived.items():
    print(f"d{dimension}: {record['path']} sha256={record['sha256']}")


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    print("Interrupted; completed items remain resumable.", file=sys.stderr)
    raise
