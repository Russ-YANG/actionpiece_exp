#!/usr/bin/env python3
"""Generate Beauty text embeddings with the Gemini Embedding Batch API."""

import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from google import genai
from google.genai import types
import numpy as np


MODEL_ID = "gemini-embedding-2"
NATIVE_DIMENSION = 3072
DERIVED_DIMENSIONS = (1536, 768)
INSTRUCTION = (
    "Represent this e-commerce product for semantic similarity and sequential "
    "recommendation. Capture its category, function, attributes, style, and "
    "likely user intent."
)
TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_PARTIALLY_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--processed-dir",
      default="cache/AmazonReviews2014/Beauty/processed",
  )
  parser.add_argument("--output-prefix")
  parser.add_argument("--limit", type=int)
  parser.add_argument("--poll-seconds", type=float, default=30.0)
  parser.add_argument("--submit-only", action="store_true")
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
    json.dump(value, stream, ensure_ascii=False, indent=2, default=str)
  temporary.replace(path)


def state_name(job: types.BatchJob) -> str:
  state = job.state
  return state.value if hasattr(state, "value") else str(state)


def load_api_key() -> str:
  key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
  if key:
    return key
  return getpass.getpass("Gemini API key (not stored): ")


def prepare_batch_input(
    path: Path,
    item_ids: list[str],
    metadata: dict[str, str],
) -> None:
  temporary = path.with_suffix(path.suffix + ".tmp")
  with temporary.open("w", encoding="utf-8") as stream:
    for item_id in item_ids:
      record = {
          "key": item_id,
          "request": {
              "output_dimensionality": NATIVE_DIMENSION,
              "content": {
                  "parts": [{
                      "text": (
                          f"{INSTRUCTION}\n\nProduct metadata:\n"
                          f"{metadata[item_id]}"
                      )
                  }]
              },
          },
      }
      stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
      stream.write("\n")
  temporary.replace(path)


def embedding_values(record: dict[str, Any]) -> list[float] | None:
  response = record.get("response")
  if not isinstance(response, dict):
    return None
  embedding = response.get("embedding")
  if not isinstance(embedding, dict):
    embeddings = response.get("embeddings")
    if isinstance(embeddings, list) and len(embeddings) == 1:
      embedding = embeddings[0]
  if not isinstance(embedding, dict):
    return None
  values = embedding.get("values")
  return values if isinstance(values, list) else None


def assemble_embeddings(
    result_path: Path,
    item_ids: list[str],
    output_path: Path,
) -> dict[str, Any]:
  rows: dict[str, list[float]] = {}
  failures: dict[str, Any] = {}
  token_count = 0
  with result_path.open("r", encoding="utf-8") as stream:
    for line_number, line in enumerate(stream, 1):
      if not line.strip():
        continue
      record = json.loads(line)
      key = record.get("key")
      if not isinstance(key, str):
        raise RuntimeError(f"Result line {line_number} has no string key.")
      if key in rows or key in failures:
        raise RuntimeError(f"Duplicate result key: {key}")
      values = embedding_values(record)
      if values is None:
        failures[key] = record.get("error", record)
        continue
      rows[key] = values
      response = record.get("response", {})
      raw_tokens = response.get("tokenCount", response.get("token_count", 0))
      try:
        token_count += int(raw_tokens)
      except (TypeError, ValueError):
        pass

  expected = set(item_ids)
  unexpected = (set(rows) | set(failures)) - expected
  missing = expected - set(rows) - set(failures)
  if unexpected or missing or failures:
    raise RuntimeError(
        "Incomplete batch result: "
        f"success={len(rows)}, failures={len(failures)}, "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )

  partial_path = output_path.with_suffix(output_path.suffix + ".partial")
  output = np.memmap(
      partial_path,
      dtype=np.float32,
      mode="w+",
      shape=(len(item_ids), NATIVE_DIMENSION),
  )
  norm_min = float("inf")
  norm_max = 0.0
  for index, item_id in enumerate(item_ids):
    vector = np.asarray(rows[item_id], dtype=np.float32)
    if vector.shape != (NATIVE_DIMENSION,):
      raise RuntimeError(f"{item_id} returned shape {vector.shape}.")
    if not np.isfinite(vector).all():
      raise RuntimeError(f"{item_id} returned NaN or Inf.")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm == 0:
      raise RuntimeError(f"{item_id} returned invalid norm {norm}.")
    output[index] = vector / norm
    norm_min = min(norm_min, norm)
    norm_max = max(norm_max, norm)
  output.flush()
  del output
  partial_path.replace(output_path)
  return {
      "token_count": token_count,
      "api_vector_norm_before_local_normalization": {
          "min": norm_min,
          "max": norm_max,
      },
  }


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


def main() -> None:
  args = parse_args()
  if args.limit is not None and args.limit <= 0:
    raise ValueError("--limit must be positive.")
  if args.poll_seconds <= 0:
    raise ValueError("--poll-seconds must be positive.")

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
  batch_input_path = Path(f"{output_prefix}.batch_input.jsonl")
  state_path = Path(f"{output_prefix}.batch_job.json")
  result_path = Path(f"{output_prefix}.batch_result.jsonl")
  output_path = Path(f"{output_prefix}.d{NATIVE_DIMENSION}.sent_emb")
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

  if not batch_input_path.exists():
    prepare_batch_input(batch_input_path, item_ids, metadata)
    print(f"prepared batch input: {batch_input_path}")

  api_key = load_api_key()
  client = genai.Client(api_key=api_key)
  if state_path.exists():
    saved_state = json.loads(state_path.read_text(encoding="utf-8"))
    for field in ("item_count", "id_mapping_sha256", "metadata_sha256"):
      if saved_state["identity"][field] != identity[field]:
        raise RuntimeError(f"Saved batch identity mismatch: {field}")
    job_name = saved_state["job_name"]
    print(f"resuming batch job: {job_name}")
  else:
    uploaded = client.files.upload(
        file=batch_input_path,
        config=types.UploadFileConfig(
            mime_type="jsonl",
            display_name=f"Beauty text embeddings ({len(item_ids)} items)",
        ),
    )
    job = client.batches.create_embeddings(
        model=MODEL_ID,
        src=types.EmbeddingsBatchJobSource(file_name=uploaded.name),
        config=types.CreateEmbeddingsBatchJobConfig(
            display_name=f"Beauty text {MODEL_ID} d{NATIVE_DIMENSION}"
        ),
    )
    job_name = job.name
    write_json_atomic(
        {
            "identity": identity,
            "batch_input_path": str(batch_input_path),
            "batch_input_sha256": sha256_file(batch_input_path),
            "uploaded_file_name": uploaded.name,
            "job_name": job_name,
            "submitted_at": str(job.create_time),
            "last_state": state_name(job),
        },
        state_path,
    )
    print(f"submitted batch job: {job_name}")

  if args.submit_only:
    return

  while True:
    job = client.batches.get(name=job_name)
    current_state = state_name(job)
    saved_state = json.loads(state_path.read_text(encoding="utf-8"))
    saved_state["last_state"] = current_state
    saved_state["last_checked_at"] = str(job.update_time)
    write_json_atomic(saved_state, state_path)
    print(f"batch state: {current_state}", flush=True)
    if current_state in TERMINAL_STATES:
      break
    time.sleep(args.poll_seconds)

  if current_state not in {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"}:
    raise RuntimeError(f"Batch ended in {current_state}: {job.error}")
  if not job.dest or not job.dest.file_name:
    raise RuntimeError(f"Successful batch has no result file: {job}")
  result_bytes = client.files.download(file=job.dest.file_name)
  temporary_result = result_path.with_suffix(result_path.suffix + ".tmp")
  temporary_result.write_bytes(result_bytes)
  temporary_result.replace(result_path)
  print(f"downloaded batch result: {result_path}")

  assembly = assemble_embeddings(result_path, item_ids, output_path)
  outputs = {
      str(NATIVE_DIMENSION): {
          "path": str(output_path),
          "bytes": output_path.stat().st_size,
          "sha256": sha256_file(output_path),
      }
  }
  outputs.update(derive_prefix_embeddings(output_path, len(item_ids), output_prefix))
  manifest = {
      **identity,
      "batch_job_name": job_name,
      "batch_result_path": str(result_path),
      "batch_result_sha256": sha256_file(result_path),
      "batch_state": current_state,
      **assembly,
      "outputs": outputs,
  }
  write_json_atomic(manifest, manifest_path)
  print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()
