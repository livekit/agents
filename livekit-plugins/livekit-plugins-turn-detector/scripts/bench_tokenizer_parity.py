#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers>=4.47.1,!=4.57.2,!=4.57.3",
#     "tokenizers>=0.20,<1",
#     "huggingface-hub>=0.25",
#     "jinja2",
#     "onnxruntime>=1.18",
#     "numpy>=1.26",
# ]
# ///
"""Parity + micro-benchmark for swapping transformers -> tokenizers in turn-detector.

For both EOU model revisions, the script:
  1. Loads the OLD pipeline (transformers.AutoTokenizer + apply_chat_template).
  2. Loads the NEW pipeline (tokenizers.Tokenizer + jinja2 chat_template).
  3. Walks .py files in the repo, builds chat contexts from their lines,
     and asserts that:
       a. The rendered chat-template string is byte-identical.
       b. The token-id sequence (after left-truncation to MAX_HISTORY_TOKENS)
          is identical.
       c. The ONNX model's eou_probability is bit-identical.
  4. Reports cold-load and per-call timings for both pipelines.

Run from the repo root (or anywhere; defaults are absolute):
    uv run livekit-plugins/livekit-plugins-turn-detector/scripts/bench_tokenizer_parity.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

HG_MODEL = "livekit/turn-detector"
ONNX_FILENAME = "model_q8.onnx"
MODEL_REVISIONS: dict[str, str] = {
    "en": "v1.2.2-en",
    "multilingual": "v0.4.1-intl",
}

MAX_HISTORY_TOKENS = 128
MAX_HISTORY_TURNS = 6


# ---------- shared text normalization (mirrors _EUORunnerBase._normalize_text) -------- #


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text.lower())
    text = "".join(
        ch for ch in text if not (unicodedata.category(ch).startswith("P") and ch not in ["'", "-"])
    )
    return re.sub(r"\s+", " ", text).strip()


def build_chat_ctx(lines: list[str]) -> list[dict[str, Any]]:
    """Build a chat context with up to MAX_HISTORY_TURNS turns alternating roles.

    Mirrors what `_format_chat_ctx` consumes (already-normalized, role-tagged).
    """
    msgs: list[dict[str, Any]] = []
    role_cycle = ("user", "assistant")
    for i, raw in enumerate(lines[-MAX_HISTORY_TURNS:]):
        content = normalize_text(raw)
        if not content:
            continue
        msgs.append({"role": role_cycle[i % 2], "content": content})
    return msgs


# --------------- OLD pipeline (transformers) ----------------------------------------- #


def load_old(revision: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(HG_MODEL, revision=revision, truncation_side="left")


def render_old(tokenizer: Any, chat_ctx: list[dict[str, Any]]) -> str:
    convo = tokenizer.apply_chat_template(
        chat_ctx, add_generation_prompt=False, add_special_tokens=False, tokenize=False
    )
    ix = convo.rfind("<|im_end|>")
    return convo[:ix]


def encode_old(tokenizer: Any, text: str) -> list[int]:
    out = tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="np",
        max_length=MAX_HISTORY_TOKENS,
        truncation=True,
    )
    return out["input_ids"].astype("int64").flatten().tolist()


# --------------- NEW pipeline (tokenizers + jinja2) ---------------------------------- #


@dataclass
class NewPipeline:
    tokenizer: Any
    template: Any


def load_new(revision: str) -> NewPipeline:
    from huggingface_hub import hf_hub_download
    from jinja2 import Environment
    from tokenizers import Tokenizer

    tok_path = hf_hub_download(HG_MODEL, "tokenizer.json", revision=revision)
    cfg_path = hf_hub_download(HG_MODEL, "tokenizer_config.json", revision=revision)

    tok = Tokenizer.from_file(tok_path)
    tok.enable_truncation(max_length=MAX_HISTORY_TOKENS, direction="left")

    with open(cfg_path) as f:
        cfg = json.load(f)
    template = Environment(autoescape=False).from_string(cfg["chat_template"])
    return NewPipeline(tokenizer=tok, template=template)


def render_new(pipe: NewPipeline, chat_ctx: list[dict[str, Any]]) -> str:
    convo = pipe.template.render(messages=chat_ctx, add_generation_prompt=False)
    ix = convo.rfind("<|im_end|>")
    return convo[:ix]


def encode_new(pipe: NewPipeline, text: str) -> list[int]:
    return pipe.tokenizer.encode(text, add_special_tokens=False).ids


# --------------- ONNX session (shared between pipelines) ----------------------------- #


def load_session(revision: str) -> Any:
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    onnx_path = hf_hub_download(HG_MODEL, ONNX_FILENAME, subfolder="onnx", revision=revision)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"], sess_options=opts)


def predict(session: Any, ids: list[int]) -> float:
    import numpy as np

    input_ids = np.asarray(ids, dtype=np.int64).reshape(1, -1)
    out = session.run(None, {"input_ids": input_ids})
    return float(out[0].flatten()[-1])


# --------------- driver --------------------------------------------------------------- #


@dataclass
class Stats:
    render: list[float] = field(default_factory=list)
    encode: list[float] = field(default_factory=list)

    def summary(self) -> str:
        def fmt(xs: list[float]) -> str:
            if not xs:
                return "n/a"
            return (
                f"mean={statistics.mean(xs) * 1e6:7.1f}us "
                f"p50={statistics.median(xs) * 1e6:7.1f}us "
                f"p95={statistics.quantiles(xs, n=20)[-1] * 1e6:7.1f}us"
                if len(xs) >= 20
                else f"mean={statistics.mean(xs) * 1e6:7.1f}us p50={statistics.median(xs) * 1e6:7.1f}us"
            )

        return f"render: {fmt(self.render)}  encode: {fmt(self.encode)}"


def gather_files(roots: list[Path], limit: int) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*.py")):
            if any(part.startswith(".") for part in p.parts):
                continue
            files.append(p)
            if len(files) >= limit:
                return files
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="max number of source files to use as samples (default: 200)",
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        type=Path,
        default=[
            REPO_ROOT / "livekit-agents",
            REPO_ROOT / "livekit-plugins",
            REPO_ROOT / "examples",
        ],
        help="directories to scan for .py samples",
    )
    parser.add_argument(
        "--revisions",
        nargs="+",
        default=list(MODEL_REVISIONS.keys()),
        choices=list(MODEL_REVISIONS.keys()),
    )
    parser.add_argument(
        "--skip-onnx",
        action="store_true",
        help="skip the ONNX equivalence check (still does string + token id parity)",
    )
    args = parser.parse_args()

    files = gather_files(args.roots, args.limit)
    if not files:
        print(f"no .py files found under {args.roots}", file=sys.stderr)
        return 2
    print(f"sampling {len(files)} files from {len(args.roots)} root(s)")

    overall_ok = True

    for model_type in args.revisions:
        revision = MODEL_REVISIONS[model_type]
        print(f"\n===== model_type={model_type} revision={revision} =====")

        t0 = time.perf_counter()
        old_tok = load_old(revision)
        old_load = time.perf_counter() - t0

        t0 = time.perf_counter()
        new_pipe = load_new(revision)
        new_load = time.perf_counter() - t0
        print(f"cold-load: old={old_load * 1000:6.1f}ms  new={new_load * 1000:6.1f}ms")

        session = None if args.skip_onnx else load_session(revision)

        old_stats = Stats()
        new_stats = Stats()

        mismatches: list[tuple[Path, str]] = []
        n_samples = 0
        for path in files:
            try:
                lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue
            chat_ctx = build_chat_ctx([ln for ln in lines if ln.strip()])
            if not chat_ctx:
                continue
            n_samples += 1

            t0 = time.perf_counter()
            old_text = render_old(old_tok, chat_ctx)
            old_stats.render.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            old_ids = encode_old(old_tok, old_text)
            old_stats.encode.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            new_text = render_new(new_pipe, chat_ctx)
            new_stats.render.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            new_ids = encode_new(new_pipe, new_text)
            new_stats.encode.append(time.perf_counter() - t0)

            if old_text != new_text:
                mismatches.append((path, "rendered chat template differs"))
                # show first diff
                for i, (a, b) in enumerate(zip(old_text, new_text, strict=False)):
                    if a != b:
                        ctx_a = old_text[max(0, i - 20) : i + 20]
                        ctx_b = new_text[max(0, i - 20) : i + 20]
                        print(f"  [{path.name}] text diff at {i}: old={ctx_a!r} new={ctx_b!r}")
                        break
                continue

            if old_ids != new_ids:
                mismatches.append((path, f"ids differ (len old={len(old_ids)} new={len(new_ids)})"))
                continue

            if session is not None:
                old_p = predict(session, old_ids)
                new_p = predict(session, new_ids)
                if not math.isclose(old_p, new_p, rel_tol=0.0, abs_tol=0.0):
                    mismatches.append((path, f"eou_probability differs: {old_p} vs {new_p}"))

        print(f"samples: {n_samples}")
        print(f"old timings: {old_stats.summary()}")
        print(f"new timings: {new_stats.summary()}")
        if mismatches:
            overall_ok = False
            print(f"MISMATCHES ({len(mismatches)}):")
            for path, msg in mismatches[:10]:
                print(f"  - {path.relative_to(REPO_ROOT)}: {msg}")
            if len(mismatches) > 10:
                print(f"  ... and {len(mismatches) - 10} more")
        else:
            print("PARITY OK (all samples match: text, ids, onnx)")

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
