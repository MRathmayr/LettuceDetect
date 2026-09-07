"""Prompted-judge baseline on the RAGTruth test split.

The paper claims that reading an observer's internal states beats asking a model
of the same size for a verdict. That needs a verdict baseline on the same
observer, so this script asks an observer directly whether a response is
supported, and scores it by P("Yes") on the hallucination-directed question.

`--model` selects which observer, from the Grounding probe's own model_config.py,
and names the output after it. It defaults to 3b, the row the paper committed
first and the only one an 8 GB card can hold in fp16.

Two things make it a same-observer comparison rather than a new experiment:

1. The prompt prefix is built by the Grounding probe's own `build_prompt` and
   tokenized with the probe's own prompt/response split and 4096-token
   right-truncation, imported from `hallu-training/py/model_utils.py`. Only a
   verdict question is appended.
2. fp16, the precision the production hidden states were extracted in. int4 or
   bf16 would be a different model and would void the comparison. bf16 is
   emulated on Pascal in any case.

`num_logits_to_keep=1` is required, not an optimization: without it transformers
materializes fp32 logits at every position and the forward pass runs out of
memory on an 8 GB card.

Usage:
    python tests/benchmarks/judge_baseline.py --limit 25          # gate run
    python tests/benchmarks/judge_baseline.py                     # 3b, 2700 rows
    python tests/benchmarks/judge_baseline.py --model 14b         # a larger observer
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_LETTUCE_ROOT = Path(__file__).resolve().parents[2]
_DIPLOMA_ROOT = _LETTUCE_ROOT.parent
sys.path.insert(0, str(_LETTUCE_ROOT))
# Appended, not inserted: `model_utils`/`model_config` are generic names and
# must not shadow anything importable from the benchmark tree.
sys.path.append(str(_DIPLOMA_ROOT / "hallu-training" / "py"))

from model_config import MODEL_CONFIGS, get_model_family  # noqa: E402
from model_utils import build_prompt  # noqa: E402  (path set above)

from tests.benchmarks.core import (  # noqa: E402
    BenchmarkTimer,
    PredictionResult,
    compute_accuracy_metrics,
)
from tests.benchmarks.core.memory import MemoryTracker  # noqa: E402
from tests.benchmarks.core.verdict_export import (  # noqa: E402
    NATIVE_DIR,
    compact_metrics,
    compact_predictions,
    write_compact,
    write_native,
)

DEFAULT_OBSERVER = "3b"


def observer_ids(size: str) -> tuple[str, str, str]:
    """HuggingFace id, run name, and method tag for one observer size.

    Read from the Grounding probe's own `model_config.py` rather than restated
    here, so the judge cannot drift onto a different checkpoint than the probe
    it is being compared against. That identity is the experiment: the two
    readouts must differ only in whether the answer comes from a decoded token
    or from a hidden state.
    """
    family = get_model_family(size)
    return MODEL_CONFIGS[size], f"judge_{family}{size}_ragtruth_test", f"judge_{size}"


# Phrased so that "Yes" means hallucinated, matching the probe's P(hallucinated).
VERDICT_SUFFIX = (
    "\n\nIs any part of the response unsupported by the context? Answer Yes or No.\nAnswer:"
)
MAX_PREFIX_TOKENS = 4096  # the probe's budget for prompt + response


def _yes_no_ids(tokenizer) -> tuple[int, int]:
    """Token ids continuing 'Answer:', so the space-prefixed variants."""
    yes = tokenizer.encode(" Yes", add_special_tokens=False)
    no = tokenizer.encode(" No", add_special_tokens=False)
    if len(yes) != 1 or len(no) != 1:
        raise ValueError(f"Yes/No are not single tokens: {yes}, {no}")
    return yes[0], no[0]


def build_input_ids(tokenizer, sample, suffix_ids: torch.Tensor) -> torch.Tensor:
    """Tokenize one sample the way the probe does, then append the verdict question.

    Mirrors `extract_hidden_state`: prompt and response are tokenized separately
    with a leading space on the response, and the response is truncated from the
    right so the prompt boundary survives.
    """
    prompt = build_prompt(sample.question, sample.context)
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"]
    response_ids = tokenizer(
        " " + sample.response, return_tensors="pt", add_special_tokens=False
    )["input_ids"]

    prompt_len = prompt_ids.shape[1]
    max_response_len = MAX_PREFIX_TOKENS - prompt_len
    if max_response_len <= 0:
        prompt_ids = prompt_ids[:, :3800]
        prompt_len = prompt_ids.shape[1]
        max_response_len = MAX_PREFIX_TOKENS - prompt_len
    response_ids = response_ids[:, :max_response_len]

    return torch.cat([prompt_ids, response_ids, suffix_ids], dim=1)


def score(model, input_ids: torch.Tensor, yes_id: int, no_id: int) -> float:
    """P('Yes') from one forward pass, softmax over the Yes/No logits only."""
    input_ids = input_ids.to(model.device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            num_logits_to_keep=1,
        )
    logits = out.logits[0, -1].float()
    probs = torch.softmax(torch.stack([logits[yes_id], logits[no_id]]), dim=0)
    return probs[0].item()


def _checkpoint_signature(model_id: str) -> dict:
    """Everything that would change the scores. Resume is refused if it differs."""
    return {
        "model": model_id,
        "verdict_suffix": VERDICT_SUFFIX,
        "max_prefix_tokens": MAX_PREFIX_TOKENS,
    }


def load_checkpoint(
    path: Path, samples: list, model_id: str, method: str
) -> tuple[list[PredictionResult], list[int], float]:
    """Resume from a partial dump, or start from scratch if there is none.

    Refuses rather than resumes when the checkpoint was produced by a different
    template, model, or sample order: a silently mixed run would be unusable and
    the failure would not be visible in the output file.
    """
    if not path.exists():
        return [], [], 0.0

    ckpt = json.loads(path.read_text())
    signature = _checkpoint_signature(model_id)
    if ckpt.get("signature") != signature:
        raise RuntimeError(
            f"{path.name} was written under a different configuration "
            f"({ckpt.get('signature')} vs {signature}). Delete it or pass --no-resume."
        )

    done = ckpt["predictions"]
    expected = [s.id for s in samples[: len(done)]]
    if [d["sample_id"] for d in done] != expected:
        raise RuntimeError(
            f"{path.name} is not a prefix of the current sample order. "
            "Delete it or pass --no-resume."
        )

    predictions = [
        PredictionResult(
            d["sample_id"],
            d["ground_truth"],
            d["predicted_score"],
            int(d["predicted_score"] >= 0.5),
            d["latency_ms"],
            method,
        )
        for d in done
    ]
    token_lengths = [d["input_tokens"] for d in done]
    print(f"Resuming from {path.name}: {len(predictions)} of {len(samples)} rows already scored")
    return predictions, token_lengths, ckpt.get("gpu_peak_mb", 0.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Sample limit (gate runs)")
    parser.add_argument(
        "--model",
        default=DEFAULT_OBSERVER,
        choices=sorted(MODEL_CONFIGS),
        help="Observer to ask for a verdict (default: 3b, the committed baseline)",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=100, help="Rows between partial dumps"
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore an existing checkpoint and score every row again",
    )
    args = parser.parse_args()
    model_id, base_run_name, method = observer_ids(args.model)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tests.benchmarks.data_adapters import RAGTruthAdapter

    samples = [s for s in RAGTruthAdapter().load(limit=args.limit) if s.context and s.response]
    print(f"Loaded {len(samples)} RAGTruth test samples")

    # A limited run is a gate, not the baseline: it writes the native file under
    # its own name and never the paper-side file, so a crashed full run cannot
    # leave a 25-row file standing in for 2700. Its checkpoint is separate too,
    # or a full run would resume from the gate's rows.
    run_name = base_run_name if args.limit is None else f"{base_run_name}_limit{args.limit}"
    partial_path = NATIVE_DIR / f"{run_name}.partial.json"
    NATIVE_DIR.mkdir(parents=True, exist_ok=True)

    predictions, token_lengths, prior_peak_mb = (
        load_checkpoint(partial_path, samples, model_id, method)
        if args.resume
        else ([], [], 0.0)
    )
    resumed_from = len(predictions)
    if resumed_from == len(samples):
        print("Checkpoint already covers every row; nothing left to score.")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    model.eval()
    print(f"Loaded {model_id}: dtype={model.dtype}, device={model.device}")
    if model.dtype != torch.float16:
        raise RuntimeError(f"Expected fp16, got {model.dtype}")

    yes_id, no_id = _yes_no_ids(tokenizer)
    suffix_ids = tokenizer(VERDICT_SUFFIX, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ]
    print(f"Yes/No token ids: {yes_id}/{no_id}, verdict suffix: {suffix_ids.shape[1]} tokens")

    timer = BenchmarkTimer(sync_cuda=True)
    memory = MemoryTracker()
    # Replay resumed latencies so the reported mean and p95 cover all rows, not
    # only the ones this session scored.
    for p in predictions:
        timer.record(p.latency_ms)
    started = time.time()

    with memory.track():
        for i, s in enumerate(samples[resumed_from:], start=resumed_from + 1):
            # Tokenization sits inside the timed region because the Stage 3
            # benchmark times the probe the same way.
            with timer.measure():
                input_ids = build_input_ids(tokenizer, s, suffix_ids)
                p_yes = score(model, input_ids, yes_id, no_id)
            token_lengths.append(input_ids.shape[1])
            predictions.append(
                PredictionResult(
                    s.id, s.ground_truth, p_yes, int(p_yes >= 0.5), timer.last_ms, method
                )
            )
            if i % args.checkpoint_every == 0 or i == len(samples):
                peak = max(torch.cuda.max_memory_allocated() / (1024 * 1024), prior_peak_mb)
                partial_path.write_text(
                    json.dumps(
                        {
                            "n_done": i,
                            "gpu_peak_mb": peak,
                            "signature": _checkpoint_signature(model_id),
                            "predictions": [
                                {
                                    "sample_id": p.sample_id,
                                    "ground_truth": p.ground_truth,
                                    "predicted_score": p.predicted_score,
                                    "latency_ms": round(p.latency_ms, 3),
                                    "input_tokens": n_tok,
                                }
                                for p, n_tok in zip(predictions, token_lengths)
                            ],
                        }
                    )
                )
                rate = (time.time() - started) / (i - resumed_from)
                print(
                    f"  {i}/{len(samples)}  {rate:.2f}s/sample  peak {peak:.0f}MB  "
                    f"eta {(len(samples) - i) * rate / 60:.0f}min",
                    flush=True,
                )

    elapsed = time.time() - started
    timing = timer.get_stats()
    mem_stats = memory.get_stats()
    metrics = compute_accuracy_metrics(predictions, compute_ci=False)

    lengths = sorted(token_lengths)
    meta = {
        "model": model_id,
        "precision": "fp16",
        "n_samples": metrics.n_samples,
        "dataset": "ragtruth_test",
        "latency_mean_ms": timing.mean_ms,
        "latency_p95_ms": timing.p95_ms,
        "gpu_peak_mb": max(mem_stats.gpu_peak_mb or 0.0, prior_peak_mb),
        # Scoring time for this session only. A resumed run splits the total
        # across sessions, so `resumed_from_n` says how many rows this one skipped.
        "total_time_sec": elapsed,
        "resumed_from_n": resumed_from,
        "prompt_template_note": (
            "Grounding probe template (hallu-training/py/model_utils.py build_prompt: "
            "'Context:\\n{context}\\nQuestion: {question}\\nResponse: {response}', prompt and "
            "response tokenized separately, response right-truncated to 4096 total) plus the "
            f"verdict question {VERDICT_SUFFIX!r}; score = P(' Yes') via softmax over the "
            "' Yes'/' No' logits of one forward pass, so higher = hallucinated"
        ),
        "input_tokens_mean": sum(lengths) / len(lengths),
        "input_tokens_p90": lengths[int(0.9 * (len(lengths) - 1))],
        "input_tokens_max": lengths[-1],
    }

    task_map = {s.id: s.task_type for s in samples if s.task_type}
    native = write_native(
        run_name,
        {
            "metadata": meta,
            "metrics": metrics.to_dict(),
            "timing": timing.to_dict(),
            "predictions": [
                {
                    "sample_id": p.sample_id,
                    "ground_truth": p.ground_truth,
                    "predicted_score": p.predicted_score,
                    "predicted_label": p.predicted_label,
                    "latency_ms": round(p.latency_ms, 3),
                    "task_type": task_map.get(p.sample_id),
                }
                for p in predictions
            ],
        },
    )
    paper = (
        write_compact(
            run_name, meta, compact_metrics(metrics), compact_predictions(predictions, task_map)
        )
        if args.limit is None
        else "(gate run: paper-side file not written)"
    )
    partial_path.unlink(missing_ok=True)

    def _f(x: float | None) -> str:
        return "N/A" if x is None else f"{x:.4f}"

    print(
        f"\nn={metrics.n_samples}  AUROC={_f(metrics.auroc)}  F1@0.5={_f(metrics.f1)}  "
        f"optF1={_f(metrics.optimal_f1)} @ {_f(metrics.optimal_threshold)}\n"
        f"latency mean {timing.mean_ms:.1f}ms p95 {timing.p95_ms:.1f}ms  "
        f"peak {meta['gpu_peak_mb']:.0f}MB  wall {elapsed / 60:.1f}min\n"
        f"tokens mean {meta['input_tokens_mean']:.0f} p90 {meta['input_tokens_p90']} "
        f"max {meta['input_tokens_max']}\n{native}\n{paper}"
    )
    if metrics.auroc is not None and metrics.auroc < 0.5:
        print(
            "WARNING: AUROC below 0.5. P(' Yes') is anti-correlated with the "
            "hallucination label, which points at the verdict question's polarity."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
