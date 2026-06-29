#!/usr/bin/env python3
"""Final evidence-gate adjudication for main/high-pixel disagreements."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infer import LocalVLM, Prediction, Sample, load_samples, parse_label, validate_predictions


EVIDENCE_GATE_SYSTEM_PROMPT = (
    "You are a careful, fair, and decisive multimodal adjudicator. You are given "
    "an image, a context, a question, three choices, and two previous local model "
    "answers. The main-resolution answer used the standard competition setting. "
    "The high-pixel answer used a larger image pixel budget. Use the previous "
    "answers only as diagnostic hints; do not vote or average them. Independently "
    "decide the single best choice using the image and context. If the context "
    "alone explicitly answers the question, use that evidence even if the image is "
    "distracting. If the answer requires visual evidence, inspect the image. First "
    "classify the decisive evidence as one of: context_explicit, image_concrete, "
    "context_image_linked, or insufficient. Use context_explicit when the text "
    "states a specific action, fact, role, or its opposite for a specific person. "
    "Use image_concrete only for unambiguous visible objects, actions, spatial "
    "relations, readable signs, or clearly linked identifiers. Use "
    "context_image_linked only when the context identifies people or objects and "
    "the image supplies a concrete missing visual fact. Use insufficient when the "
    "decision would rely on social stereotypes, appearance, posture, clothing, "
    "vague emotion, or an unlinked label/card/sign. Prefer the main-resolution "
    "answer unless the evidence classification makes the high-pixel answer "
    "clearly better. Output only a JSON object with {\"label\": 0|1|2}."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--split", choices=("test",), default="test")
    parser.add_argument("--expected_test_rows", type=int, required=True)
    parser.add_argument("--model_path", type=Path, default=Path("models/Qwen3.5-9B"))
    parser.add_argument("--base_submission", type=Path, required=True)
    parser.add_argument("--candidate_submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw_output", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--repair_max_new_tokens", type=int, default=32)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--fail_on_cpu_offload", action="store_true")
    parser.add_argument("--attn_implementation", choices=("auto", "eager", "sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--image_max_side", type=int, default=0)
    parser.add_argument("--image_max_pixels", type=int, default=401408)
    parser.add_argument("--image_min_pixels", type=int, default=50176)
    parser.add_argument("--disable_thinking", action="store_true")
    parser.add_argument("--log_every", type=int, default=200)
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch_size must be >= 1")
    if args.max_new_tokens < 1:
        parser.error("--max_new_tokens must be >= 1")
    if args.repair_max_new_tokens < 1:
        parser.error("--repair_max_new_tokens must be >= 1")
    return args


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")


def read_submission(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out: dict[str, int] = {}
    for row in rows:
        label = int(row["label"])
        if label not in (0, 1, 2):
            raise ValueError(f"{path}: invalid label {label} for {row['sample_id']}")
        out[row["sample_id"]] = label
    return out


def write_submission(predictions: list[Prediction], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "label"])
        writer.writeheader()
        for pred in predictions:
            writer.writerow({"sample_id": pred.sample_id, "label": pred.label})
    os.replace(tmp_path, path)


def write_raw(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def engine_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model_path=args.model_path.expanduser().resolve(),
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        fail_on_cpu_offload=args.fail_on_cpu_offload,
        image_max_side=args.image_max_side,
        image_max_pixels=args.image_max_pixels,
        image_min_pixels=args.image_min_pixels,
        disable_thinking=args.disable_thinking,
    )


def build_user_prompt(sample: Sample, base_label: int, candidate_label: int) -> str:
    options = "\n".join(f"{idx}. {answer}" for idx, answer in enumerate(sample.answers))
    return (
        f"Context: {sample.context}\n"
        f"Question: {sample.question}\n"
        f"Choices:\n{options}\n\n"
        f"Previous main-resolution answer: {base_label}. {sample.answers[base_label]}\n"
        f"Previous high-pixel answer: {candidate_label}. {sample.answers[candidate_label]}\n\n"
        'Adjudicate independently from the image and context. Return only {"label": n}.'
    )


def format_prompt(engine: LocalVLM, prompt: str, user_text: str) -> str:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        },
    ]
    if hasattr(engine.processor, "apply_chat_template"):
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if engine.args.disable_thinking:
            template_kwargs["enable_thinking"] = False
        return engine.processor.apply_chat_template(messages, **template_kwargs)
    return f"{prompt}\n\nImage: <image>\n{user_text}"


def make_inputs(engine: LocalVLM, samples: list[Sample], prompts: list[str]) -> Any:
    images = [engine._load_image(sample.image_path) for sample in samples]
    try:
        inputs = engine.processor(text=prompts, images=images, return_tensors="pt", padding=True)
    except ValueError as exc:
        if "inconsistently sized batches of images" not in str(exc):
            raise
        inputs = engine.processor(text=prompts, images=[[image] for image in images], return_tensors="pt", padding=True)
    device = engine._input_device()
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def generate(engine: LocalVLM, samples: list[Sample], prompts: list[str], max_new_tokens: int) -> list[str]:
    inputs = make_inputs(engine, samples, prompts)
    tokenizer = getattr(engine.processor, "tokenizer", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    generation_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = pad_token_id
    elif eos_token_id is not None:
        generation_kwargs["pad_token_id"] = eos_token_id
    with engine.torch.inference_mode():
        output_ids = engine.model.generate(**inputs, **generation_kwargs)
    return engine._decode(output_ids, inputs)


def batched(items: list[tuple[Sample, int, int]], batch_size: int) -> Any:
    for idx in range(0, len(items), batch_size):
        yield items[idx : idx + batch_size]


def repair_prompt(user_text: str, bad_output: str) -> str:
    return (
        f"{user_text}\n\n"
        f"Your previous output was invalid: {bad_output!r}\n"
        'Return only a valid JSON object with {"label": 0}, {"label": 1}, or {"label": 2}.'
    )


def main() -> None:
    args = parse_args()
    setup_logging()
    samples = load_samples(args.data_dir, args.split)
    if len(samples) != args.expected_test_rows:
        raise ValueError(f"Expected {args.expected_test_rows} test rows, found {len(samples)}")

    base = read_submission(args.base_submission)
    candidate = read_submission(args.candidate_submission)
    for sample in samples:
        if sample.sample_id not in base:
            raise ValueError(f"{args.base_submission}: missing {sample.sample_id}")
        if sample.sample_id not in candidate:
            raise ValueError(f"{args.candidate_submission}: missing {sample.sample_id}")

    disagreements = [
        (sample, base[sample.sample_id], candidate[sample.sample_id])
        for sample in samples
        if base[sample.sample_id] != candidate[sample.sample_id]
    ]
    logging.info("Rows: %s | disagreements to adjudicate: %s", len(samples), len(disagreements))

    judged: dict[str, tuple[int, str, bool]] = {}
    if disagreements:
        engine = LocalVLM(engine_args(args))
        processed = 0
        t0 = time.time()
        for batch in batched(disagreements, args.batch_size):
            batch_samples = [item[0] for item in batch]
            batch_user_texts = [build_user_prompt(*item) for item in batch]
            prompts = [format_prompt(engine, EVIDENCE_GATE_SYSTEM_PROMPT, user_text) for user_text in batch_user_texts]
            outputs = generate(engine, batch_samples, prompts, args.max_new_tokens)
            for sample, user_text, output in zip(batch_samples, batch_user_texts, outputs):
                label = parse_label(output)
                repaired = False
                final_output = output
                if label is None:
                    repair = repair_prompt(user_text, output)
                    repair_output = generate(
                        engine,
                        [sample],
                        [format_prompt(engine, EVIDENCE_GATE_SYSTEM_PROMPT, repair)],
                        args.repair_max_new_tokens,
                    )[0]
                    label = parse_label(repair_output)
                    if label is None:
                        raise ValueError(
                            f"{sample.sample_id}: could not parse adjudication output. "
                            f"raw={output!r}, repair={repair_output!r}"
                        )
                    repaired = True
                    final_output = repair_output
                judged[sample.sample_id] = (label, final_output, repaired)
            processed += len(batch)
            if processed % args.log_every == 0 or processed == len(disagreements):
                dt = time.time() - t0
                logging.info(
                    "Adjudicated %s/%s disagreements in %.1fs (%.3fs/disagreement)",
                    processed,
                    len(disagreements),
                    dt,
                    dt / processed,
                )

    predictions: list[Prediction] = []
    raw_records: list[dict[str, Any]] = []
    changed_from_base = 0
    chose_candidate = 0
    chose_third = 0
    for sample in samples:
        base_label = base[sample.sample_id]
        candidate_label = candidate[sample.sample_id]
        if sample.sample_id in judged:
            final_label, judge_output, repaired = judged[sample.sample_id]
            adjudicated = True
            changed_from_base += final_label != base_label
            chose_candidate += final_label == candidate_label
            chose_third += final_label not in (base_label, candidate_label)
        else:
            final_label, judge_output, repaired = base_label, None, False
            adjudicated = False
        predictions.append(
            Prediction(
                sample_id=sample.sample_id,
                label=final_label,
                raw_output=judge_output or json.dumps({"label": final_label}),
                repaired=repaired,
            )
        )
        raw_records.append(
            {
                "sample_id": sample.sample_id,
                "label": final_label,
                "base_label": base_label,
                "candidate_label": candidate_label,
                "adjudicated": adjudicated,
                "judge_output": judge_output,
                "repaired": repaired,
            }
        )

    validate_predictions(samples, predictions)
    write_submission(predictions, args.output)
    write_raw(raw_records, args.raw_output)
    counts = Counter(pred.label for pred in predictions)
    repairs = sum(record["repaired"] for record in raw_records)
    logging.info("Label distribution: %s", dict(sorted(counts.items())))
    logging.info("Repair count: %s", repairs)
    logging.info("Changed from base: %s", changed_from_base)
    logging.info("Chose candidate on disagreements: %s", chose_candidate)
    logging.info("Chose third label on disagreements: %s", chose_third)
    logging.info("Wrote %s", args.output)
    logging.info("Wrote %s", args.raw_output)


if __name__ == "__main__":
    main()
