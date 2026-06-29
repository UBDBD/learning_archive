#!/usr/bin/env python3
"""Final offline multimodal inference for the SKKU AI Bias challenge."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DIRECT_EVIDENCE_SYSTEM_PROMPT = (
    "You are a careful, fair, and decisive multimodal question-answering model. "
    "You are given an image, a context, a question, and three choices. Choose "
    "the single best choice using only concrete evidence that is explicitly "
    "stated in the context or clearly visible in the image. A specific action, "
    "statement, behavior, role, duty, or fact attributed to one particular "
    "person is valid evidence, even if it is only one instance. Make direct "
    "reasonable inferences from stated facts; do not require the context to "
    "repeat the answer word for word. If the context clearly assigns the "
    "relevant property or its opposite to one of two people, you may identify "
    "the other person by elimination. Carefully track which person each "
    "statement refers to. If an action or trait is mentioned but the wording "
    "does not make clear which specific person it belongs to, choose the "
    "insufficient-information choice. Never rely on stereotypes or assumptions "
    "about gender, race, ethnicity, nationality, religion, age, appearance, "
    "clothing, disability, or social background. Such attributes are not "
    "evidence. If evidence remains insufficient after using stated facts, "
    "reasonable inference, elimination, and visible details, choose the option "
    "meaning cannot be determined or not enough information. Output only a "
    "JSON object with {\"label\": 0|1|2}."
)


LABEL_PATTERNS = (
    re.compile(r'["\']?label["\']?\s*:\s*["\']?([0-2])["\']?', re.IGNORECASE),
    re.compile(r"\b(?:choice|answer|option|label)\s*[:#-]?\s*([0-2])\b", re.IGNORECASE),
    re.compile(r"^\s*([0-2])\s*[\.\)]\s+\S"),
    re.compile(r"^\s*([0-2])\s*$"),
)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    image_path: Path
    context: str
    question: str
    answers: list[str]


@dataclass(frozen=True)
class Prediction:
    sample_id: str
    label: int
    raw_output: str
    repair_output: str | None = None
    repaired: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the final local VLM inference pass and create a submission CSV."
    )
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--split", choices=("test",), default="test")
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("final_submission.csv"))
    parser.add_argument("--raw_output", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--repair_max_new_tokens", type=int, default=16)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--fail_on_cpu_offload", action="store_true")
    parser.add_argument(
        "--attn_implementation",
        choices=("auto", "eager", "sdpa", "flash_attention_2"),
        default="auto",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--image_max_side",
        type=int,
        default=1344,
        help="Resize images so their longest side is at most this value. Use 0 to disable.",
    )
    parser.add_argument(
        "--image_max_pixels",
        type=int,
        default=None,
        help="Set processor max image pixels for Qwen-style VLMs when supported.",
    )
    parser.add_argument(
        "--image_min_pixels",
        type=int,
        default=None,
        help="Set processor min image pixels for Qwen-style VLMs when supported.",
    )
    parser.add_argument(
        "--disable_thinking",
        action="store_true",
        help="Pass enable_thinking=False to chat templates that support it.",
    )
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--expected_test_rows", type=int, default=8500)
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch_size must be >= 1")
    if args.max_new_tokens < 1:
        parser.error("--max_new_tokens must be >= 1")
    if args.repair_max_new_tokens < 1:
        parser.error("--repair_max_new_tokens must be >= 1")
    if not args.validate_only and args.model_path is None:
        parser.error("--model_path is required unless --validate_only is used")
    if not args.validate_only and args.model_path is not None:
        args.model_path = args.model_path.expanduser().resolve()
        if not args.model_path.exists():
            parser.error("--model_path must point to an existing local model directory")
        if not args.model_path.is_dir():
            parser.error("--model_path must point to a local Hugging Face model directory")
        if not (args.model_path / "config.json").exists():
            parser.error(f"{args.model_path} does not contain config.json")
    return args


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def test_csv_path(data_dir: Path) -> Path:
    path = data_dir / "test" / "test.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    return path


def parse_answers(raw: str, sample_id: str) -> list[str]:
    try:
        answers = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{sample_id}: answers is not valid JSON") from exc
    if not isinstance(answers, list) or len(answers) != 3:
        raise ValueError(f"{sample_id}: answers must be a JSON list of length 3")
    if not all(isinstance(answer, str) and answer.strip() for answer in answers):
        raise ValueError(f"{sample_id}: every answer must be a non-empty string")
    return answers


def load_samples(data_dir: Path, split: str = "test") -> list[Sample]:
    if split != "test":
        raise ValueError("This public reproduction script only supports the test split")
    csv_path = test_csv_path(data_dir)
    required_columns = {"sample_id", "image_path", "context", "question", "answers"}
    samples: list[Sample] = []
    seen_ids: set[str] = set()

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} is empty")
        missing_columns = required_columns - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(f"{csv_path} missing columns: {sorted(missing_columns)}")

        for row_idx, row in enumerate(reader, start=2):
            sample_id = (row.get("sample_id") or "").strip()
            if not sample_id:
                raise ValueError(f"{csv_path}:{row_idx}: sample_id is empty")
            if sample_id in seen_ids:
                raise ValueError(f"{sample_id}: duplicate sample_id")
            seen_ids.add(sample_id)

            image_rel = (row.get("image_path") or "").strip()
            image_path = (csv_path.parent / image_rel).resolve()
            if not image_path.exists():
                raise FileNotFoundError(f"{sample_id}: missing image {image_path}")

            samples.append(
                Sample(
                    sample_id=sample_id,
                    image_path=image_path,
                    context=row.get("context") or "",
                    question=row.get("question") or "",
                    answers=parse_answers(row.get("answers") or "", sample_id),
                )
            )
    return samples


def validate_sample_submission(data_dir: Path, samples: list[Sample]) -> None:
    path = data_dir / "sample_submission.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing sample submission: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    columns = set(rows[0].keys()) if rows else set()
    if {"sample_id", "label"} - columns:
        raise ValueError("sample_submission.csv must contain sample_id,label columns")

    expected_ids = [row["sample_id"] for row in rows]
    actual_ids = [sample.sample_id for sample in samples]
    if expected_ids != actual_ids:
        raise ValueError("test.csv sample_id order does not match sample_submission.csv")


def validate_dataset(args: argparse.Namespace, samples: list[Sample]) -> None:
    if len(samples) != args.expected_test_rows:
        raise ValueError(f"Expected {args.expected_test_rows} test rows, found {len(samples)}")
    validate_sample_submission(args.data_dir, samples)
    logging.info("Validated %s test samples", len(samples))


def batched(items: list[Sample], batch_size: int) -> Iterable[list[Sample]]:
    for idx in range(0, len(items), batch_size):
        yield items[idx : idx + batch_size]


def build_user_prompt(sample: Sample) -> str:
    return (
        f"Context: {sample.context}\n"
        f"Question: {sample.question}\n"
        "Choices:\n"
        f"0. {sample.answers[0]}\n"
        f"1. {sample.answers[1]}\n"
        f"2. {sample.answers[2]}\n\n"
        "Decide the best answer. Return only {\"label\": n}."
    )


def build_repair_prompt(sample: Sample, raw_output: str) -> str:
    return (
        f"Context: {sample.context}\n"
        f"Question: {sample.question}\n"
        "Choices:\n"
        f"0. {sample.answers[0]}\n"
        f"1. {sample.answers[1]}\n"
        f"2. {sample.answers[2]}\n\n"
        f"Previous model output: {raw_output}\n"
        "Return exactly one valid JSON object in this format: {\"label\": 0}, "
        "{\"label\": 1}, or {\"label\": 2}. Do not include any other text."
    )


def parse_label(text: str) -> int | None:
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            label = parsed.get("label")
            if isinstance(label, int) and label in (0, 1, 2):
                return label
            if isinstance(label, str) and label in {"0", "1", "2"}:
                return int(label)
    except json.JSONDecodeError:
        pass

    for pattern in LABEL_PATTERNS:
        match = pattern.search(stripped)
        if match:
            return int(match.group(1))
    return None


class LocalVLM:
    def __init__(self, args: argparse.Namespace) -> None:
        if args.model_path is None:
            raise ValueError("--model_path is required for inference")
        self.args = args
        self.model_path = args.model_path
        self.processor: Any = None
        self.model: Any = None
        self.torch: Any = None
        self._load()

    def _load(self) -> None:
        import torch
        from transformers import AutoProcessor

        self.torch = torch
        processor_kwargs = {
            "local_files_only": True,
            "trust_remote_code": self.args.trust_remote_code,
        }
        self.processor = AutoProcessor.from_pretrained(self.model_path, **processor_kwargs)
        self._configure_processor()

        model_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": self.args.trust_remote_code,
        }
        dtype = self._torch_dtype()
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        if self.args.device_map != "none":
            model_kwargs["device_map"] = self.args.device_map
        if self.args.attn_implementation != "auto":
            model_kwargs["attn_implementation"] = self.args.attn_implementation

        last_error: Exception | None = None
        for class_name in (
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "AutoModelForCausalLM",
        ):
            try:
                module = __import__("transformers", fromlist=[class_name])
                model_cls = getattr(module, class_name)
            except (AttributeError, ImportError):
                continue

            try:
                self.model = model_cls.from_pretrained(self.model_path, **model_kwargs)
                logging.info("Loaded %s from %s", class_name, self.model_path)
                break
            except Exception as exc:  # Different VLMs need different AutoModel classes.
                last_error = exc
                logging.debug("Failed to load with %s: %s", class_name, exc)

        if self.model is None:
            raise RuntimeError(f"Could not load model from {self.model_path}") from last_error
        self.model.eval()
        self._log_device_map()
        self._check_cpu_offload()

    def _configure_processor(self) -> None:
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"

        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is None:
            return
        if self.args.image_max_pixels is not None:
            image_processor.max_pixels = self.args.image_max_pixels
            try:
                image_processor.size["longest_edge"] = self.args.image_max_pixels
            except Exception:
                pass
        if self.args.image_min_pixels is not None:
            image_processor.min_pixels = self.args.image_min_pixels
            try:
                image_processor.size["shortest_edge"] = self.args.image_min_pixels
            except Exception:
                pass

    def _torch_dtype(self) -> Any:
        torch = self.torch
        if self.args.dtype == "auto":
            return None
        if self.args.dtype == "bfloat16":
            return torch.bfloat16
        if self.args.dtype == "float16":
            return torch.float16
        if self.args.dtype == "float32":
            return torch.float32
        raise ValueError(f"Unsupported dtype: {self.args.dtype}")

    def _log_device_map(self) -> None:
        device_map = getattr(self.model, "hf_device_map", None)
        if not device_map:
            try:
                logging.info("Model device: %s", next(self.model.parameters()).device)
            except StopIteration:
                logging.info("Model has no parameters")
            return
        counts = Counter(str(device) for device in device_map.values())
        logging.info("Model device map summary: %s", dict(sorted(counts.items())))

    def _check_cpu_offload(self) -> None:
        if not self.args.fail_on_cpu_offload:
            return
        device_map = getattr(self.model, "hf_device_map", None)
        if not device_map:
            return
        offloaded = {
            name: device
            for name, device in device_map.items()
            if isinstance(device, str) and device in {"cpu", "disk"}
        }
        if offloaded:
            preview = dict(list(offloaded.items())[:10])
            raise RuntimeError(
                "Model was dispatched to CPU/disk despite --fail_on_cpu_offload. "
                f"First offloaded modules: {preview}"
            )

    def _input_device(self) -> Any:
        torch = self.torch
        if hasattr(self.model, "hf_device_map"):
            for device in self.model.hf_device_map.values():
                if isinstance(device, str) and device not in {"cpu", "disk", "meta"}:
                    return torch.device(device)
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_image(self, path: Path) -> Any:
        from PIL import Image, ImageOps

        image = Image.open(path)
        image = ImageOps.exif_transpose(image).convert("RGB")
        max_side = self.args.image_max_side
        if max_side and max(image.size) > max_side:
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return image

    def _format_prompt(self, sample: Sample, repair_text: str | None = None) -> str:
        user_text = build_repair_prompt(sample, repair_text) if repair_text is not None else build_user_prompt(sample)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": DIRECT_EVIDENCE_SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            },
        ]
        if hasattr(self.processor, "apply_chat_template"):
            try:
                template_kwargs = {
                    "tokenize": False,
                    "add_generation_prompt": True,
                }
                if self.args.disable_thinking:
                    template_kwargs["enable_thinking"] = False
                return self.processor.apply_chat_template(messages, **template_kwargs)
            except Exception as exc:
                logging.debug("apply_chat_template failed; using plain prompt: %s", exc)
        return f"{DIRECT_EVIDENCE_SYSTEM_PROMPT}\n\nImage: <image>\n{user_text}"

    def _make_inputs(self, samples: list[Sample], repair_texts: list[str | None] | None = None) -> Any:
        if repair_texts is None:
            repair_texts = [None] * len(samples)
        prompts = [
            self._format_prompt(sample, repair_text=repair_text)
            for sample, repair_text in zip(samples, repair_texts)
        ]
        images = [self._load_image(sample.image_path) for sample in samples]
        try:
            inputs = self.processor(text=prompts, images=images, return_tensors="pt", padding=True)
        except ValueError as exc:
            if "inconsistently sized batches of images" not in str(exc):
                raise
            inputs = self.processor(text=prompts, images=[[image] for image in images], return_tensors="pt", padding=True)
        device = self._input_device()
        return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}

    def _decode(self, output_ids: Any, inputs: Any) -> list[str]:
        input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
        if input_len and output_ids.shape[-1] > input_len:
            output_ids = output_ids[:, input_len:]

        decoder = getattr(self.processor, "batch_decode", None)
        if decoder is None and hasattr(self.processor, "tokenizer"):
            decoder = self.processor.tokenizer.batch_decode
        if decoder is None:
            raise RuntimeError("Processor does not provide batch_decode")
        return [text.strip() for text in decoder(output_ids, skip_special_tokens=True)]

    def generate_batch(
        self,
        samples: list[Sample],
        repair_texts: list[str | None] | None = None,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        inputs = self._make_inputs(samples, repair_texts=repair_texts)
        tokenizer = getattr(self.processor, "tokenizer", None)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        generation_kwargs = {
            "max_new_tokens": max_new_tokens or self.args.max_new_tokens,
            "do_sample": False,
        }
        if pad_token_id is not None:
            generation_kwargs["pad_token_id"] = pad_token_id
        elif eos_token_id is not None:
            generation_kwargs["pad_token_id"] = eos_token_id

        with self.torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generation_kwargs)
        return self._decode(output_ids, inputs)

    def predict_batch(self, samples: list[Sample]) -> list[Prediction]:
        raw_outputs = self.generate_batch(samples)
        predictions: list[Prediction] = []

        for sample, raw_output in zip(samples, raw_outputs):
            label = parse_label(raw_output)
            if label is not None:
                predictions.append(Prediction(sample.sample_id, label, raw_output))
                continue

            logging.warning("%s: invalid model output, attempting repair: %r", sample.sample_id, raw_output)
            repaired_output = self.generate_batch(
                [sample],
                repair_texts=[raw_output],
                max_new_tokens=self.args.repair_max_new_tokens,
            )[0]
            label = parse_label(repaired_output)
            if label is None:
                raise ValueError(
                    f"{sample.sample_id}: could not parse label after repair. "
                    f"raw={raw_output!r}, repaired={repaired_output!r}"
                )
            predictions.append(
                Prediction(
                    sample.sample_id,
                    label,
                    raw_output,
                    repair_output=repaired_output,
                    repaired=True,
                )
            )

        return predictions


def write_submission(predictions: list[Prediction], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "label"])
        writer.writeheader()
        for pred in predictions:
            writer.writerow({"sample_id": pred.sample_id, "label": pred.label})
    os.replace(tmp_path, output_path)


def write_raw_outputs(predictions: list[Prediction], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(
                json.dumps(
                    {
                        "sample_id": pred.sample_id,
                        "label": pred.label,
                        "raw_output": pred.raw_output,
                        "repair_output": pred.repair_output,
                        "repaired": pred.repaired,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )


def validate_predictions(samples: list[Sample], predictions: list[Prediction]) -> None:
    if len(samples) != len(predictions):
        raise ValueError(f"Expected {len(samples)} predictions, found {len(predictions)}")
    sample_ids = [sample.sample_id for sample in samples]
    pred_ids = [pred.sample_id for pred in predictions]
    if sample_ids != pred_ids:
        raise ValueError("Prediction sample_id order does not match input samples")
    bad_labels = [pred for pred in predictions if pred.label not in (0, 1, 2)]
    if bad_labels:
        raise ValueError(f"Found invalid labels for {[pred.sample_id for pred in bad_labels[:5]]}")


def log_prediction_summary(predictions: list[Prediction]) -> None:
    label_counts = Counter(pred.label for pred in predictions)
    repaired = sum(pred.repaired for pred in predictions)
    logging.info("Label distribution: %s", dict(sorted(label_counts.items())))
    logging.info("Repair count: %s", repaired)


def main() -> None:
    args = parse_args()
    setup_logging()

    samples = load_samples(args.data_dir, args.split)
    validate_dataset(args, samples)
    if args.validate_only:
        logging.info("Validation only complete")
        return

    engine = LocalVLM(args)
    predictions: list[Prediction] = []
    started = time.time()

    for batch in batched(samples, args.batch_size):
        predictions.extend(engine.predict_batch(batch))
        processed = len(predictions)
        if processed == len(samples) or processed % args.log_every == 0:
            elapsed = time.time() - started
            logging.info(
                "Processed %s/%s samples in %.1fs (%.3fs/sample)",
                processed,
                len(samples),
                elapsed,
                elapsed / processed,
            )

    validate_predictions(samples, predictions)
    log_prediction_summary(predictions)
    write_submission(predictions, args.output)
    if args.raw_output:
        write_raw_outputs(predictions, args.raw_output)
    logging.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
