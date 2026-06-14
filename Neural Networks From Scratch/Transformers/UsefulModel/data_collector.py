"""
Build clean text corpora for a small GPT-style conversational model.

Default output:
    Data/pretrain_train.txt
    Data/pretrain_val.txt
    Data/sft_train.txt
    Data/sft_val.txt
    Data/preview.txt
    Data/manifest.json

The defaults are sized for a practical first run under roughly 50M parameters:
about 120M estimated base-pretraining tokens and 8M estimated SFT tokens.
Use smaller values for quick tests, for example:

    python data_collector.py --pretrain-tokens 1000000 --sft-tokens 200000
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator


EOS_TOKEN = "<|endoftext|>"
DOC_TOKEN = "<|document|>"
INST_TOKEN = "[INST]"
END_INST_TOKEN = "[/INST]"

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "Data"
DEFAULT_PRETRAIN_TOKENS = 120_000_000
DEFAULT_SFT_TOKENS = 8_000_000
DEFAULT_CHARS_PER_TOKEN = 4.0

TEXT_MIN_CHARS = 180
TEXT_MAX_CHARS = 32_000
CHAT_MIN_CHARS = 40
CHAT_MAX_CHARS = 12_000

CONTROL_CHARS = dict.fromkeys(range(0, 9), None)
CONTROL_CHARS.update(dict.fromkeys(range(11, 32), None))
WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
NEWLINE_RE = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class Source:
    name: str
    weight: float
    iterator_factory: Callable[[], Iterator[str]]


def clean_text(text: object, max_chars: int = TEXT_MAX_CHARS) -> str:
    if text is None:
        return ""

    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(CONTROL_CHARS)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(WHITESPACE_RE.sub(" ", line).strip() for line in text.split("\n"))
    text = NEWLINE_RE.sub("\n\n", text).strip()

    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].strip()

    return text


def looks_like_good_text(text: str, min_chars: int = TEXT_MIN_CHARS) -> bool:
    if len(text) < min_chars:
        return False

    alpha = sum(ch.isalpha() for ch in text)
    if alpha / max(1, len(text)) < 0.45:
        return False

    words = text.split()
    if len(words) < 35:
        return False

    unique_ratio = len(set(words)) / max(1, len(words))
    if len(words) > 80 and unique_ratio < 0.18:
        return False

    lowered = text.lower()
    junk_phrases = (
        "enable javascript",
        "cookie policy",
        "all rights reserved",
        "subscribe to our newsletter",
        "click here",
    )
    if sum(phrase in lowered for phrase in junk_phrases) >= 2:
        return False

    return True


def format_document(text: str) -> str:
    return f"{DOC_TOKEN}\n{text}\n{EOS_TOKEN}\n"


def format_inst_pair(user: str, assistant: str, system: str = "", add_eos: bool = False) -> str:
    user = clean_text(user, max_chars=CHAT_MAX_CHARS)
    assistant = clean_text(assistant, max_chars=CHAT_MAX_CHARS)
    system = clean_text(system, max_chars=CHAT_MAX_CHARS)

    if not user or not assistant:
        return ""

    if system:
        user = f"<<SYS>>\n{system}\n<</SYS>>\n\n{user}"

    rendered = f"{INST_TOKEN} {user} {END_INST_TOKEN} {assistant}"
    if add_eos:
        rendered = f"{rendered}\n{EOS_TOKEN}\n"

    if len(rendered) < CHAT_MIN_CHARS or len(rendered) > CHAT_MAX_CHARS:
        return ""

    return rendered


def normalize_role(role: object) -> str:
    role = str(role or "").lower().strip()
    if role in {"human", "prompter"}:
        return "user"
    if role in {"gpt", "bot", "assistant"}:
        return "assistant"
    if role == "system":
        return "system"
    return role


def format_messages(messages: object) -> str:
    if not isinstance(messages, list):
        return ""

    turns: list[str] = []
    system_parts: list[str] = []
    pending_user = ""
    used_system = False

    for message in messages:
        if not isinstance(message, dict):
            continue

        role = normalize_role(message.get("role"))
        content = clean_text(message.get("content"), max_chars=CHAT_MAX_CHARS)
        if not content:
            continue

        if role == "system":
            system_parts.append(content)
        elif role == "user":
            pending_user = content
        elif role == "assistant" and pending_user:
            system = "\n\n".join(system_parts) if system_parts and not used_system else ""
            pair = format_inst_pair(pending_user, content, system=system, add_eos=False)
            if pair:
                turns.append(pair.rstrip())
                used_system = True
            pending_user = ""

    if not turns:
        return ""

    rendered = "\n".join(turns) + f"\n{EOS_TOKEN}\n"
    if len(rendered) > CHAT_MAX_CHARS:
        return ""

    return rendered


def require_datasets():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install the Hugging Face datasets package before running this script: "
            "pip install datasets"
        ) from exc

    return load_dataset


def iter_smollm_corpus(
    subset: str,
    *,
    split: str,
    seed: int,
    shuffle_buffer: int,
) -> Iterator[str]:
    load_dataset = require_datasets()
    ds = load_dataset("HuggingFaceTB/smollm-corpus", subset, split=split, streaming=True)
    if shuffle_buffer > 0:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)

    for example in ds:
        text = clean_text(example.get("text"))
        if looks_like_good_text(text):
            yield format_document(text)


def iter_fineweb_edu_sample(
    *,
    split: str,
    seed: int,
    shuffle_buffer: int,
) -> Iterator[str]:
    load_dataset = require_datasets()
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split=split, streaming=True)
    if shuffle_buffer > 0:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)

    for example in ds:
        text = clean_text(example.get("text"))
        if looks_like_good_text(text):
            yield format_document(text)


def iter_smol_smoltalk(
    *,
    split: str,
    seed: int,
    shuffle_buffer: int,
) -> Iterator[str]:
    load_dataset = require_datasets()
    ds = load_dataset("HuggingFaceTB/smol-smoltalk", split=split, streaming=True)
    if shuffle_buffer > 0:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)

    for example in ds:
        rendered = format_messages(example.get("messages"))
        if rendered:
            yield rendered


def iter_oasst1_pairs(
    *,
    split: str,
    seed: int,
    shuffle_buffer: int,
    max_rank: int,
) -> Iterator[str]:
    load_dataset = require_datasets()
    ds = load_dataset("OpenAssistant/oasst1", split=split, streaming=True)
    if shuffle_buffer > 0:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)

    prompts_by_id: dict[str, str] = {}

    for example in ds:
        if example.get("lang") != "en":
            continue
        if example.get("deleted") is True or example.get("review_result") is False:
            continue

        message_id = str(example.get("message_id") or "")
        parent_id = str(example.get("parent_id") or "")
        role = normalize_role(example.get("role"))
        text = clean_text(example.get("text"), max_chars=CHAT_MAX_CHARS)
        if not message_id or not text:
            continue

        if role == "user":
            prompts_by_id[message_id] = text
            continue

        if role != "assistant" or parent_id not in prompts_by_id:
            continue

        rank = example.get("rank")
        if rank is not None:
            try:
                if int(rank) > max_rank:
                    continue
            except (TypeError, ValueError):
                pass

        rendered = format_inst_pair(prompts_by_id[parent_id], text, add_eos=True)
        if rendered:
            yield rendered


def choose_source(active: list[dict], rng: random.Random) -> dict:
    total = sum(item["weight"] for item in active)
    pick = rng.random() * total
    upto = 0.0

    for item in active:
        upto += item["weight"]
        if upto >= pick:
            return item

    return active[-1]


class SplitWriter:
    def __init__(
        self,
        out_dir: Path,
        prefix: str,
        *,
        val_fraction: float,
        seed: int,
        preview_limit: int,
    ):
        self.out_dir = out_dir
        self.prefix = prefix
        self.val_fraction = val_fraction
        self.rng = random.Random(seed)
        self.preview_limit = preview_limit
        self.preview: list[dict[str, str]] = []
        self.stats = {
            "train_samples": 0,
            "val_samples": 0,
            "train_chars": 0,
            "val_chars": 0,
            "sources": defaultdict(lambda: {"samples": 0, "chars": 0}),
        }

        self.train_path = out_dir / f"{prefix}_train.txt"
        self.val_path = out_dir / f"{prefix}_val.txt"
        self._train = self.train_path.open("w", encoding="utf-8")
        self._val = self.val_path.open("w", encoding="utf-8")

    def write(self, source_name: str, text: str) -> int:
        text = text.strip() + "\n"
        chars = len(text)
        is_val = self.rng.random() < self.val_fraction
        handle = self._val if is_val else self._train
        split = "val" if is_val else "train"

        handle.write(text)
        handle.write("\n")

        self.stats[f"{split}_samples"] += 1
        self.stats[f"{split}_chars"] += chars
        self.stats["sources"][source_name]["samples"] += 1
        self.stats["sources"][source_name]["chars"] += chars

        if len(self.preview) < self.preview_limit:
            self.preview.append(
                {
                    "stage": self.prefix,
                    "source": source_name,
                    "text": text[:1200],
                }
            )

        return chars

    def close(self):
        self._train.close()
        self._val.close()
        self.stats["sources"] = dict(self.stats["sources"])


def collect_stage(
    *,
    stage_name: str,
    sources: list[Source],
    target_chars: int,
    out_dir: Path,
    val_fraction: float,
    seed: int,
    preview_limit: int,
    log_every_chars: int,
    max_source_retries: int,
    retry_base_seconds: float,
) -> tuple[dict, list[dict[str, str]]]:
    writer = SplitWriter(
        out_dir,
        stage_name,
        val_fraction=val_fraction,
        seed=seed,
        preview_limit=preview_limit,
    )

    rng = random.Random(seed)
    active = [
        {
            "name": source.name,
            "weight": source.weight,
            "factory": source.iterator_factory,
            "iterator": source.iterator_factory(),
            "retries": 0,
        }
        for source in sources
        if source.weight > 0
    ]

    chars_written = 0
    next_log_at = log_every_chars
    start_time = time.time()

    try:
        while chars_written < target_chars and active:
            selected = choose_source(active, rng)
            try:
                text = next(selected["iterator"])
            except StopIteration:
                active.remove(selected)
                continue
            except Exception as exc:
                selected["retries"] += 1
                if selected["retries"] > max_source_retries:
                    raise RuntimeError(
                        f"{stage_name}: source {selected['name']} failed after "
                        f"{max_source_retries} retries"
                    ) from exc

                sleep_for = retry_base_seconds * min(8, 2 ** (selected["retries"] - 1))
                print(
                    f"{stage_name}: source {selected['name']} raised "
                    f"{exc.__class__.__name__}: {exc}. "
                    f"Reopening stream in {sleep_for:.1f}s "
                    f"({selected['retries']}/{max_source_retries})."
                )
                time.sleep(sleep_for)
                selected["iterator"] = selected["factory"]()
                continue

            selected["retries"] = 0
            chars_written += writer.write(selected["name"], text)

            if log_every_chars > 0 and chars_written >= next_log_at:
                elapsed = max(1.0, time.time() - start_time)
                mb = chars_written / 1_000_000
                rate = mb / elapsed
                print(f"{stage_name}: wrote {mb:.1f}M chars at {rate:.2f}M chars/sec")
                next_log_at += log_every_chars
    finally:
        writer.close()

    stats = dict(writer.stats)
    stats["target_chars"] = target_chars
    stats["actual_chars"] = chars_written
    stats["elapsed_seconds"] = round(time.time() - start_time, 2)
    stats["source_weights"] = {source.name: source.weight for source in sources}
    return stats, writer.preview


def build_pretrain_sources(args) -> list[Source]:
    split = "train"

    if args.use_direct_fineweb_edu:
        fineweb_factory = lambda: iter_fineweb_edu_sample(
            split=split,
            seed=args.seed,
            shuffle_buffer=args.shuffle_buffer,
        )
        fineweb_name = "fineweb-edu/sample-10BT"
    else:
        fineweb_factory = lambda: iter_smollm_corpus(
            "fineweb-edu-dedup",
            split=split,
            seed=args.seed,
            shuffle_buffer=args.shuffle_buffer,
        )
        fineweb_name = "smollm-corpus/fineweb-edu-dedup"

    return [
        Source(fineweb_name, args.pretrain_fineweb_weight, fineweb_factory),
        Source(
            "smollm-corpus/cosmopedia-v2",
            args.pretrain_cosmopedia_weight,
            lambda: iter_smollm_corpus(
                "cosmopedia-v2",
                split=split,
                seed=args.seed + 1,
                shuffle_buffer=args.shuffle_buffer,
            ),
        ),
    ]


def build_sft_sources(args) -> list[Source]:
    split = "train"
    return [
        Source(
            "smol-smoltalk",
            args.sft_smoltalk_weight,
            lambda: iter_smol_smoltalk(
                split=split,
                seed=args.seed + 2,
                shuffle_buffer=args.shuffle_buffer,
            ),
        ),
        Source(
            "oasst1/en-ranked",
            args.sft_oasst_weight,
            lambda: iter_oasst1_pairs(
                split=split,
                seed=args.seed + 3,
                shuffle_buffer=args.shuffle_buffer,
                max_rank=args.oasst_max_rank,
            ),
        ),
    ]


def chars_from_token_target(token_target: int, chars_per_token: float) -> int:
    return int(token_target * chars_per_token)


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_manifest(out_dir: Path, args, stage_stats: dict, previews: list[dict[str, str]]):
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "special_tokens": [EOS_TOKEN, DOC_TOKEN, INST_TOKEN, END_INST_TOKEN],
        "format": {
            "pretrain": f"{DOC_TOKEN}\\n...text...\\n{EOS_TOKEN}",
            "sft": f"{INST_TOKEN} user {END_INST_TOKEN} assistant ... {EOS_TOKEN}",
        },
        "args": json_safe(vars(args)),
        "stage_stats": stage_stats,
    }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(json_safe(manifest), indent=2), encoding="utf-8")

    preview_path = out_dir / "preview.txt"
    with preview_path.open("w", encoding="utf-8") as f:
        for i, item in enumerate(previews, start=1):
            f.write(f"===== sample {i}: {item['stage']} / {item['source']} =====\n")
            f.write(item["text"].rstrip())
            f.write("\n\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect clean pretraining and SFT data.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=2_000)
    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument("--chars-per-token", type=float, default=DEFAULT_CHARS_PER_TOKEN)

    parser.add_argument("--pretrain-tokens", type=int, default=DEFAULT_PRETRAIN_TOKENS)
    parser.add_argument("--sft-tokens", type=int, default=DEFAULT_SFT_TOKENS)
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--skip-sft", action="store_true")

    parser.add_argument("--pretrain-fineweb-weight", type=float, default=0.65)
    parser.add_argument("--pretrain-cosmopedia-weight", type=float, default=0.35)
    parser.add_argument("--use-direct-fineweb-edu", action="store_true")

    parser.add_argument("--sft-smoltalk-weight", type=float, default=0.90)
    parser.add_argument("--sft-oasst-weight", type=float, default=0.0)
    parser.add_argument("--oasst-max-rank", type=int, default=1)

    parser.add_argument("--preview-limit", type=int, default=24)
    parser.add_argument("--log-every-chars", type=int, default=10_000_000)
    parser.add_argument("--max-source-retries", type=int, default=8)
    parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    stage_stats = {}
    previews: list[dict[str, str]] = []

    if not args.skip_pretrain:
        pretrain_chars = chars_from_token_target(args.pretrain_tokens, args.chars_per_token)
        stats, preview = collect_stage(
            stage_name="pretrain",
            sources=build_pretrain_sources(args),
            target_chars=pretrain_chars,
            out_dir=args.out_dir,
            val_fraction=args.val_fraction,
            seed=args.seed,
            preview_limit=args.preview_limit,
            log_every_chars=args.log_every_chars,
            max_source_retries=args.max_source_retries,
            retry_base_seconds=args.retry_base_seconds,
        )
        stage_stats["pretrain"] = stats
        previews.extend(preview)

    if not args.skip_sft:
        sft_chars = chars_from_token_target(args.sft_tokens, args.chars_per_token)
        stats, preview = collect_stage(
            stage_name="sft",
            sources=build_sft_sources(args),
            target_chars=sft_chars,
            out_dir=args.out_dir,
            val_fraction=args.val_fraction,
            seed=args.seed + 10,
            preview_limit=args.preview_limit,
            log_every_chars=args.log_every_chars,
            max_source_retries=args.max_source_retries,
            retry_base_seconds=args.retry_base_seconds,
        )
        stage_stats["sft"] = stats
        previews.extend(preview)

    write_manifest(args.out_dir, args, stage_stats, previews)
    print(f"Wrote dataset files to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
