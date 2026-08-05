"""Export CineMatch logs to chat-style JSONL for SFT/LoRA fine-tuning.

The script reads Firestore logs, keeps high-quality turns, and writes one JSON
object per line:

{"messages": [{"role": "system", ...}, {"role": "user", ...}, ...]}

Examples:
  python scripts/export_finetune_dataset.py --days 30 --limit 2000
  python scripts/export_finetune_dataset.py --input-type voice --require-voice-qa
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from firebase_admin import firestore
from google.cloud.firestore_v1 import FieldFilter

from database import COL_CHAT_LOGS, COL_VOICE_AI_EVALUATIONS, _get_db


CHAT_SYSTEM_PROMPT = (
    "Sen CineMatch uygulamasinin resmi Turkce sinema asistanisin. "
    "Yalnizca filmler, diziler, yonetmenler, oyuncular, sinema sektoru ve "
    "CineMatch uygulamasi hakkinda cevap ver. Film onerilerinde kullanicinin "
    "zevklerini dikkate al, dogrulayamadigin sayisal bilgileri uydurma."
)

VOICE_SYSTEM_PROMPT = (
    "Sen CineMatch uygulamasinin Turkce sesli sinema asistanisin. "
    "Cevaplarin dogal, kisa ve konusmaya uygun olsun; normalde 2-4 cumle "
    "kur. Markdown, tablo, baglanti, emoji ve makine isaretleri kullanma. "
    "Dogrulayamadigin sayisal bilgileri uydurma."
)

DEFAULT_ALLOWED_OUTCOMES = {
    "islem_basarili",
    "kapsam_disi_yonlendirildi",
    "veri_bulunamadi",
}

MOVIE_MARKER_RE = re.compile(r"\s*\[\[FILMLER:.*?\]\]\s*$", re.IGNORECASE | re.DOTALL)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?90\s*)?(?:0\s*)?5\d{2}[\s.-]?\d{3}[\s.-]?\d{2}[\s.-]?\d{2}(?!\d)")
SECRETISH_RE = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{12,}|[A-Za-z0-9_-]{32,})\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Firestore chat logs as fine-tuning JSONL."
    )
    parser.add_argument(
        "--output",
        default="var/finetune/cinematch_sft.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Only scan logs created in the last N days. Use 0 for no date filter.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Maximum number of chat log documents to scan.",
    )
    parser.add_argument(
        "--channel",
        default="all",
        help="Filter by channel, for example telegram, api, voice, or all.",
    )
    parser.add_argument(
        "--input-type",
        default="all",
        choices=["all", "text", "voice", "image"],
        help="Filter by input_type.",
    )
    parser.add_argument(
        "--allowed-outcomes",
        default=",".join(sorted(DEFAULT_ALLOWED_OUTCOMES)),
        help="Comma-separated outcome values that may become training examples.",
    )
    parser.add_argument(
        "--min-outcome-confidence",
        type=float,
        default=0.85,
        help="Minimum deterministic outcome confidence.",
    )
    parser.add_argument(
        "--require-voice-qa",
        action="store_true",
        help="For voice rows, require a completed high-score voice QA evaluation.",
    )
    parser.add_argument(
        "--min-voice-overall-score",
        type=float,
        default=80.0,
        help="Minimum voice QA overall_score when --require-voice-qa is used.",
    )
    parser.add_argument(
        "--min-voice-match-score",
        type=float,
        default=85.0,
        help="Minimum transcript match_score when --require-voice-qa is used.",
    )
    parser.add_argument(
        "--system-prompt",
        choices=["auto", "chat", "voice"],
        default="auto",
        help="Which system prompt to write into each training example.",
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include non-training metadata. Disable for provider uploads that reject extra keys.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
        help="Also write train/validation splits when greater than 0.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def clean_text(value: Any, *, voice: bool = False) -> str:
    text = str(value or "").strip()
    text = MOVIE_MARKER_RE.sub("", text).strip()
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    text = SECRETISH_RE.sub("[SECRET]", text)
    if voice:
        text = text.replace("**", "").replace("__", "")
    return re.sub(r"\s+", " ", text).strip()


def is_voice_row(row: dict[str, Any]) -> bool:
    channel = str(row.get("channel") or "").lower()
    input_type = str(row.get("input_type") or "").lower()
    return input_type == "voice" or channel in {"voice", "webrtc", "websocket"}


def system_prompt_for(row: dict[str, Any], mode: str) -> str:
    if mode == "chat":
        return CHAT_SYSTEM_PROMPT
    if mode == "voice":
        return VOICE_SYSTEM_PROMPT
    return VOICE_SYSTEM_PROMPT if is_voice_row(row) else CHAT_SYSTEM_PROMPT


def score_at_least(value: Any, minimum: float) -> bool:
    return isinstance(value, (int, float)) and value >= minimum


def cutoff_for_days(days: int) -> datetime | None:
    if days <= 0:
        return None
    return datetime.now(timezone.utc) - timedelta(days=days)


def stream_recent_chat_logs(db: Any, cutoff: datetime | None, limit: int) -> list[Any]:
    collection = db.collection(COL_CHAT_LOGS)
    if cutoff is not None:
        query = (
            collection.where(filter=FieldFilter("created_at", ">=", cutoff))
            .order_by("created_at", direction=firestore.Query.DESCENDING)
            .limit(limit)
        )
    else:
        query = collection.order_by(
            "created_at", direction=firestore.Query.DESCENDING
        ).limit(limit)

    try:
        return list(query.stream())
    except Exception as exc:
        if cutoff is None:
            raise
        print(
            f"warning: date-filtered query failed ({type(exc).__name__}); "
            "falling back to latest documents only",
            file=sys.stderr,
        )
        query = collection.order_by(
            "created_at", direction=firestore.Query.DESCENDING
        ).limit(limit)
        return list(query.stream())


def load_good_voice_recordings(
    db: Any,
    cutoff: datetime | None,
    *,
    min_overall: float,
    min_match: float,
    limit: int,
) -> set[str]:
    collection = db.collection(COL_VOICE_AI_EVALUATIONS)
    query = collection.where(filter=FieldFilter("status", "==", "completed")).limit(limit)
    try:
        docs = list(query.stream())
    except Exception as exc:
        print(
            f"warning: voice QA query failed ({type(exc).__name__}); "
            "voice QA gate will reject voice rows",
            file=sys.stderr,
        )
        return set()

    good_ids: set[str] = set()
    for doc in docs:
        row = doc.to_dict() or {}
        created_at = row.get("created_at")
        if cutoff is not None and isinstance(created_at, datetime) and created_at < cutoff:
            continue

        comparison = row.get("transcript_comparison") or {}
        if not score_at_least(row.get("overall_score"), min_overall):
            continue
        if not score_at_least(comparison.get("match_score"), min_match):
            continue

        recording_id = str(row.get("recording_id") or doc.id)
        if recording_id:
            good_ids.add(recording_id)
    return good_ids


def keep_row(
    row: dict[str, Any],
    *,
    allowed_outcomes: set[str],
    min_outcome_confidence: float,
    channel: str,
    input_type: str,
    require_voice_qa: bool,
    good_voice_recordings: set[str],
) -> tuple[bool, str]:
    user_message = clean_text(row.get("user_message"))
    bot_response = clean_text(row.get("bot_response"), voice=is_voice_row(row))
    if not user_message or not bot_response:
        return False, "empty_text"

    if channel != "all" and str(row.get("channel") or "").lower() != channel:
        return False, "channel"

    if input_type != "all" and str(row.get("input_type") or "").lower() != input_type:
        return False, "input_type"

    outcome = str(row.get("outcome") or "")
    if outcome not in allowed_outcomes:
        return False, "outcome"

    confidence = row.get("outcome_confidence")
    if isinstance(confidence, (int, float)) and confidence < min_outcome_confidence:
        return False, "outcome_confidence"

    if row.get("error_stage") or row.get("error_type"):
        return False, "technical_error"

    if require_voice_qa and is_voice_row(row):
        recording_id = str(row.get("recording_id") or "")
        if not recording_id or recording_id not in good_voice_recordings:
            return False, "voice_qa"

    return True, "kept"


def to_training_example(
    doc_id: str,
    row: dict[str, Any],
    *,
    system_prompt_mode: str,
    include_metadata: bool,
) -> dict[str, Any]:
    voice = is_voice_row(row)
    example: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": system_prompt_for(row, system_prompt_mode)},
            {"role": "user", "content": clean_text(row.get("user_message"))},
            {"role": "assistant", "content": clean_text(row.get("bot_response"), voice=voice)},
        ]
    }
    if include_metadata:
        example["metadata"] = {
            "source": "firestore.bot_chat_logs",
            "doc_id": doc_id,
            "session_id": row.get("session_id"),
            "recording_id": row.get("recording_id"),
            "channel": row.get("channel"),
            "input_type": row.get("input_type"),
            "intent": row.get("intent"),
            "outcome": row.get("outcome"),
            "intent_confidence": row.get("intent_confidence"),
            "outcome_confidence": row.get("outcome_confidence"),
        }
    return example


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_splits(output: Path, rows: list[dict[str, Any]], ratio: float, seed: int) -> None:
    if ratio <= 0 or not rows:
        return
    ratio = min(max(ratio, 0.0), 0.5)
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    valid_count = max(1, round(len(shuffled) * ratio)) if len(shuffled) > 1 else 0
    valid_rows = shuffled[:valid_count]
    train_rows = shuffled[valid_count:]
    stem = output.with_suffix("")
    write_jsonl(stem.with_name(stem.name + ".train.jsonl"), train_rows)
    write_jsonl(stem.with_name(stem.name + ".valid.jsonl"), valid_rows)


def main() -> int:
    args = parse_args()
    allowed_outcomes = {
        item.strip()
        for item in args.allowed_outcomes.split(",")
        if item.strip()
    }
    cutoff = cutoff_for_days(args.days)
    db = _get_db()

    good_voice_recordings: set[str] = set()
    if args.require_voice_qa:
        good_voice_recordings = load_good_voice_recordings(
            db,
            cutoff,
            min_overall=args.min_voice_overall_score,
            min_match=args.min_voice_match_score,
            limit=args.limit,
        )

    examples: list[dict[str, Any]] = []
    skip_counts: dict[str, int] = {}
    docs = stream_recent_chat_logs(db, cutoff, args.limit)
    channel = args.channel.lower()
    input_type = args.input_type.lower()

    for doc in docs:
        row = doc.to_dict() or {}
        kept, reason = keep_row(
            row,
            allowed_outcomes=allowed_outcomes,
            min_outcome_confidence=args.min_outcome_confidence,
            channel=channel,
            input_type=input_type,
            require_voice_qa=args.require_voice_qa,
            good_voice_recordings=good_voice_recordings,
        )
        if not kept:
            skip_counts[reason] = skip_counts.get(reason, 0) + 1
            continue
        examples.append(
            to_training_example(
                doc.id,
                row,
                system_prompt_mode=args.system_prompt,
                include_metadata=args.include_metadata,
            )
        )

    output = Path(args.output)
    write_jsonl(output, examples)
    write_splits(output, examples, args.validation_ratio, args.seed)

    manifest = {
        "output": str(output),
        "total_scanned": len(docs),
        "total_exported": len(examples),
        "skip_counts": skip_counts,
        "allowed_outcomes": sorted(allowed_outcomes),
        "require_voice_qa": args.require_voice_qa,
        "good_voice_recording_count": len(good_voice_recordings),
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
