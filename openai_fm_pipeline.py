"""Automation pipeline for openai.fm text-to-speech downloads.

The script performs the following sequence:

1. Selects the requested voice preset exactly once.
2. Applies a user supplied instruction once.
3. Iterates over a list of prompts, requesting an audio download for each.
4. Waits until each download is ready and stores it on disk.

Usage example (headless execution with custom prompts file)::

    python openai_fm_pipeline.py \
        --voice "Alloy" \
        --instruction-file instruction.txt \
        --prompts-file prompts.txt \
        --prompt-text-file long_script.txt \
        --output-dir downloads \
        --headless

The prompts file can contain one prompt per line, while ``--prompt-text-file``
accepts an arbitrarily long script which is automatically chunked into pieces
that obey the 999 character UI limit. An instruction may be provided either
inline via ``--instruction`` or through ``--instruction-file``. When both flags
are set, the inline value takes priority. Files are saved using an incrementing
index and an optional prefix. Provide ``--voice-cycle`` to rotate voices per
chunk automatically. Use ``--report-file`` to persist structured metadata about
each download, including the suggested filename from openai.fm.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import re
from typing import Iterable, List, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imported lazily for optional features
    from playwright.async_api import Browser, Page


DEFAULT_URL = "https://www.openai.fm/"
PROMPT_SELECTOR = "#prompt"
INSTRUCTION_SELECTOR = "#input"
VOICE_DESCRIPTIONS_PATH = pathlib.Path(__file__).with_name("voices.json")


def read_lines(path: pathlib.Path) -> List[str]:
    """Return non-empty, stripped lines from ``path``."""

    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def chunk_text(text: str, chunk_size: int) -> List[str]:
    """Split ``text`` into <= ``chunk_size`` character chunks preserving words."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return []

    chunks: List[str] = []
    current_words: List[str] = []
    current_length = 0

    for word in normalized.split(" "):
        if not word:
            continue
        prospective = len(word) if not current_words else current_length + 1 + len(word)
        if current_words and prospective > chunk_size:
            chunks.append(" ".join(current_words))
            current_words = [word]
            current_length = len(word)
        else:
            if current_words:
                current_length += 1 + len(word)
            else:
                current_length = len(word)
            current_words.append(word)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def extend_with_chunking(prompts: List[str], texts: Iterable[str], chunk_size: int) -> None:
    """Extend ``prompts`` with ``texts`` chunked to ``chunk_size``."""

    for text in texts:
        cleaned = text.strip()
        if not cleaned:
            continue
        if len(cleaned) <= chunk_size:
            prompts.append(cleaned)
        else:
            prompts.extend(chunk_text(cleaned, chunk_size))


def load_voice_descriptions(path: pathlib.Path) -> List[dict[str, str]]:
    """Load the voice description metadata from ``path``."""

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Voice description file must contain a list")
    normalized: List[dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Voice description entries must be objects")
        name = str(item.get("name", "")).strip()
        description = str(item.get("description", "")).strip()
        code = str(item.get("code", name)).strip()
        if not name or not description:
            raise ValueError("Each voice entry requires 'name' and 'description'")
        normalized.append({"name": name, "description": description, "code": code})
    return normalized


def coerce_instruction(args: argparse.Namespace) -> str | None:
    """Resolve the instruction text from CLI arguments."""

    if args.instruction is not None:
        return args.instruction
    if args.instruction_file is not None:
        return pathlib.Path(args.instruction_file).read_text(encoding="utf-8").strip() or None
    return None


def coerce_prompts(args: argparse.Namespace) -> Sequence[str]:
    """Load prompts either from CLI values or text files, with chunking support."""

    prompts: List[str] = []
    chunk_size: int = args.chunk_size

    if chunk_size <= 0:
        raise SystemExit("--chunk-size must be greater than zero.")

    if args.prompts:
        extend_with_chunking(prompts, args.prompts, chunk_size)

    if args.prompts_file:
        extend_with_chunking(prompts, read_lines(pathlib.Path(args.prompts_file)), chunk_size)

    if args.prompt_text_file:
        long_text = pathlib.Path(args.prompt_text_file).read_text(encoding="utf-8")
        extend_with_chunking(prompts, [long_text], chunk_size)

    if not prompts:
        raise SystemExit(
            "No prompts supplied. Use --prompts, --prompts-file, or --prompt-text-file."
        )
    return prompts


def resolve_voice_sequence(args: argparse.Namespace, prompt_count: int) -> List[str | None]:
    """Determine which voice to use for each prompt."""

    if prompt_count <= 0:
        return []

    if getattr(args, "voice_cycle", None):
        voices = [voice.strip() for voice in args.voice_cycle if voice and voice.strip()]
        if not voices:
            raise SystemExit("--voice-cycle requires at least one non-empty value.")
        if len(voices) < prompt_count:
            repeats = (prompt_count + len(voices) - 1) // len(voices)
            voices = (voices * repeats)[:prompt_count]
        else:
            voices = voices[:prompt_count]
        return voices

    if args.voice:
        return [args.voice] * prompt_count

    return [None] * prompt_count


def safe_stem(text: str, fallback: str) -> str:
    """Return a filesystem-friendly stem for ``text`` or ``fallback``."""

    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")
    return slug if slug else fallback


async def configure_page(
    page: "Page", voice: str | None, instruction: str | None, timeout: float
) -> None:
    """Apply the initial configuration (voice + instruction)."""

    await page.goto(DEFAULT_URL, wait_until="networkidle")
    await page.wait_for_timeout(2000)

    if voice:
        await page.get_by_role("button", name=voice).click()

    if instruction:
        await page.fill(INSTRUCTION_SELECTOR, instruction)

    await page.wait_for_timeout(500)


async def iterate_prompts(
    page: "Page",
    prompts: Sequence[str],
    output_dir: pathlib.Path,
    prefix: str,
    timeout: float,
    voices: Sequence[str | None],
    initial_voice: str | None,
    report_path: pathlib.Path | None,
) -> None:
    """Process every prompt and persist the resulting download."""

    output_dir.mkdir(parents=True, exist_ok=True)
    download_button = page.get_by_role("button", name="Download")

    current_voice: str | None = initial_voice
    report_entries: List[dict[str, object]] = []
    for index, prompt in enumerate(prompts, start=1):
        voice = voices[index - 1] if index - 1 < len(voices) else None
        if voice and voice != current_voice:
            await page.get_by_role("button", name=voice).click()
            current_voice = voice

        await page.fill(PROMPT_SELECTOR, prompt)

        async with page.expect_download() as download_info:
            await download_button.click()

        download = await download_info.value
        handle = await download_button.element_handle()
        if handle is not None:
            await page.wait_for_function(
                "(el) => !el.hasAttribute('data-disabled')",
                arg=handle,
                timeout=timeout,
            )
        else:
            await page.wait_for_timeout(1000)

        stem = safe_stem(prompt[:32], f"audio_{index:02d}")
        suffix = f"_{safe_stem(voice, voice)}" if voice else ""
        name = (
            f"{prefix}{index:02d}_{stem}{suffix}.mp3"
            if prefix
            else f"{index:02d}_{stem}{suffix}.mp3"
        )
        saved_path = output_dir / name
        await download.save_as(str(saved_path))

        report_entries.append(
            {
                "index": index,
                "voice": voice,
                "prompt": prompt,
                "saved_path": str(saved_path),
                "suggested_filename": download.suggested_filename,
            }
        )

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(report_entries, ensure_ascii=False, indent=2)
        report_path.write_text(f"{payload}\n", encoding="utf-8")


async def run_pipeline(args: argparse.Namespace) -> None:
    prompts = coerce_prompts(args)
    instruction = coerce_instruction(args)
    voices = resolve_voice_sequence(args, len(prompts))
    initial_voice = next((voice for voice in voices if voice), None)
    report_path = pathlib.Path(args.report_file) if args.report_file else None

    from playwright.async_api import async_playwright  # Imported lazily

    async with async_playwright() as playwright:
        browser: "Browser" = await playwright.chromium.launch(
            headless=args.headless,
            slow_mo=args.slowmo,
        )
        context = await browser.new_context(ignore_https_errors=args.ignore_https_errors)
        page: "Page" = await context.new_page()

        await configure_page(page, initial_voice, instruction, args.timeout)
        await iterate_prompts(
            page,
            prompts,
            pathlib.Path(args.output_dir),
            args.prefix,
            args.timeout,
            voices,
            initial_voice,
            report_path,
        )

        await context.close()
        await browser.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Automate voice downloads from openai.fm")
    parser.add_argument("--voice", help="Voice preset to select once at startup.")
    parser.add_argument(
        "--voice-cycle",
        nargs="+",
        help="Sequence of voices to rotate for each prompt (overrides --voice).",
    )
    parser.add_argument("--instruction", help="Instruction text to apply once.")
    parser.add_argument("--instruction-file", help="Path to a file that contains the instruction text.")
    parser.add_argument(
        "--prompts",
        nargs="*",
        help="One or more prompt strings provided directly on the command line.",
    )
    parser.add_argument(
        "--prompts-file",
        help="Path to a file with prompts (one per line).",
    )
    parser.add_argument(
        "--prompt-text-file",
        help="Path to a long text file that should be chunked and processed sequentially.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=999,
        help="Maximum character length for each prompt chunk (default: 999).",
    )
    parser.add_argument(
        "--output-dir",
        default="downloads",
        help="Directory where audio files will be saved (created if missing).",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Optional filename prefix applied before the index (e.g. 'take_').",
    )
    parser.add_argument(
        "--report-file",
        help="Optional path to write a JSON report with prompt/voice metadata.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60000,
        help="Timeout in milliseconds for waiting on the download button to reactivate.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the browser in headless mode.",
    )
    parser.add_argument(
        "--slowmo",
        type=float,
        default=0,
        help="Delay (in ms) between Playwright actions for easier debugging.",
    )
    parser.add_argument(
        "--ignore-https-errors",
        action="store_true",
        help="Ignore HTTPS/TLS errors when loading openai.fm (useful in controlled environments).",
    )
    parser.add_argument(
        "--voices-file",
        default=str(VOICE_DESCRIPTIONS_PATH),
        help="Path to the JSON file describing available voices.",
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List available voices and exit.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.list_voices:
            voices_path = pathlib.Path(args.voices_file)
            voices = load_voice_descriptions(voices_path)
            for entry in voices:
                code = entry["code"]
                label = entry["name"]
                description = entry["description"]
                print(f"{label} ({code})\n  {description}\n")
            return
        asyncio.run(run_pipeline(args))
    except KeyboardInterrupt:
        raise SystemExit("Interrupted by user") from None


if __name__ == "__main__":
    main()
