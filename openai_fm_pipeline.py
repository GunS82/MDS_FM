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
        --output-dir downloads \
        --headless

The prompts file should contain one prompt per line. Blank lines are ignored.
An instruction may be provided either inline via ``--instruction`` or through
``--instruction-file``. When both flags are set, the inline value takes
priority. Files are saved using an incrementing index and an optional prefix.
"""

from __future__ import annotations

import argparse
import asyncio
import pathlib
import re
from typing import List, Sequence

from playwright.async_api import Browser, Page, async_playwright


DEFAULT_URL = "https://www.openai.fm/"
PROMPT_SELECTOR = "#prompt"
INSTRUCTION_SELECTOR = "#input"


def read_lines(path: pathlib.Path) -> List[str]:
    """Return non-empty, stripped lines from ``path``."""

    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def coerce_instruction(args: argparse.Namespace) -> str | None:
    """Resolve the instruction text from CLI arguments."""

    if args.instruction is not None:
        return args.instruction
    if args.instruction_file is not None:
        return pathlib.Path(args.instruction_file).read_text(encoding="utf-8").strip() or None
    return None


def coerce_prompts(args: argparse.Namespace) -> Sequence[str]:
    """Load prompts either from CLI values or a text file."""

    prompts: List[str] = []
    if args.prompts:
        prompts.extend([value.strip() for value in args.prompts if value.strip()])
    if args.prompts_file:
        prompts.extend(read_lines(pathlib.Path(args.prompts_file)))

    if not prompts:
        raise SystemExit("No prompts supplied. Use --prompts or --prompts-file.")
    return prompts


def safe_stem(text: str, fallback: str) -> str:
    """Return a filesystem-friendly stem for ``text`` or ``fallback``."""

    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")
    return slug if slug else fallback


async def configure_page(page: Page, voice: str | None, instruction: str | None, timeout: float) -> None:
    """Apply the initial configuration (voice + instruction)."""

    await page.goto(DEFAULT_URL, wait_until="networkidle")
    await page.wait_for_timeout(2000)

    if voice:
        await page.get_by_role("button", name=voice).click()

    if instruction:
        await page.fill(INSTRUCTION_SELECTOR, instruction)

    await page.wait_for_timeout(500)


async def iterate_prompts(
    page: Page,
    prompts: Sequence[str],
    output_dir: pathlib.Path,
    prefix: str,
    timeout: float,
) -> None:
    """Process every prompt and persist the resulting download."""

    output_dir.mkdir(parents=True, exist_ok=True)
    download_button = page.get_by_role("button", name="Download")

    for index, prompt in enumerate(prompts, start=1):
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
        name = f"{prefix}{index:02d}_{stem}.mp3" if prefix else f"{index:02d}_{stem}.mp3"
        await download.save_as(str(output_dir / name))


async def run_pipeline(args: argparse.Namespace) -> None:
    prompts = coerce_prompts(args)
    instruction = coerce_instruction(args)

    async with async_playwright() as playwright:
        browser: Browser = await playwright.chromium.launch(
            headless=args.headless,
            slow_mo=args.slowmo,
        )
        context = await browser.new_context(ignore_https_errors=args.ignore_https_errors)
        page: Page = await context.new_page()

        await configure_page(page, args.voice, instruction, args.timeout)
        await iterate_prompts(page, prompts, pathlib.Path(args.output_dir), args.prefix, args.timeout)

        await context.close()
        await browser.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Automate voice downloads from openai.fm")
    parser.add_argument("--voice", help="Voice preset to select once at startup.")
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
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        asyncio.run(run_pipeline(args))
    except KeyboardInterrupt:
        raise SystemExit("Interrupted by user") from None


if __name__ == "__main__":
    main()
