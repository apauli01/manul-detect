#!/usr/bin/env python3
"""
collector.py

Downloads images by ID from a URL template and saves them to an output folder.

Usage examples:
  python collector.py --url-template "https://example.com/images/{id}.jpg" --outdir images --start 1 --end 100
  python collector.py --url-template "https://example.com/images/{id}.jpg" --outdir images --start 1 --max-misses 50

Features:
- Skip IDs when the image is missing (404) or content-type is not an image
- Increment ID and continue until --end or until max consecutive misses reached
- Resume by skipping already-downloaded files
- Optional concurrency (default 4)

"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Generator

import requests


LOG = logging.getLogger("collector")


def download_image(session: requests.Session, url: str, out_path: Path, timeout: int = 10) -> bool:
    """Download a single image URL to out_path.

    Returns True if downloaded and saved, False if not (404 or non-image or other error).
    """
    try:
        resp = session.get(url, timeout=timeout, stream=True)
    except requests.RequestException as exc:
        LOG.debug("Request failed for %s: %s", url, exc)
        return False

    if resp.status_code == 404:
        LOG.debug("Not found (404): %s", url)
        return False

    if resp.status_code >= 400:
        LOG.warning("HTTP %s for %s", resp.status_code, url)
        return False

    content_type = resp.headers.get("Content-Type", "")
    if not content_type.startswith("image"):
        LOG.debug("URL did not return image for %s: Content-Type=%s", url, content_type)
        return False

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(8192):
                if chunk:
                    f.write(chunk)
    except Exception as exc:
        LOG.warning("Failed to write file %s: %s", out_path, exc)
        return False

    return True


def id_generator(start: int, end: Optional[int]) -> Generator[int, None, None]:
    """Yield IDs from start up to end (inclusive) if end provided; otherwise infinite."""
    i = start
    while True:
        if end is not None and i > end:
            break
        yield i
        i += 1


def run(args: argparse.Namespace) -> int:
    url_template = args.url_template
    outdir = Path(args.outdir)
    start = args.start
    end = args.end
    max_misses = args.max_misses
    concurrency = max(1, args.concurrency)

    session = requests.Session()
    # optional headers
    session.headers.update({"User-Agent": args.user_agent})

    consecutive_misses = 0
    total_downloaded = 0
    total_checked = 0

    LOG.info("Starting collector: start=%s end=%s outdir=%s", start, end, outdir)

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {}

        def submit_for_id(n: int):
            url = url_template.format(id=n)
            filename = f"{n}.jpg"
            out_path = outdir / filename
            if out_path.exists() and not args.overwrite:
                LOG.debug("Already exists, skipping: %s", out_path)
                return None
            fut = ex.submit(download_image, session, url, out_path, args.timeout)
            futures[fut] = (n, url, out_path)
            return fut

        gen = id_generator(start, end)
        # pre-fill queue
        in_flight = 0
        for _ in range(concurrency):
            try:
                n = next(gen)
            except StopIteration:
                break
            submit_for_id(n)
            in_flight += 1

        while futures:
            done, _ = wait_for_any(futures)
            for fut in done:
                n, url, out_path = futures.pop(fut)
                in_flight -= 1
                total_checked += 1
                try:
                    ok = fut.result()
                except Exception as exc:
                    LOG.warning("Download task error for id=%s url=%s: %s", n, url, exc)
                    ok = False

                if ok:
                    LOG.info("Downloaded id=%s -> %s", n, out_path)
                    consecutive_misses = 0
                    total_downloaded += 1
                else:
                    LOG.debug("Missing or invalid for id=%s url=%s", n, url)
                    consecutive_misses += 1

                # decide whether to stop
                if max_misses and consecutive_misses >= max_misses:
                    LOG.info("Reached max consecutive misses (%s). Stopping.", max_misses)
                    # cancel all remaining futures
                    for ff in list(futures.keys()):
                        ff.cancel()
                    return 0

                # submit next id if available
                try:
                    n2 = next(gen)
                except StopIteration:
                    # no more
                    continue
                submit_for_id(n2)
                in_flight += 1

        LOG.info("Finished. checked=%s downloaded=%s", total_checked, total_downloaded)
    return 0


def wait_for_any(futures_dict: dict):
    """Wait until at least one future completes and return (done, not_done) sets based on the keys of futures_dict."""
    from concurrent.futures import wait, FIRST_COMPLETED

    if not futures_dict:
        return set(), set()
    futs = list(futures_dict.keys())
    done, not_done = wait(futs, return_when=FIRST_COMPLETED)
    return done, not_done


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download images by ID from a URL template")
    p.add_argument("--url-template", required=True, help="URL template with {id} placeholder, e.g. https://.../image/{id}.jpg")
    p.add_argument("--outdir", required=True, help="Output directory to save images")
    p.add_argument("--start", type=int, default=1, help="Starting ID (inclusive)")
    p.add_argument("--end", type=int, default=None, help="Ending ID (inclusive). If omitted, runs until max-misses is reached")
    p.add_argument("--max-misses", type=int, default=50, help="Stop after this many consecutive missing IDs when end is not provided")
    p.add_argument("--concurrency", type=int, default=4, help="Number of concurrent download workers")
    p.add_argument("--timeout", type=int, default=15, help="Request timeout in seconds")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    p.add_argument("--user-agent", default="collector/1.0", help="User-Agent header for requests")
    p.add_argument("--log-level", default="INFO", help="Logging level")
    return p.parse_args(argv)


def setup_logging(level: str):
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric, format="%(asctime)s %(levelname)s %(message)s")


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)
    try:
        return run(args)
    except KeyboardInterrupt:
        LOG.info("Interrupted by user")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
