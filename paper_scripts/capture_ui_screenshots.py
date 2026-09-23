#!/usr/bin/env python3
"""Capture Burnwatch prototype screenshots for the paper appendix."""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:5010"
OUT = Path("/Users/macbookprom4/Documents/AI_Detector/paper/figures")
OUT.mkdir(parents=True, exist_ok=True)

SAMPLE_TEXT = (
    "In recent years, artificial intelligence has fundamentally transformed the "
    "landscape of digital content creation. Moreover, large language models can "
    "now generate coherent passages that are difficult to distinguish from human "
    "writing. It is important to note that this capability raises significant "
    "questions for publishers and educators alike. Furthermore, researchers have "
    "proposed a variety of detection methods, including likelihood ratios, rank "
    "statistics, and stylometric features. In conclusion, careful evaluation of "
    "false-alert control remains essential before any deployment decision."
)


def wait_ready(page, timeout_ms: int = 30000) -> None:
    page.wait_for_function(
        """() => {
          const el = document.querySelector('#nav-cal');
          if (!el) return false;
          const t = (el.textContent || '').toLowerCase();
          return t.includes('ready') || t.includes('260') || t.includes('✓');
        }""",
        timeout=timeout_ms,
    )


def dismiss_overlays(page) -> None:
    page.add_init_script(
        """
        try {
          localStorage.setItem('conformal-privacy-choice', 'allowed');
          localStorage.setItem('conformal-help', '0');
        } catch (e) {}
        """
    )


def strip_banner(page) -> None:
    page.evaluate(
        """() => {
          const b = document.querySelector('#privacy-banner');
          if (b) b.remove();
        }"""
    )


def screenshot(page, name: str, full: bool = False) -> None:
    path = OUT / name
    page.screenshot(path=str(path), full_page=full)
    print(f"wrote {path} ({path.stat().st_size} bytes)")


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1440, "height": 900},
            device_scale_factor=2,
            locale="en-US",
        )
        page = context.new_page()
        dismiss_overlays(page)

        page.goto(f"{BASE}/#screen", wait_until="networkidle")
        wait_ready(page)
        page.wait_for_timeout(500)
        strip_banner(page)
        screenshot(page, "ui_screen_landing.png")

        page.locator("#constructions").scroll_into_view_if_needed()
        page.wait_for_timeout(200)
        screenshot(page, "ui_screen_controls.png")

        page.locator("#text-input").fill(SAMPLE_TEXT)
        page.wait_for_timeout(300)
        page.locator("#btn-detect").click()
        page.wait_for_selector("#results", state="visible", timeout=120000)
        page.wait_for_function(
            """() => {
              const el = document.querySelector('#screen-verdict');
              if (!el) return false;
              const t = (el.innerText || '').trim();
              return t.length > 40;
            }""",
            timeout=120000,
        )
        page.wait_for_timeout(800)
        page.locator("#results").scroll_into_view_if_needed()
        page.wait_for_timeout(300)
        screenshot(page, "ui_screen_results.png")
        page.locator("#results").screenshot(path=str(OUT / "ui_results_panels.png"))
        print(f"wrote {OUT / 'ui_results_panels.png'}")

        page.locator("#benchmark").scroll_into_view_if_needed()
        page.wait_for_timeout(200)
        screenshot(page, "ui_benchmark_setup.png")

        status = page.evaluate(
            """async () => {
              const r = await fetch('/api/status');
              return await r.json();
            }"""
        )
        if not status.get("busy"):
            page.locator("#btn-bench").click()
            try:
                page.wait_for_function(
                    """() => {
                      const t = (document.querySelector('#bench-table')?.innerText || '');
                      return t.includes('Construction') && t.split('\\n').length > 3;
                    }""",
                    timeout=180000,
                )
                page.wait_for_timeout(500)
                page.locator("#benchmark").scroll_into_view_if_needed()
                screenshot(page, "ui_benchmark_results.png")
            except Exception as e:
                print(f"benchmark screenshot skipped: {e}")
        else:
            print("busy; skip benchmark run")

        page.goto(f"{BASE}/#how", wait_until="networkidle")
        wait_ready(page)
        strip_banner(page)
        page.wait_for_timeout(400)
        screenshot(page, "ui_how_overview.png")
        page.get_by_text("Calibrate", exact=False).first.scroll_into_view_if_needed()
        page.wait_for_timeout(200)
        screenshot(page, "ui_how_route.png")
        page.locator("#ds-plot").scroll_into_view_if_needed()
        page.wait_for_timeout(300)
        screenshot(page, "ui_how_limits.png")

        page.goto(f"{BASE}/#faq", wait_until="networkidle")
        strip_banner(page)
        page.wait_for_timeout(300)
        screenshot(page, "ui_faq.png")

        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
