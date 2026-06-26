#!/usr/bin/env python3
"""Fetch PyPI download stats for all LiveKit packages and show popularity/growth."""

import concurrent.futures
import json
import pathlib
import sys
import urllib.request
from datetime import datetime, timedelta


def _read_pypi_name(plugin_dir: pathlib.Path) -> str:
    """Read the actual PyPI package name from pyproject.toml or setup.py."""
    import re

    pyproject = plugin_dir / "pyproject.toml"
    if pyproject.exists():
        m = re.search(r'^name\s*=\s*"([^"]+)"', pyproject.read_text(), re.MULTILINE)
        if m:
            return m.group(1)
    setup_py = plugin_dir / "setup.py"
    if setup_py.exists():
        m = re.search(r'name\s*=\s*"([^"]+)"', setup_py.read_text())
        if m:
            return m.group(1)
    return plugin_dir.name


def _iter_plugin_names() -> list[str]:
    plugins_root = pathlib.Path(__file__).parent.parent / "livekit-plugins"
    names = []
    for d in sorted(plugins_root.glob("livekit-plugins-*")):
        if d.is_dir():
            names.append(_read_pypi_name(d))
    return names


def _fetch_daily(package: str, retries: int = 3) -> tuple[str, dict[str, int]]:
    """Fetch daily download counts excluding mirrors, with retries."""
    import time

    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                f"https://pypistats.org/api/packages/{package}/overall?mirrors=false",
                headers={"User-Agent": "livekit-stats/1.0"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            daily = {row["date"]: row["downloads"] for row in data.get("data", [])}
            if daily:
                return package, daily
        except Exception:
            pass
        if attempt < retries - 1:
            time.sleep(1)
    return package, {}


def _sum_range(daily: dict[str, int], start: str, end: str) -> int:
    return sum(count for date, count in daily.items() if start <= date <= end)


def _growth_pct(current: int, previous: int) -> float | None:
    if previous <= 0:
        return None
    return (current - previous) / previous * 100


def _fmt_growth(val: float | None) -> str:
    return f"{val:+.0f}%" if val is not None else "—"


def _fmt_downloads(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def _short_name(pkg: str) -> str:
    if pkg.startswith("livekit-plugins-"):
        return "plugins-" + pkg[len("livekit-plugins-"):]
    return pkg


def _fetch_all() -> list[tuple[str, int, int, int, float | None, float | None, float | None]]:
    packages = [
        "livekit",
        "livekit-api",
        "livekit-protocol",
        "livekit-agents",
        "livekit-blingfire",
        "livekit-blockguard",
        "livekit-durable",
    ] + _iter_plugin_names()

    today = datetime.now().date()

    wow_cur_end = (today - timedelta(days=1)).isoformat()
    wow_cur_start = (today - timedelta(days=7)).isoformat()
    wow_prev_start = (today - timedelta(days=14)).isoformat()
    wow_prev_end = (today - timedelta(days=8)).isoformat()

    mom_cur_start = (today - timedelta(days=30)).isoformat()
    mom_prev_start = (today - timedelta(days=60)).isoformat()
    mom_prev_end = (today - timedelta(days=31)).isoformat()

    qoq_cur_start = (today - timedelta(days=90)).isoformat()
    qoq_prev_start = (today - timedelta(days=180)).isoformat()
    qoq_prev_end = (today - timedelta(days=91)).isoformat()

    print(f"Fetching stats for {len(packages)} packages...", file=sys.stderr)
    all_daily: dict[str, dict[str, int]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(_fetch_daily, pkg): pkg for pkg in packages}
        for fut in concurrent.futures.as_completed(futures):
            pkg, daily = fut.result()
            all_daily[pkg] = daily

    results = []
    for pkg in packages:
        daily = all_daily[pkg]
        last_7d = _sum_range(daily, wow_cur_start, wow_cur_end)
        prev_7d = _sum_range(daily, wow_prev_start, wow_prev_end)
        last_30d = _sum_range(daily, mom_cur_start, wow_cur_end)
        prev_30d = _sum_range(daily, mom_prev_start, mom_prev_end)
        last_90d = _sum_range(daily, qoq_cur_start, wow_cur_end)
        prev_90d = _sum_range(daily, qoq_prev_start, qoq_prev_end)
        results.append((
            pkg, last_7d, last_30d, last_90d,
            _growth_pct(last_7d, prev_7d),
            _growth_pct(last_30d, prev_30d),
            _growth_pct(last_90d, prev_90d),
        ))

    results.sort(key=lambda r: r[1], reverse=True)
    return results


def main() -> None:
    results = _fetch_all()
    today = datetime.now().date()

    w = 24  # name column width
    print(f"PyPI Stats (no mirrors) — {today}")
    print(f"{'Package':<{w}} {'7d':>6} {'30d':>6} {'90d':>6}  {'WoW':>5} {'MoM':>5} {'QoQ':>5}")
    print("-" * (w + 38))
    for pkg, last_7d, last_30d, last_90d, wow, mom, qoq in results:
        name = _short_name(pkg)
        if len(name) > w:
            name = name[: w - 1] + "…"
        print(
            f"{name:<{w}} {_fmt_downloads(last_7d):>6} {_fmt_downloads(last_30d):>6}"
            f" {_fmt_downloads(last_90d):>6}  {_fmt_growth(wow):>5} {_fmt_growth(mom):>5}"
            f" {_fmt_growth(qoq):>5}"
        )

    for label, idx, min_prev in [("WoW", 4, 500), ("MoM", 5, 2000), ("QoQ", 6, 5000)]:
        prev_idx = {4: 1, 5: 2, 6: 3}  # map growth idx -> current period volume field
        growers = [
            (r[0], r[idx]) for r in results
            if r[idx] is not None and r[idx] > 0 and r[prev_idx[idx]] >= min_prev
        ]
        growers.sort(key=lambda x: x[1], reverse=True)
        if growers:
            print(f"\nFastest growing ({label}):")
            for pkg, g in growers[:10]:
                print(f"  {_short_name(pkg):<{w}} {g:+.0f}%")


if __name__ == "__main__":
    main()
