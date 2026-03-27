#!/usr/bin/env python3
"""Fetch PyPI download stats for all LiveKit packages and show popularity/growth."""

import concurrent.futures
import json
import pathlib
import sys
import urllib.request
from datetime import datetime, timedelta


def _iter_plugin_names() -> list[str]:
    plugins_root = pathlib.Path(__file__).parent.parent / "livekit-plugins"
    names = []
    for d in sorted(plugins_root.glob("livekit-plugins-*")):
        if d.is_dir():
            names.append(d.name)
    community = plugins_root / "community"
    if community.is_dir():
        for d in sorted(community.glob("livekit-plugins-*")):
            if d.is_dir():
                names.append(d.name)
    return names


def _fetch_json(url: str) -> dict | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "livekit-stats/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def _fetch_daily(package: str) -> tuple[str, dict[str, int]]:
    """Fetch daily download counts excluding mirrors."""
    data = _fetch_json(
        f"https://pypistats.org/api/packages/{package}/overall?mirrors=false"
    )
    daily: dict[str, int] = {}
    if data:
        for row in data.get("data", []):
            daily[row["date"]] = row["downloads"]
    return package, daily


def _sum_range(daily: dict[str, int], start: str, end: str) -> int:
    return sum(count for date, count in daily.items() if start <= date <= end)


def _growth_pct(current: int, previous: int) -> float | None:
    if previous <= 0:
        return None
    return (current - previous) / previous * 100


def _fmt_growth(val: float | None) -> str:
    return f"{val:+.0f}%" if val is not None else "—"


def main() -> None:
    packages = [
        "livekit",
        "livekit-api",
        "livekit-protocol",
        "livekit-agents",
    ] + _iter_plugin_names()

    today = datetime.now().date()

    # WoW: last 7d vs previous 7d
    wow_cur_end = (today - timedelta(days=1)).isoformat()
    wow_cur_start = (today - timedelta(days=7)).isoformat()
    wow_prev_end = (today - timedelta(days=8)).isoformat()
    wow_prev_start = (today - timedelta(days=14)).isoformat()

    # MoM: last 30d vs previous 30d
    mom_cur_end = wow_cur_end
    mom_cur_start = (today - timedelta(days=30)).isoformat()
    mom_prev_end = (today - timedelta(days=31)).isoformat()
    mom_prev_start = (today - timedelta(days=60)).isoformat()

    # QoQ: last 90d vs previous 90d
    qoq_cur_end = wow_cur_end
    qoq_cur_start = (today - timedelta(days=90)).isoformat()
    qoq_prev_end = (today - timedelta(days=91)).isoformat()
    qoq_prev_start = (today - timedelta(days=180)).isoformat()

    # fetch all packages in parallel
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
        wow = _growth_pct(last_7d, prev_7d)

        last_30d = _sum_range(daily, mom_cur_start, mom_cur_end)
        prev_30d = _sum_range(daily, mom_prev_start, mom_prev_end)
        mom = _growth_pct(last_30d, prev_30d)

        last_90d = _sum_range(daily, qoq_cur_start, qoq_cur_end)
        prev_90d = _sum_range(daily, qoq_prev_start, qoq_prev_end)
        qoq = _growth_pct(last_90d, prev_90d)

        results.append((pkg, last_7d, last_30d, last_90d, wow, mom, qoq))

    # sort by last 7d downloads
    results.sort(key=lambda r: r[1], reverse=True)

    hdr = (
        f"{'Package':<40} {'Last 7d':>9} {'Last 30d':>10} {'Last 90d':>10}"
        f" {'WoW':>7} {'MoM':>7} {'QoQ':>7}"
    )
    print(f"\nPyPI Download Stats — without mirrors (as of {today})")
    print(hdr)
    print("-" * len(hdr))
    for pkg, last_7d, last_30d, last_90d, wow, mom, qoq in results:
        print(
            f"{pkg:<40} {last_7d:>9,} {last_30d:>10,} {last_90d:>10,}"
            f" {_fmt_growth(wow):>7} {_fmt_growth(mom):>7} {_fmt_growth(qoq):>7}"
        )

    # top growers by each metric
    for label, idx, min_prev in [("WoW", 4, 500), ("MoM", 5, 2000), ("QoQ", 6, 5000)]:
        # filter: need enough previous-period volume and positive growth
        prev_idx = {4: 2, 5: 3, 6: 4}  # map growth idx -> prev period volume field
        # use last_7d/last_30d/last_90d as proxy for "has enough volume"
        growers = [
            (r[0], r[idx]) for r in results if r[idx] is not None and r[idx] > 0 and r[prev_idx[idx]] >= min_prev
        ]
        growers.sort(key=lambda x: x[1], reverse=True)
        if growers:
            print(f"\nFastest growing ({label}):")
            for pkg, g in growers[:10]:
                print(f"  {pkg:<40} {g:+.0f}%")


if __name__ == "__main__":
    main()
