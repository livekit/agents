"""Draft the summary at the top of a release's notes from the PRs merged since the previous release."""

import json
import re
import subprocess

import click
import openai

MODEL = "gpt-6.1-sol"

SYSTEM_PROMPT = """\
You write the notes at the top of a GitHub release for LiveKit Agents, the Python framework for realtime voice AI agents (the `livekit-agents` package and the `livekit-plugins-*` provider plugins). GitHub appends the full list of merged PRs under your notes, so be selective the way a maintainer is: a reader skims your notes for what to adopt and what to update.

Match the house style of past releases, in this order:
- Up to three `## <Feature name>` sections, one for each new framework capability whose PR body shows the call a user writes to adopt it. A new option or method on one plugin gets no section and no bullet anywhere, since the PR list below carries it. Each has one or two short paragraphs, never bullets, addressed to the developer as "you", on the problem it solves and what it does, then a short Python block showing how to enable or call it, then a "Read more in [our docs](<url>)." line. Write the code block and the link only when a PR body shows that call or links a docs.livekit.io page.
- `## New providers and models`: every new plugin and every model newly supported, including through a dependency upgrade, as "**<Name>**: <what it adds, in one clause>".
- `## Security`: fixes that close a hole in LiveKit code, such as an API key leaked into logs, or that raise a dependency floor past a published CVE or GHSA advisory, each saying what was exposed. Hardening listed in an upstream changelog is not one, and a bump that only touches `uv.lock` ships to no user.
- `## Breaking changes`: only changes after which code or config that worked on the previous release fails or silently does something else, such as a removed or renamed public API or a renamed telemetry attribute, each saying what to update. Raised dependency floors, new defaults and corrected behavior go instead in `## Behavior changes`, stated softly as what the release now does, like "`livekit-agents` now requires `pydantic>=2.5`."
- One closing sentence naming the areas that got more reliable, when the release has fixes.

In every section, give one bullet per provider or feature, merging its models or changes into that bullet, like "**Claude Opus 5.5 and Sonnet 5.5**" or one **Speechmatics** bullet that lists each update. Leave out any section that would be empty, and keep the whole text under 250 words unless a headline feature needs its code block. Describe only what the PR titles and bodies support, with no invented APIs, numbers or links, and output only the markdown."""


def gh(*args: str) -> str:
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout


@click.command()
@click.option("--tag", required=True, help="Tag of the new release, e.g. livekit-agents@1.8.7.")
@click.option("--target", required=True, help="Branch or commit the release is cut from.")
@click.option("--previous-tag", required=True, help="Tag of the previous release on that branch.")
@click.option("--repo", default="livekit/agents", show_default=True)
def main(tag: str, target: str, previous_tag: str, repo: str) -> None:
    generated = json.loads(
        gh(
            "api",
            f"repos/{repo}/releases/generate-notes",
            "-f",
            f"tag_name={tag}",
            "-f",
            f"target_commitish={target}",
            "-f",
            f"previous_tag_name={previous_tag}",
        )
    )["body"]
    # each "What's Changed" line reads "* <title> by @<author> in <PR url>"
    entries = re.findall(r"^\* (.+) by @\S+ in \S+/pull/(\d+)$", generated, re.MULTILINE)

    prs = []
    for title, number in entries:
        if title.startswith("livekit-agents@"):
            continue
        try:
            pr = json.loads(gh("pr", "view", number, "--repo", repo, "--json", "body,labels,files"))
        except subprocess.CalledProcessError as e:
            # a PR can vanish (e.g. its author's account was deleted); the title is still known
            if "Could not resolve to a PullRequest" not in e.stderr:
                raise
            pr = {"body": "", "labels": [], "files": []}
        prs.append(
            {
                "number": int(number),
                "title": title,
                "labels": [label["name"] for label in pr["labels"]],
                "files": [f["path"] for f in pr["files"]],
                "body": pr["body"],
            }
        )

    if not prs:
        raise click.ClickException(f"no PRs between {previous_tag} and {target}")

    response = openai.OpenAI().responses.create(
        model=MODEL,
        reasoning={"effort": "medium"},
        instructions=SYSTEM_PROMPT,
        input=f"Release {tag} (previous: {previous_tag}). Merged PRs:\n\n"
        + json.dumps(prs, indent=1),
    )
    if response.status != "completed":
        raise click.ClickException(f"drafting stopped early: {response.status}")
    click.echo(response.output_text.strip())


if __name__ == "__main__":
    main()
