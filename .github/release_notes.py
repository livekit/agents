"""Draft the summary at the top of a release's notes from the PRs merged since the previous release."""

import json
import re
import subprocess

import click
import openai

MODEL = "gpt-5.5"

SYSTEM_PROMPT = """\
You write the summary at the top of a GitHub release for LiveKit Agents, the Python framework for realtime voice AI agents (the `livekit-agents` package and the `livekit-plugins-*` provider plugins). GitHub appends the full list of merged PRs under your summary, so do not restate every PR: pick what a developer upgrading to this version needs to know.

Use only these sections, in this order, and leave out any section that would be empty:
- `### Highlights`: one to four bullets on the changes most users will care about; a PR named here appears in no other section.
- `### Breaking changes`: only changes that make existing code, config or telemetry queries stop working or behave differently in a way users would notice, each with what the user must change.
- `### New`: new plugins, models, options and APIs.
- `### Fixes`: user-visible fixes, each bullet led by a short bold area such as **Voice**, **OpenAI** or **Telemetry**.

Each bullet is one plain sentence ending with the PR reference, like (#1234). Skip CI, tests, dependency bumps, docs and internal refactors unless a user would notice them. Describe only what the PR title and body support; never guess at impact. Output only the markdown, with no heading above the first section."""


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
        except subprocess.CalledProcessError:
            # a PR can vanish (e.g. its author's account was deleted); the title is still known
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
