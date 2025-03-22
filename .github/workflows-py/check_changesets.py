#!/usr/bin/env python3
import os
import subprocess
import requests
import uuid
import urllib.parse

# Get required environment variables
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO = os.environ.get("GITHUB_REPOSITORY")  # Format: owner/repo
PR_NUMBER = os.environ.get("GITHUB_PR_NUMBER")
BASE_REF = os.environ.get("GITHUB_BASE_REF", "main")
HEAD_REF = os.environ.get("GITHUB_HEAD_REF", "main")

headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}


def get_changed_files():
    # Ensure the base branch is fetched
    subprocess.run(["git", "fetch", "origin", BASE_REF], check=True)
    # List changed files between the fetched base and HEAD
    result = subprocess.run(
        ["git", "diff", "--name-only", f"origin/{BASE_REF}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip().splitlines()


def parse_changes(files):
    changes = set()
    changeset_exists = False
    for f in files:
        if f.startswith("livekit-agents"):
            changes.add("livekit-agents")
        elif f.startswith("livekit-plugins/"):
            parts = f.split("/")
            if len(parts) > 1:
                # Check that the plugin folder exists
                plugin_dir = os.path.join("livekit-plugins", parts[1])
                if os.path.isdir(plugin_dir):
                    changes.add(f"livekit-plugins-{parts[1]}")
        elif f.startswith(".github/next-release/"):
            changeset_exists = True
    return changes, changeset_exists


def get_pr_title():
    # Fetch the PR details to get its title
    pr_url = f"https://api.github.com/repos/{REPO}/pulls/{PR_NUMBER}"
    r = requests.get(pr_url, headers=headers)
    r.raise_for_status()
    pr = r.json()
    return pr.get("title", "Your changes description here.")


def generate_template(changes, description):
    # Build a minimal changeset file content
    lines = ["---"]
    for change in sorted(changes):
        lines.append(f'"{change}": patch')
    lines.append("---")
    lines.append(f"\n{description}")
    return "\n".join(lines)


def get_existing_changeset_summary():
    # If a changeset file already exists in .github/next-release, summarize its front matter.
    summary = []
    path = ".github/next-release"
    if os.path.isdir(path):
        for fname in os.listdir(path):
            fpath = os.path.join(path, fname)
            if os.path.isfile(fpath):
                with open(fpath, "r") as f:
                    content = f.read()
                    if content.startswith("---"):
                        end = content.find("---", 3)
                        if end != -1:
                            summary.append(content[3:end].strip())
    return "\n\n".join(summary)


def post_or_update_comment(body):
    # Use a marker to ensure the comment is unique/updated.
    marker = "<!-- changeset-checker -->"
    list_url = f"https://api.github.com/repos/{REPO}/issues/{PR_NUMBER}/comments"
    r = requests.get(list_url, headers=headers)
    r.raise_for_status()
    comments = r.json()
    comment_id = None
    for comment in comments:
        if marker in comment.get("body", ""):
            comment_id = comment["id"]
            break
    body_with_marker = marker + "\n" + body
    if comment_id:
        # Use the correct update URL endpoint.
        update_url = f"https://api.github.com/repos/{REPO}/issues/comments/{comment_id}"
        r = requests.patch(update_url, headers=headers, json={"body": body_with_marker})
        r.raise_for_status()
    else:
        r = requests.post(list_url, headers=headers, json={"body": body_with_marker})
        r.raise_for_status()


def main():
    files = get_changed_files()
    changes, changeset_exists = parse_changes(files)
    if changes:
        if changeset_exists:
            summary = get_existing_changeset_summary()
            comment = (
                f"Changeset file detected in this PR. It looks good!\n\nRelease summary:\n{summary}"
            )
        else:
            pr_title = get_pr_title()
            template = generate_template(changes, pr_title)
            # Generate a random filename for the changeset file
            file_name = f"changeset-{uuid.uuid4().hex[:8]}.md"
            # Build a link to GitHub’s file creation page on the contributor's branch
            base_url = f"https://github.com/{REPO}/new/{HEAD_REF}"
            params = {"filename": f".github/next-release/{file_name}", "value": template}
            link = base_url + "?" + urllib.parse.urlencode(params)

            # Pretty formatting: separate livekit-agents and plugins, using collapsible details for plugins.
            agents = []
            plugins = []
            for change in sorted(changes):
                if change == "livekit-agents":
                    agents.append(change)
                elif change.startswith("livekit-plugins-"):
                    plugins.append(change)

            message_lines = []
            message_lines.append("Detected changes in relevant packages:\n")
            if agents:
                for a in agents:
                    message_lines.append(f"- `{a}`")
            if plugins:
                message_lines.append("")
                message_lines.append("<details>")
                message_lines.append("  <summary>`livekit-plugins`</summary>")
                message_lines.append("")
                for p in plugins:
                    message_lines.append(f"  - `{p}`")
                message_lines.append("</details>")
            message_lines.append("")
            message_lines.append(
                f"Please add a changeset file for your changes by [clicking here]({link})."
            )
            comment = "\n".join(message_lines)
        post_or_update_comment(comment)


if __name__ == "__main__":
    main()
