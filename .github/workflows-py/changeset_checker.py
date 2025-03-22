#!/usr/bin/env python3
import os
import re
import random
import string
import subprocess
from pathlib import Path
import requests

# GitHub API access
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_OWNER = os.environ.get("REPO_OWNER")
REPO_NAME = os.environ.get("REPO_NAME")
PR_NUMBER = os.environ.get("PR_NUMBER")
BASE_REF = os.environ.get("BASE_REF", "main")
HEAD_REF = os.environ.get("HEAD_REF")
REPO_URL = os.environ.get("REPO_URL")

# API headers
HEADERS = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}


def get_changed_files():
    """Get files changed in the PR compared to base branch."""
    try:
        output = subprocess.check_output(
            ["git", "diff", "--name-only", f"origin/{BASE_REF}...HEAD"], text=True
        )
        return output.strip().split("\n") if output.strip() else []
    except subprocess.CalledProcessError:
        print(f"Warning: Could not get changed files. Using fallback method.")
        api_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{PR_NUMBER}/files"
        response = requests.get(api_url, headers=HEADERS)
        response.raise_for_status()
        return [file["filename"] for file in response.json()]


def analyze_changes(files):
    """Analyze which packages changed."""
    agents_changed = False
    plugins_changed = set()

    for file in files:
        if file.startswith("livekit-agents/"):
            agents_changed = True
        elif file.startswith("livekit-plugins/"):
            match = re.match(r"livekit-plugins/([^/]+)/", file)
            if match:
                plugins_changed.add(match.group(1))

    return agents_changed, list(plugins_changed)


def check_existing_changesets():
    """Check if changeset files already exist."""
    next_release_dir = Path(".github/next-release")
    next_release_dir.mkdir(parents=True, exist_ok=True)

    changeset_files = list(next_release_dir.glob("*.md"))
    if not changeset_files:
        return False, {}

    # Parse existing changesets
    version_changes = {"patch": [], "minor": [], "major": []}

    for file in changeset_files:
        content = file.read_text()
        for line in content.split("\n"):
            match = re.search(r'"([^"]+)":\s*(patch|minor|major)', line)
            if match:
                package, change_type = match.groups()
                version_changes[change_type].append(package)

    return True, version_changes


def generate_changeset_template(agents_changed, plugins_changed):
    """Generate a changeset template file."""
    next_release_dir = Path(".github/next-release")
    next_release_dir.mkdir(parents=True, exist_ok=True)

    # Generate random filename
    random_id = "".join(random.choices(string.hexdigits.lower(), k=12))
    changeset_path = next_release_dir / f"{random_id}.md"

    # Create changeset content
    content = ["---"]

    if agents_changed:
        content.append('"livekit-agents": patch')

    for plugin in plugins_changed:
        content.append(f'"livekit-plugins-{plugin}": patch')

    content.extend(["---", "<!-- Add your changeset description here -->"])

    # Write to file
    changeset_path.write_text("\n".join(content))

    return random_id, "\n".join(content)


def find_bot_comment():
    """Find existing bot comment on the PR."""
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/issues/{PR_NUMBER}/comments"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()

    for comment in response.json():
        body = comment.get("body", "")
        if "## Changeset Required 📝" in body or "## Changeset Detected ✅" in body:
            return comment["id"]

    return None


def update_or_create_comment(body, comment_id=None):
    """Update existing comment or create a new one."""
    if comment_id:
        url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/issues/comments/{comment_id}"
        response = requests.patch(url, headers=HEADERS, json={"body": body})
    else:
        url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/issues/{PR_NUMBER}/comments"
        response = requests.post(url, headers=HEADERS, json={"body": body})

    response.raise_for_status()
    return response.json()


def main():
    print("Checking for package changes...")

    # Get and analyze changed files
    changed_files = get_changed_files()
    if not changed_files:
        print("No files changed or couldn't detect changes.")
        return

    agents_changed, plugins_changed = analyze_changes(changed_files)

    # Check if any relevant packages changed
    if not agents_changed and not plugins_changed:
        print("No relevant package changes detected.")
        return

    print(
        f"Found changes in: {'livekit-agents' if agents_changed else ''} {' '.join(f'livekit-plugins-{p}' for p in plugins_changed)}"
    )

    # Check for existing changesets
    changesets_exist, version_changes = check_existing_changesets()

    # Find existing bot comment
    comment_id = find_bot_comment()

    if changesets_exist:
        print("Found existing changeset files")
        # Create summary for existing changesets
        summary = []
        for change_type, packages in version_changes.items():
            if packages:
                summary.append(f"- {change_type.capitalize()}: {', '.join(packages)}")

        # Prepare "looks good" comment
        body = "## Changeset Detected ✅\n\n"
        body += "Great! This pull request includes a changeset file. Here's a summary of the version changes:\n\n"
        body += "\n".join(summary) + "\n\n"
        body += "These changes will be included in the next release."
    else:
        print("No changeset files found, generating template")
        # Generate changeset template
        random_id, template = generate_changeset_template(agents_changed, plugins_changed)

        # Prepare package list
        package_changes = []
        if agents_changed:
            package_changes.append("livekit-agents")
        for plugin in plugins_changed:
            package_changes.append(f"livekit-plugins-{plugin}")

        # Create file URL for quick creation
        encoded_template = template.replace("\n", "%0A")
        file_url = f"{REPO_URL}/new/{HEAD_REF}?filename=.github/next-release/{random_id}.md&value={encoded_template}"

        # Prepare "needs changeset" comment
        body = "## Changeset Required 📝\n\n"
        body += "Changes detected in:\n"
        body += "\n".join([f"- `{pkg}`" for pkg in package_changes]) + "\n\n"
        body += "Please add a changeset file describing your changes.\n\n"
        body += f"[👉 Click here to create a changeset file]({file_url})\n\n"
        body += "Your changeset file should look like this:\n\n"
        body += f"```markdown\n{template}\n```\n\n"
        body += "Replace the comment with a brief description of your changes."

    # Update or create comment
    if comment_id:
        print(f"Updating existing comment {comment_id}")
    else:
        print("Creating new comment")
    update_or_create_comment(body, comment_id)
    print("Done!")


if __name__ == "__main__":
    main()
