#!/usr/bin/env python3
import os
import re
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
                # Check that the plugin folder exists; if it is new in this PR it should be present.
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
    title = pr.get("title", "Your changes description here.")
    return f"{title} (#{PR_NUMBER})"


def generate_template(changes, description):
    # Build a minimal changeset file content with one package per line.
    lines = ["---"]
    for change in sorted(changes):
        lines.append(f'"{change}": patch')
    lines.append("---")
    lines.append(f"\n{description}")
    return "\n".join(lines)


def validate_changeset_content(content):
    """
    Validate that the changeset file format is correct.
      - Must start with '---' and include a closing '---'
      - Each non-empty line in the front matter must match: "package-name": patch|minor|major
      - No duplicate package entries.
      - A non-empty change description must follow.
    """
    if not content.startswith("---"):
        return False, "File does not start with '---'."
    lines = content.splitlines()
    if len(lines) < 3:
        return False, "Not enough lines for a valid changeset."
    try:
        # Find the closing '---' for front matter.
        second_delim_index = lines[1:].index("---") + 1
    except ValueError:
        return False, "Missing closing '---' for front matter."
    front_matter = lines[1:second_delim_index]
    if not front_matter:
        return False, "Front matter is empty."
    packages = set()
    for line in front_matter:
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^"([^"]+)":\s*(patch|minor|major)$', line)
        if not m:
            return (
                False,
                f"Invalid front matter line: '{line}'. It must be in the format \"package-name\": patch|minor|major.",
            )
        pkg = m.group(1)
        if pkg in packages:
            return False, f"Duplicate package entry found: '{pkg}'."
        packages.add(pkg)
    # Ensure there is a non-empty change description after front matter.
    description_lines = lines[second_delim_index + 1 :]
    if not any(l.strip() for l in description_lines):
        return False, "Missing change description after front matter."
    return True, ""


def validate_all_changeset_files():
    """
    Iterates over each file in .github/next-release, validates its content,
    and collects formatted summaries and release descriptions.
    If any file is invalid, returns an error.
    """
    errors = []
    entries = []  # Will hold tuples (order, bump, pkg)
    descriptions = []
    path = ".github/next-release"
    if os.path.isdir(path):
        for fname in os.listdir(path):
            fpath = os.path.join(path, fname)
            if os.path.isfile(fpath):
                with open(fpath, "r") as f:
                    content = f.read()
                    is_valid, error = validate_changeset_content(content)
                    if not is_valid:
                        errors.append(f"{fname}: {error}")
                    else:
                        # Split content into front matter and description.
                        lines = content.splitlines()
                        try:
                            second_delim_index = lines[1:].index("---") + 1
                        except ValueError:
                            second_delim_index = None
                        if second_delim_index is not None:
                            front_matter = lines[1:second_delim_index]
                            bump_order = {"major": 0, "minor": 1, "patch": 2}
                            for line in front_matter:
                                m = re.match(r'^"([^"]+)":\s*(patch|minor|major)$', line.strip())
                                if m:
                                    pkg = m.group(1)
                                    bump = m.group(2)
                                    entries.append((bump_order[bump], bump, pkg))
                            # Capture the description (all lines after front matter).
                            description = "\n".join(lines[second_delim_index + 1 :]).strip()
                            if description:
                                descriptions.append(description)
    if errors:
        return False, "\n".join(errors), None, None
    # Sort entries: major first, then minor, then patch.
    entries.sort(key=lambda x: x[0])
    summary_lines = [f"- `{bump}` - `{pkg}`" for _, bump, pkg in entries]
    summary_text = "\n".join(summary_lines)
    description_text = "\n\n".join(descriptions)
    return True, "", summary_text, description_text


def post_or_update_comment(body):
    # Use a marker so that we update the same comment.
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
            valid, error_msg, formatted_summary, change_desc = validate_all_changeset_files()
            if not valid:
                comment = (
                    "### :x: Invalid Changeset Format Detected\n\n"
                    "One or more changeset files in this PR have an invalid format. Please ensure the file(s) adhere to the following:\n\n"
                    "- Start with `---` and include a closing `---` on its own line.\n"
                    "- Each package line must be on its own line in the format:\n"
                    '  `"package-name": patch|minor|major`\n'
                    "- No duplicate package entries are allowed.\n"
                    "- A non-empty change description must follow the front matter.\n\n"
                    f"**Error details:**\n{error_msg}"
                )
            else:
                comment = (
                    "### ✅ Changeset File Detected\n\n"
                    "The following changeset entries were found:\n\n"
                    f"{formatted_summary}\n\n"
                    f"**Release description:**\n{change_desc}"
                )
        else:
            pr_title = get_pr_title()
            template = generate_template(changes, pr_title)
            # Generate a random filename for the changeset file.
            file_name = f"changeset-{uuid.uuid4().hex[:8]}.md"
            # Build a link to GitHub’s file creation page on the contributor's branch.
            base_url = f"https://github.com/{REPO}/new/{HEAD_REF}"
            params = {"filename": f".github/next-release/{file_name}", "value": template}
            link = base_url + "?" + urllib.parse.urlencode(params)
            message_lines = []
            message_lines.append("### :warning: Changeset Required")
            message_lines.append("")
            message_lines.append(
                "We detected changes in the following package(s) but **no changeset file was found**. Please add one to ensure proper versioning:"
            )
            message_lines.append("")
            for item in sorted(changes):
                message_lines.append(f"- `{item}`")
            message_lines.append("")
            message_lines.append(
                f"👉 Please create a changeset file for your changes by [clicking here]({link})."
            )
            comment = "\n".join(message_lines)
        post_or_update_comment(comment)


if __name__ == "__main__":
    main()
