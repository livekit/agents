#!/usr/bin/env python3
import os
import re
import subprocess
import requests
import uuid
import urllib.parse

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO = os.environ.get("GITHUB_REPOSITORY")
PR_NUMBER = os.environ.get("GITHUB_PR_NUMBER")
BASE_REF = os.environ.get("GITHUB_BASE_REF", "main")
headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}

def get_changed_files():
    subprocess.run(["git", "fetch", "origin", BASE_REF], check=True)
    subprocess.run(["git", "fetch", "origin", f"pull/{PR_NUMBER}/head:pr_head"], check=True)
    result = subprocess.run(
        ["git", "diff", "--name-only", f"origin/{BASE_REF}", "pr_head"],
        capture_output=True, text=True, check=True
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
            if len(parts) > 1 and os.path.isdir(os.path.join("livekit-plugins", parts[1])):
                changes.add(parts[1])
        elif f.startswith(".github/next-release/"):
            changeset_exists = True
    return changes, changeset_exists

def get_pr_details():
    pr_url = f"https://api.github.com/repos/{REPO}/pulls/{PR_NUMBER}"
    r = requests.get(pr_url, headers=headers)
    r.raise_for_status()
    pr_data = r.json()
    title = pr_data.get("title", "Your changes description here.")
    head_ref = pr_data["head"]["ref"]
    head_repo = pr_data["head"]["repo"]["full_name"]
    return title, head_ref, head_repo

def generate_template(changes, description):
    lines = ["---"]
    for change in sorted(changes):
        lines.append(f'"{change}": patch')
    lines.append("---")
    lines.append(f"\n{description}")
    return "\n".join(lines)

def validate_changeset_content(content):
    if not content.startswith("---"):
        return False, "File does not start with '---'."
    lines = content.splitlines()
    if len(lines) < 3:
        return False, "Not enough lines for a valid changeset."
    try:
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
            return False, f"Invalid front matter line: '{line}'. It must be in the format \"package-name\": patch|minor|major."
        pkg = m.group(1)
        if pkg in packages:
            return False, f"Duplicate package entry found: '{pkg}'."
        packages.add(pkg)
    # Check that each plugin (other than livekit-agents) exists.
    for pkg in packages:
        if pkg != "livekit-agents" and not os.path.isdir(os.path.join("livekit-plugins", pkg)):
            return False, f"Plugin '{pkg}' directory does not exist."
    description_lines = lines[second_delim_index + 1:]
    if not any(l.strip() for l in description_lines):
        return False, "Missing change description after front matter."
    return True, ""

def validate_all_changeset_files(diff_files):
    errors, entries, descriptions = [], [], []
    for f in diff_files:
        try:
            # Read the file content from the pr_head branch using git show.
            content = subprocess.run(
                ["git", "show", f"pr_head:{f}"],
                capture_output=True, text=True, check=True
            ).stdout
        except subprocess.CalledProcessError as e:
            errors.append(f"{f}: Failed to read file from git branch 'pr_head'.")
            continue

        is_valid, error = validate_changeset_content(content)
        if not is_valid:
            errors.append(f"{f}: {error}")
        else:
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
                description = "\n".join(lines[second_delim_index + 1:]).strip()
                if description:
                    descriptions.append(description)
    if errors:
        return False, "\n".join(errors), None, None
    entries.sort(key=lambda x: x[0])
    summary_lines = [f"- `{bump}` - `{pkg}`" for _, bump, pkg in entries]
    summary_text = "\n".join(summary_lines)
    description_text = "\n\n".join(descriptions)
    return True, "", summary_text, description_text

def post_or_update_comment(body):
    marker = "<!-- changeset-checker -->"
    list_url = f"https://api.github.com/repos/{REPO}/issues/{PR_NUMBER}/comments"
    r = requests.get(list_url, headers=headers)
    r.raise_for_status()
    comments = r.json()
    comment_id = next((c["id"] for c in comments if marker in c.get("body", "")), None)
    body_with_marker = marker + "\n" + body
    if comment_id:
        update_url = f"https://api.github.com/repos/{REPO}/issues/comments/{comment_id}"
        r = requests.patch(update_url, headers=headers, json={"body": body_with_marker})
    else:
        r = requests.post(list_url, headers=headers, json={"body": body_with_marker})
    r.raise_for_status()

def main():
    diff_files = get_changed_files()
    changes, changeset_exists = parse_changes(diff_files)
    if changes:
        if changeset_exists:
            diff_changeset_files = [f for f in diff_files if f.startswith(".github/next-release/")]
            valid, error_msg, formatted_summary, change_desc = validate_all_changeset_files(diff_changeset_files)
            if not valid:
                comment = ("### :x: Invalid Changeset Format Detected\n\n"
                           "One or more changeset files in this PR have an invalid format. Please ensure they adhere to:\n\n"
                           "- Start with `---` and include a closing `---` on its own line.\n"
                           '- Each package line must be in the format:\n'
                           '  `"package-name": patch|minor|major`\n'
                           "- No duplicate package entries allowed.\n"
                           "- A non-empty change description must follow the front matter.\n\n"
                           f"**Error details:**\n{error_msg}")
            else:
                comment = ("### ✅ Changeset File Detected\n\n"
                           "The following changeset entries were found:\n\n"
                           f"{formatted_summary}\n\n"
                           f"**Change description:**\n{change_desc}")
        else:
            # Get PR details to construct the new file link using the head repo & branch.
            title, head_ref, head_repo = get_pr_details()
            pr_title = f"{title} (#{PR_NUMBER})"
            template = generate_template(changes, pr_title)
            file_name = f"changeset-{uuid.uuid4().hex[:8]}.md"
            base_url = f"https://github.com/{head_repo}/new/{head_ref}"
            params = {"filename": f".github/next-release/{file_name}", "value": template}
            link = base_url + "?" + urllib.parse.urlencode(params)
            message_lines = [
                "### :warning: Changeset Required",
                "",
                "We detected changes in the following package(s) but **no changeset file was found**. Please add one for proper versioning:",
                "",
            ]
            for item in sorted(changes):
                message_lines.append(f"- `{item}`")
            message_lines.extend(["", f"👉 Create a changeset file by [clicking here]({link})."])
            comment = "\n".join(message_lines)
        post_or_update_comment(comment)

if __name__ == "__main__":
    main()
