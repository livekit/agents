from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

OUT_DIR = Path(__file__).parent

INK = "#1C2329"
MUTED = "#555C61"
FAINT = "#858A8D"
BORDER = "#D8D4CA"
PAPER = "#F3F1EB"
CARD = "#FFFFFF"
SIGNAL = "#3157C8"
SIGNAL_TINT = "#E9EEFC"
HUMAN = "#28705A"
HUMAN_TINT = "#E6F1EC"
UNAVAILABLE = "#A94335"
UNAVAILABLE_TINT = "#F6E9E5"


DEFS = f"""
  <defs>
    <symbol id="icon-circle-question" viewBox="0 0 24 24">
      <circle cx="12" cy="12" r="10"/>
      <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>
      <path d="M12 17h.01"/>
    </symbol>
    <symbol id="icon-bot" viewBox="0 0 24 24">
      <path d="M12 8V4H8"/>
      <rect width="16" height="12" x="4" y="8" rx="2"/>
      <path d="M2 14h2M20 14h2M15 13v2M9 13v2"/>
    </symbol>
    <symbol id="icon-mailbox" viewBox="0 0 24 24">
      <path d="M22 17a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V9.5C2 7 4 5 6.5 5H18c2.2 0 4 1.8 4 4v8Z"/>
      <polyline points="15,9 18,9 18,11"/>
      <path d="M6.5 5C9 5 11 7 11 9.5V17a2 2 0 0 1-2 2"/>
      <line x1="6" x2="7" y1="10" y2="10"/>
    </symbol>
    <symbol id="icon-grid-3x3" viewBox="0 0 24 24">
      <rect width="18" height="18" x="3" y="3" rx="2"/>
      <path d="M3 9h18M3 15h18M9 3v18M15 3v18"/>
    </symbol>
    <symbol id="icon-user-round" viewBox="0 0 24 24">
      <circle cx="12" cy="8" r="5"/>
      <path d="M20 21a8 8 0 0 0-16 0"/>
    </symbol>
    <symbol id="icon-ban" viewBox="0 0 24 24">
      <circle cx="12" cy="12" r="10"/>
      <path d="M4.929 4.929 19.07 19.071"/>
    </symbol>
    <symbol id="icon-waveform" viewBox="0 0 24 24">
      <path d="M2 10v4M6 6v12M10 3v18M14 8v8M18 5v14M22 10v4"/>
    </symbol>
    <symbol id="icon-check" viewBox="0 0 24 24">
      <path d="M20 6 9 17l-5-5"/>
    </symbol>

    <marker id="arrow-ink" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="10" markerHeight="10" markerUnits="userSpaceOnUse" orient="auto">
      <path d="M1 1 9 5 1 9Z" fill="{INK}"/>
    </marker>
    <marker id="arrow-signal" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="10" markerHeight="10" markerUnits="userSpaceOnUse" orient="auto">
      <path d="M1 1 9 5 1 9Z" fill="{SIGNAL}"/>
    </marker>
    <marker id="arrow-human" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="10" markerHeight="10" markerUnits="userSpaceOnUse" orient="auto">
      <path d="M1 1 9 5 1 9Z" fill="{HUMAN}"/>
    </marker>
    <marker id="arrow-unavailable" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="10" markerHeight="10" markerUnits="userSpaceOnUse" orient="auto">
      <path d="M1 1 9 5 1 9Z" fill="{UNAVAILABLE}"/>
    </marker>

    <style>
      text {{ font-family: "IBM Plex Sans", "Helvetica Neue", Helvetica, Arial, sans-serif; }}
      .display {{ font-size: 48px; font-weight: 600; letter-spacing: -0.035em; fill: {INK}; }}
      .subtitle {{ font-size: 15px; fill: {MUTED}; }}
      .kicker {{ font-size: 11px; font-weight: 700; letter-spacing: 0.16em; fill: {MUTED}; }}
      .micro {{ font-size: 10.5px; font-weight: 600; letter-spacing: 0.08em; fill: {FAINT}; }}
      .title {{ font-size: 18px; font-weight: 600; fill: {INK}; }}
      .copy {{ font-size: 12.5px; fill: {MUTED}; }}
      .code {{ font-size: 11px; font-weight: 600; letter-spacing: 0.01em; fill: {MUTED}; }}
      .icon {{ fill: none; stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round; }}
      .wire {{ fill: none; stroke-linecap: round; stroke-linejoin: round; }}
      .ink {{ stroke: {INK}; stroke-width: 1.8; }}
      .signal {{ stroke: {SIGNAL}; stroke-width: 2; }}
      .human {{ stroke: {HUMAN}; stroke-width: 2; }}
      .unavailable {{ stroke: {UNAVAILABLE}; stroke-width: 2; }}
      .stage {{ stroke: #77756E; stroke-width: 1.5; }}
      .repeat {{ stroke: {SIGNAL}; stroke-width: 1.7; stroke-dasharray: 5 7; }}
    </style>
  </defs>
"""


def icon(name: str, x: int, y: int, color: str, size: int = 24) -> str:
    return (
        f'<use href="#icon-{name}" x="{x}" y="{y}" width="{size}" height="{size}" '
        f'class="icon" stroke="{color}"/>'
    )


def card(
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
    copy: str,
    *,
    number: str | None = None,
    icon_name: str | None = None,
    stroke: str = BORDER,
    fill: str = CARD,
    accent: str = SIGNAL,
    title_fill: str = INK,
) -> str:
    label_x = x + 24
    marker = ""
    if number is not None:
        marker = (
            f'<circle cx="{x + 30}" cy="{y + height // 2}" r="15" fill="{PAPER}" '
            f'stroke="{stroke}"/><text x="{x + 30}" y="{y + height // 2 + 4}" '
            f'text-anchor="middle" class="code" fill="{accent}">{escape(number)}</text>'
        )
        label_x = x + 58
    elif icon_name is not None:
        marker = icon(icon_name, x + 20, y + height // 2 - 12, accent)
        label_x = x + 58

    copy_markup = ""
    if copy:
        copy_markup = (
            f'<text x="{label_x}" y="{y + height // 2 + 23}" class="copy">{escape(copy)}</text>'
        )

    return f"""
    <g>
      <rect x="{x}" y="{y}" width="{width}" height="{height}" rx="12" fill="{fill}" stroke="{stroke}"/>
      {marker}
      <text x="{label_x}" y="{y + height // 2 - 1}" class="title" fill="{title_fill}">{escape(title)}</text>
      {copy_markup}
    </g>
    """


def outcome(
    x: int,
    y: int,
    width: int,
    title: str,
    copy: str,
    *,
    icon_name: str,
    color: str,
    fill: str,
) -> str:
    return card(
        x,
        y,
        width,
        100,
        title,
        copy,
        icon_name=icon_name,
        stroke=color,
        fill=fill,
        accent=color,
        title_fill=color,
    )


def path(d: str, cls: str = "stage", marker: str | None = None) -> str:
    marker_markup = f' marker-end="url(#arrow-{marker})"' if marker else ""
    return f'<path d="{d}" class="wire {cls}"{marker_markup}/>'


def label(x: int, y: int, text: str, *, anchor: str = "start", color: str = MUTED) -> str:
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" class="code" '
        f'fill="{color}">{escape(text)}</text>'
    )


def diagram(
    *,
    filename: str,
    title: str,
    subtitle: str,
    badge: str,
    description: str,
    body: str,
    height: int = 620,
) -> None:
    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="{height}" viewBox="0 0 1600 {height}"
  role="img" aria-labelledby="title description" shape-rendering="geometricPrecision">
  <title id="title">{escape(title)}</title>
  <desc id="description">{escape(description)}</desc>
  <metadata id="lucide-license">
    Icons are derived from Lucide Icons and Contributors.
    ISC License. Copyright (c) 2026 Lucide Icons and Contributors.
    Permission to use, copy, modify, and/or distribute this software for any purpose with or
    without fee is hereby granted, provided that the above copyright notice and this permission
    notice appear in all copies.
  </metadata>
{DEFS}
  <rect width="1600" height="{height}" fill="{PAPER}"/>
  <text x="64" y="85" class="display">{escape(title)}</text>
  <text x="64" y="117" class="subtitle">{escape(subtitle)}</text>
  <g transform="translate(1450 64)">
    <rect x="0" y="-18" width="86" height="28" rx="14" fill="none" stroke="#ACA79C"/>
    <text x="43" y="1" text-anchor="middle" class="micro" fill="{MUTED}">{escape(badge)}</text>
  </g>
  <line x1="64" y1="140" x2="1536" y2="140" stroke="{BORDER}"/>
{body}
</svg>
"""
    (OUT_DIR / filename).write_text(svg, encoding="utf-8")


def make_uncertain() -> None:
    body = """
  <g>
    <path d="M334 358H414" class="wire signal" marker-end="url(#arrow-signal)"/>
    <path d="M714 358H782" class="wire signal" marker-end="url(#arrow-signal)"/>
    <path d="M824 358H868M868 226V622" class="wire stage"/>
    <path d="M868 226H920" class="wire signal" marker-end="url(#arrow-signal)"/>
    <path d="M868 358H920" class="wire stage" marker-end="url(#arrow-ink)"/>
    <path d="M868 490H920" class="wire human" marker-end="url(#arrow-human)"/>
    <path d="M868 622H920" class="wire unavailable" marker-end="url(#arrow-unavailable)"/>
    <path d="M803 337L824 358 803 379 782 358Z" fill="#FBFAF7" stroke="#1C2329" stroke-width="1.5"/>
    <text x="374" y="294" text-anchor="middle" class="micro" fill="#3157C8">COMMITTED TURN</text>
  </g>
"""
    body += card(
        64,
        308,
        270,
        100,
        "Uncertain",
        "No concrete stage yet",
        icon_name="circle-question",
        accent="#77756E",
    )
    body += card(414, 308, 300, 100, "Client-side EOT", "Classify; one ordered result", number="1")
    body += card(
        920,
        176,
        340,
        100,
        "Enter machine stage",
        "Screening, voicemail, or IVR",
        number="2",
        stroke=SIGNAL,
        fill=SIGNAL_TINT,
    )
    body += card(
        920,
        308,
        340,
        100,
        "Stay uncertain",
        "Release the normal Agent response",
        number="2",
        stroke="#ACA79C",
        fill="#FBFAF7",
        accent="#77756E",
    )
    body += outcome(
        920,
        440,
        340,
        "Human",
        "Release the held response",
        icon_name="user-round",
        color=HUMAN,
        fill=HUMAN_TINT,
    )
    body += outcome(
        920,
        572,
        340,
        "Unavailable",
        "Cancel the held response",
        icon_name="ban",
        color=UNAVAILABLE,
        fill=UNAVAILABLE_TINT,
    )
    body += """
  <line x1="64" y1="704" x2="1536" y2="704" stroke="#D8D4CA"/>
  <text x="64" y="736" class="kicker">STAGE RULE</text>
  <text x="184" y="736" class="copy">Normal conversation may continue while AMD waits for a concrete category.</text>
"""
    diagram(
        filename="amd-stage-uncertain.svg",
        title="Uncertain",
        subtitle="The normal Agent path remains available while AMD resolves the remote party.",
        badge="ACTIVE",
        description="Uncertain-stage processing from a committed remote turn to a machine stage, human, unavailable, or another uncertain turn.",
        body=body,
        height=768,
    )


def make_screening() -> None:
    body = """
  <path d="M320 352H368" class="wire signal" marker-end="url(#arrow-signal)"/>
  <path d="M624 352H672" class="wire signal" marker-end="url(#arrow-signal)"/>
  <path d="M928 352H976" class="wire signal" marker-end="url(#arrow-signal)"/>
  <path d="M1232 352H1280" class="wire signal" marker-end="url(#arrow-signal)"/>
  <path d="M1408 402V470Q1408 482 1396 482H204Q192 482 192 470V402" class="wire repeat" marker-end="url(#arrow-signal)"/>
  <g transform="translate(824 482)">
    <rect x="-132" y="-13" width="264" height="26" rx="13" fill="#F3F1EB"/>
    <text x="0" y="4" text-anchor="middle" class="micro" fill="#3157C8">NEXT COMMITTED SCREENING TURN</text>
  </g>

  <path d="M496 302V290Q496 278 508 278H1424" class="wire stage"/>
  <path d="M944 278V254" class="wire human" marker-end="url(#arrow-human)"/>
  <path d="M1184 278V254" class="wire signal" marker-end="url(#arrow-signal)"/>
  <path d="M1424 278V254" class="wire unavailable" marker-end="url(#arrow-unavailable)"/>
"""
    body += card(
        64, 302, 256, 100, "Screening prompt", "Remote proxy speaks", icon_name="bot", accent=SIGNAL
    )
    body += card(368, 302, 256, 100, "Client-side EOT", "Classify; emit amd_prediction", number="1")
    body += card(
        672, 302, 256, 100, "Inject instructions", "Retain one AMD control item", number="2"
    )
    body += card(976, 302, 256, 100, "Generate one reply", "Same Agent, normal voice", number="3")
    body += card(1280, 302, 256, 100, "Wait", "Next prompt", number="4")
    body += outcome(
        832,
        154,
        224,
        "Human",
        "Person answers",
        icon_name="user-round",
        color=HUMAN,
        fill=HUMAN_TINT,
    )
    body += card(
        1072,
        154,
        224,
        100,
        "Voicemail",
        "Sent to message-taking",
        icon_name="mailbox",
        stroke=SIGNAL,
        fill=SIGNAL_TINT,
        accent=SIGNAL,
    )
    body += outcome(
        1312,
        154,
        224,
        "Unavailable",
        "Rejected",
        icon_name="ban",
        color=UNAVAILABLE,
        fill=UNAVAILABLE_TINT,
    )
    body += """
  <line x1="64" y1="526" x2="1536" y2="526" stroke="#D8D4CA"/>
  <text x="64" y="558" class="kicker">STAGE RULE</text>
  <text x="184" y="558" class="copy">Every screening turn is classified; there is no direct Screening → IVR edge.</text>
"""
    diagram(
        filename="amd-stage-screening.svg",
        title="Screening",
        subtitle="Answer the platform’s screening proxy until it connects, rejects, or routes to voicemail.",
        badge="ACTIVE",
        description="Screening-stage turn loop with exits to a human, voicemail, or unavailable.",
        body=body,
    )


def make_voicemail() -> None:
    body = """
  <text x="64" y="204" class="kicker">MESSAGE PLAYBACK</text>
  <path d="M320 286H368" class="wire signal" marker-end="url(#arrow-signal)"/>
  <path d="M624 286H672" class="wire signal" marker-end="url(#arrow-signal)"/>
  <path d="M928 286H976" class="wire signal" marker-end="url(#arrow-signal)"/>
  <path d="M1232 286H1280" class="wire signal" marker-end="url(#arrow-signal)"/>

  <text x="64" y="402" class="kicker">INDEPENDENT REPLY GUARD</text>
  <path d="M334 496H414" class="wire signal" marker-end="url(#arrow-signal)"/>
  <path d="M714 496H782" class="wire signal" marker-end="url(#arrow-signal)"/>
  <path d="M803 475L824 496 803 517 782 496Z" fill="#FBFAF7" stroke="#1C2329" stroke-width="1.5"/>
  <path d="M824 496H868M868 432V560" class="wire stage"/>
  <path d="M868 432H920" class="wire signal" marker-end="url(#arrow-signal)"/>
  <path d="M868 560H920" class="wire stage" marker-end="url(#arrow-ink)"/>
"""
    body += card(
        64,
        236,
        256,
        100,
        "Voicemail detected",
        "Enter machine-vm",
        icon_name="mailbox",
        accent=SIGNAL,
    )
    body += card(368, 236, 256, 100, "Generate once", "Inject voicemail instructions", number="1")
    body += card(672, 236, 256, 100, "Play the message", "Session handles interruption", number="2")
    body += card(976, 236, 256, 100, "Playback ends", "Flag full playback only", number="3")
    body += card(1280, 236, 256, 100, "Keep listening", "AMD stays active", number="4")
    body += card(
        64,
        446,
        270,
        100,
        "Client-side EOT",
        "During or after playback",
        number="1",
    )
    body += card(414, 446, 300, 100, "Classify the turn", "Guard the next reply", number="2")
    body += card(
        920,
        382,
        340,
        100,
        "Same stage",
        "Suppress a new voicemail",
        stroke=SIGNAL,
        fill=SIGNAL_TINT,
        icon_name="check",
    )
    body += card(
        920,
        510,
        340,
        100,
        "Stage changes",
        "Human · IVR · unavailable",
        stroke="#ACA79C",
        fill="#FBFAF7",
        accent="#77756E",
        icon_name="check",
    )
    body += """
  <line x1="64" y1="658" x2="1536" y2="658" stroke="#D8D4CA"/>
  <text x="64" y="690" class="kicker">INTERRUPTION</text>
  <text x="184" y="690" class="copy">False interruption can resume speech; confirmed interruption can stop it. Neither requires a new reply.</text>
  <text x="64" y="730" class="kicker">STAGE RULE</text>
  <text x="184" y="730" class="copy">Voicemail can move to IVR for submission, review, recording, or re-recording; it cannot move to Screening.</text>
"""
    diagram(
        filename="amd-stage-voicemail.svg",
        title="Voicemail",
        subtitle="Generate one message. AgentSession handles interruption; AMD guards the next reply.",
        badge="ACTIVE",
        description="Voicemail playback and independent reply authorization. AgentSession handles interruptions. AMD suppresses a new message for the same stage or applies a new stage's response policy.",
        body=body,
        height=770,
    )


def make_ivr() -> None:
    body = """
  <path d="M320 352H368" class="wire signal" marker-end="url(#arrow-signal)"/>
  <path d="M624 352H672" class="wire signal" marker-end="url(#arrow-signal)"/>
  <path d="M928 352H976" class="wire signal" marker-end="url(#arrow-signal)"/>
  <path d="M1232 352H1280" class="wire signal" marker-end="url(#arrow-signal)"/>
  <path d="M1408 402V470Q1408 482 1396 482H204Q192 482 192 470V402" class="wire repeat" marker-end="url(#arrow-signal)"/>
  <g transform="translate(824 482)">
    <rect x="-112" y="-13" width="224" height="26" rx="13" fill="#F3F1EB"/>
    <text x="0" y="4" text-anchor="middle" class="micro" fill="#3157C8">NEXT COMMITTED IVR TURN</text>
  </g>

  <path d="M496 302V290Q496 278 508 278H1424" class="wire stage"/>
  <path d="M944 278V254" class="wire human" marker-end="url(#arrow-human)"/>
  <path d="M1184 278V254" class="wire signal" marker-end="url(#arrow-signal)"/>
  <path d="M1424 278V254" class="wire unavailable" marker-end="url(#arrow-unavailable)"/>
"""
    body += card(
        64,
        302,
        256,
        100,
        "IVR prompt",
        "Spoken or keypad menu",
        icon_name="grid-3x3",
        accent=SIGNAL,
    )
    body += card(368, 302, 256, 100, "Client-side EOT", "Classify; emit amd_prediction", number="1")
    body += card(672, 302, 256, 100, "Inject instructions", "Describe the current menu", number="2")
    body += card(976, 302, 256, 100, "Choose an action", "Speak or call the DTMF tool", number="3")
    body += card(1280, 302, 256, 100, "Wait", "Next prompt", number="4")
    body += outcome(
        832,
        154,
        224,
        "Human",
        "Operator answers",
        icon_name="user-round",
        color=HUMAN,
        fill=HUMAN_TINT,
    )
    body += card(
        1072,
        154,
        224,
        100,
        "Voicemail",
        "Record or re-record",
        icon_name="mailbox",
        stroke=SIGNAL,
        fill=SIGNAL_TINT,
        accent=SIGNAL,
    )
    body += outcome(
        1312,
        154,
        224,
        "Unavailable",
        "Cannot continue",
        icon_name="ban",
        color=UNAVAILABLE,
        fill=UNAVAILABLE_TINT,
    )
    body += """
  <line x1="64" y1="526" x2="1536" y2="526" stroke="#D8D4CA"/>
  <text x="64" y="558" class="kicker">STAGE RULE</text>
  <text x="184" y="558" class="copy">If IVR follows voicemail, keep the local message-played flag and the prior-stage category.</text>
"""
    diagram(
        filename="amd-stage-ivr.svg",
        title="IVR",
        subtitle="Navigate a multi-turn menu while AMD watches for an operator, voicemail, or failure.",
        badge="ACTIVE",
        description="IVR-stage turn loop with speech or DTMF and exits to a human, voicemail, or unavailable.",
        body=body,
    )


def completion_body(*, human: bool) -> str:
    color = HUMAN if human else UNAVAILABLE
    tint = HUMAN_TINT if human else UNAVAILABLE_TINT
    icon_name = "user-round" if human else "ban"
    category = "human" if human else "machine-unavailable"
    first_title = "Human detected" if human else "Unavailable detected"
    first_copy = "Gateway completion category" if human else "Rejected or cannot continue"
    action_title = "Release held response" if human else "Cancel held response"
    action_copy = "Use the normal Agent reply" if human else "Do not generate a machine reply"
    final_title = "Normal Agent path" if human else "Application decides"
    final_copy = "AMD is no longer active" if human else "Choose the next call action"
    wire_class = "human" if human else "unavailable"
    marker = "human" if human else "unavailable"

    body = f"""
  <path d="M320 298H368" class="wire {wire_class}" marker-end="url(#arrow-{marker})"/>
  <path d="M624 298H672" class="wire {wire_class}" marker-end="url(#arrow-{marker})"/>
  <path d="M928 298H976" class="wire {wire_class}" marker-end="url(#arrow-{marker})"/>
  <path d="M1232 298H1280" class="wire {wire_class}" marker-end="url(#arrow-{marker})"/>
"""
    body += card(
        64,
        248,
        256,
        100,
        first_title,
        first_copy,
        icon_name=icon_name,
        stroke=color,
        fill=tint,
        accent=color,
        title_fill=color,
    )
    body += card(
        368,
        248,
        256,
        100,
        "Emit prediction",
        f"category = {category}",
        number="1",
        stroke=color,
        fill="#FBFAF7",
        accent=color,
    )
    body += card(
        672,
        248,
        256,
        100,
        action_title,
        action_copy,
        number="2",
        stroke=color,
        fill="#FBFAF7",
        accent=color,
    )
    body += card(
        976,
        248,
        256,
        100,
        "Complete AMD",
        "Close stream; reason = finished",
        number="3",
        stroke=color,
        fill="#FBFAF7",
        accent=color,
    )
    body += card(
        1280,
        248,
        256,
        100,
        final_title,
        final_copy,
        icon_name="check",
        stroke=color,
        fill=tint,
        accent=color,
        title_fill=color,
    )
    body += f"""
  <line x1="64" y1="430" x2="1536" y2="430" stroke="{BORDER}"/>
  <text x="64" y="464" class="kicker">EVENT ORDER</text>
  <text x="184" y="464" class="copy">amd_prediction is emitted before the transition; amd_completed is emitted after the action is active.</text>
  <text x="64" y="514" class="kicker">COMPLETION</text>
  <text x="184" y="514" class="copy">execute() resolves with the same AMDCompletedEvent instance delivered to listeners.</text>
"""
    return body


def make_human() -> None:
    diagram(
        filename="amd-stage-human.svg",
        title="Human",
        subtitle="AMD completes when a human is detected.",
        badge="COMPLETE",
        description="Human completion flow from gateway prediction through response release and AMD completion.",
        body=completion_body(human=True),
    )


def make_unavailable() -> None:
    diagram(
        filename="amd-stage-unavailable.svg",
        title="Unavailable",
        subtitle="AMD reports the outcome and leaves call policy to the application.",
        badge="COMPLETE",
        description="Unavailable completion flow from gateway prediction through response cancellation and AMD completion.",
        body=completion_body(human=False),
    )


def main() -> None:
    make_uncertain()
    make_screening()
    make_voicemail()
    make_ivr()
    make_human()
    make_unavailable()


if __name__ == "__main__":
    main()
