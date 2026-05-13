"""MCP `prompts/*` primitives for MeetMind.

Prompts are reusable, parameterized templates an MCP client can pull and
render before sending to its own LLM. They are NOT executed server-side
— MeetMind ships the *text*, the client decides what model to feed it
to. This is the spec's intent: prompts are user-discoverable, not
server-private.

The catalog is small and curated. Each entry is a known-good template
that pairs naturally with our resources (transcripts, summaries,
decisions, action items) — e.g. ``daily_digest`` reads
``meetmind://meetings`` filtered to the past 24h and asks the client to
synthesize. ``what_changed`` compares two meeting URIs.

Wire shape per MCP 2025-11:

  prompts/list  → {prompts: [{name, description, arguments: [...]}]}
  prompts/get   → {messages: [{role, content: {type:'text', text:'...'}}]}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PromptArgument:
    name: str
    description: str
    required: bool = False

    def to_wire(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "required": self.required,
        }


@dataclass(frozen=True)
class PromptDescriptor:
    name: str
    description: str
    arguments: tuple[PromptArgument, ...]
    template: str  # uses ``{argname}`` placeholders

    def to_wire(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "arguments": [a.to_wire() for a in self.arguments],
        }

    def render(self, args: dict[str, Any]) -> str:
        # Validate required args first so we can give a useful error
        # rather than KeyError-ing inside .format().
        missing = [a.name for a in self.arguments if a.required and not args.get(a.name)]
        if missing:
            raise KeyError(f"prompt {self.name!r} missing required args: {missing}")
        # Fill defaults for missing optional args so format() doesn't blow up.
        filled = {a.name: "" for a in self.arguments}
        filled.update({k: v for k, v in args.items() if v is not None})
        try:
            return self.template.format(**filled)
        except KeyError as e:  # pragma: no cover — guards against malformed templates
            raise KeyError(f"prompt template references unknown arg: {e}") from e


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


_DAILY_DIGEST = PromptDescriptor(
    name="daily_digest",
    description="Produce a one-screen digest of today's meetings with decisions and action items.",
    arguments=(
        PromptArgument("date", "ISO date (YYYY-MM-DD); defaults to today.", required=False),
    ),
    template=(
        "You are reviewing meeting notes for {date}. Use the MeetMind MCP server's "
        "`meetmind://meetings` resource to enumerate today's meetings, then for each meeting "
        "pull `meetmind://meeting/{{id}}/summary`, `…/decisions`, and `…/actions`.\n\n"
        "Produce a single Markdown digest with three sections:\n"
        "1. **What happened** — one sentence per meeting, grouped by hour.\n"
        "2. **Decisions made** — bulleted, each with which meeting it came from.\n"
        "3. **Open action items** — only `status: open`, with owner and due date.\n\n"
        "Be terse. No filler. Cite meeting titles, never IDs."
    ),
)


_WHAT_CHANGED = PromptDescriptor(
    name="what_changed",
    description="Compare two meetings on the same topic and surface what shifted.",
    arguments=(
        PromptArgument("meeting_a_id", "Earlier meeting ULID.", required=True),
        PromptArgument("meeting_b_id", "Later meeting ULID.", required=True),
    ),
    template=(
        "Compare these two MeetMind meetings:\n"
        "  • Earlier: meetmind://meeting/{meeting_a_id}\n"
        "  • Later:   meetmind://meeting/{meeting_b_id}\n\n"
        "Pull the `…/transcript`, `…/decisions`, and `…/actions` subresources for both.\n\n"
        "Report:\n"
        "1. **Same topic, new position** — claims that flipped or shifted between meetings.\n"
        "2. **Newly decided** — decisions in B that weren't in A.\n"
        "3. **Action items moved or dropped** — open in A, closed/cancelled/missing in B.\n"
        "4. **People who changed their stance** — quote both sides.\n\n"
        "Quote verbatim from the transcripts. Don't speculate beyond what's recorded."
    ),
)


_PREP_FOR_MEETING = PromptDescriptor(
    name="prep_for_meeting",
    description="Brief the user on an upcoming meeting using their history with the attendees.",
    arguments=(
        PromptArgument("attendees", "Comma-separated list of attendee names.", required=True),
        PromptArgument("topic", "What the meeting is about.", required=False),
    ),
    template=(
        "I have an upcoming meeting with: {attendees}.\n"
        "Topic: {topic}\n\n"
        "Use the MeetMind MCP server to:\n"
        "1. Call `get_speaker_history` for each attendee on the topic.\n"
        "2. Call `search_meetings` for the topic across recent meetings.\n"
        "3. Call `list_action_items` filtered to `status: open` and these owners.\n\n"
        "Brief me with:\n"
        "  • **What they last said about this** — one quote per person.\n"
        "  • **Open items they owe me / I owe them** — with due dates.\n"
        "  • **Decisions already made** — so I don't re-litigate.\n"
        "  • **Three sharp questions to ask**, grounded in the history above.\n\n"
        "Cap the brief at 250 words."
    ),
)


_FOLLOW_UP_DRAFT = PromptDescriptor(
    name="follow_up_draft",
    description="Draft a post-meeting follow-up email from a meeting's decisions and actions.",
    arguments=(
        PromptArgument("meeting_id", "Meeting ULID to draft from.", required=True),
        PromptArgument(
            "tone",
            "Tone: 'crisp', 'warm', or 'formal'. Default: crisp.",
            required=False,
        ),
    ),
    template=(
        "Draft a follow-up email summarizing meeting meetmind://meeting/{meeting_id}.\n\n"
        "Pull `…/summary`, `…/decisions`, and `…/actions` from the resource server.\n\n"
        "Tone: {tone}\n\n"
        "Structure:\n"
        "  1. One-line subject.\n"
        "  2. Two-sentence opener — what we met about, what we decided.\n"
        "  3. **Decisions** — bulleted.\n"
        "  4. **Next steps** — only open actions, with owner and due date in bold.\n"
        "  5. Sign-off.\n\n"
        "No greetings beyond 'Hi team,'. No 'just circling back'. Action items first, "
        "rationale second."
    ),
)


_REVIEW_MY_TALK_TIME = PromptDescriptor(
    name="review_my_talk_time",
    description="Honest feedback on the user's speaking patterns across recent meetings.",
    arguments=(
        PromptArgument("speaker_id", "Your speaker_id in MeetMind.", required=True),
        PromptArgument("days", "How many days back to review. Default: 7.", required=False),
    ),
    template=(
        "Review my participation as speaker {speaker_id} over the last {days} days.\n\n"
        "Use the MeetMind MCP server to:\n"
        "  • List recent meetings via `meetmind://meetings`.\n"
        "  • For each, call `attendee_stats` to see my talk-time share.\n"
        "  • Call `get_speaker_history` to sample what I actually said.\n\n"
        "Tell me:\n"
        "  • **Where I dominated** (>40% talk-time) — was the discussion better for it?\n"
        "  • **Where I went quiet** — any meetings where I should have spoken up?\n"
        "  • **Patterns in what I say** — repeated phrases, hedge words, interruptions.\n"
        "  • **One concrete change** I could try at my next meeting.\n\n"
        "Be direct. I asked for honesty, not encouragement."
    ),
)


_WEEKLY_REVIEW = PromptDescriptor(
    name="weekly_review",
    description="Summarize what happened across the past week and what's left open.",
    arguments=(
        PromptArgument("week_of", "ISO date for the Monday of the week to review.", required=False),
    ),
    template=(
        "Synthesize my week from the MeetMind MCP server.\n\n"
        "Pull `meetmind://meetings` filtered to the 7 days ending {week_of} (or this week "
        "if blank). For each meeting fetch `…/decisions` and `…/actions`.\n\n"
        "Produce:\n"
        "  • **Highlights** — three-sentence narrative of the week.\n"
        "  • **Decisions made this week** — grouped by initiative.\n"
        "  • **Open action items I own** — sorted by due date.\n"
        "  • **Open action items others owe me** — same sort.\n"
        "  • **Themes** — any topic that came up in 3+ meetings.\n\n"
        "Cite meetings by title; never raw IDs."
    ),
)


_QUARTERLY_SUMMARY = PromptDescriptor(
    name="quarterly_summary",
    description="High-level quarterly recap suitable for a perf doc or board update.",
    arguments=(PromptArgument("quarter", "Which quarter, e.g. 'Q1 2026'.", required=True),),
    template=(
        "Build a {quarter} recap from MeetMind.\n\n"
        "Use the resource server to enumerate meetings in the quarter, then sample "
        "summaries + decisions + action items across the period.\n\n"
        "Output structure:\n"
        "  1. **What we shipped** — concrete decisions and outcomes (3–6 bullets).\n"
        "  2. **What we decided NOT to do** — things explicitly killed or deferred.\n"
        "  3. **Patterns** — how my time was spent (talk-time share, meeting volume, "
        "     recurring themes).\n"
        "  4. **What's still open** — outstanding action items I own.\n\n"
        "Tone: candid and concrete, no filler. Suitable to drop into a perf doc."
    ),
)


_PROMPTS: dict[str, PromptDescriptor] = {
    p.name: p
    for p in (
        _DAILY_DIGEST,
        _WHAT_CHANGED,
        _PREP_FOR_MEETING,
        _FOLLOW_UP_DRAFT,
        _REVIEW_MY_TALK_TIME,
        _WEEKLY_REVIEW,
        _QUARTERLY_SUMMARY,
    )
}


def list_prompts() -> list[PromptDescriptor]:
    return list(_PROMPTS.values())


def get_prompt(name: str) -> PromptDescriptor | None:
    return _PROMPTS.get(name)


def render_prompt(name: str, args: dict[str, Any] | None = None) -> str:
    """Render a prompt to its final text. Raises ``KeyError`` for unknown name."""
    prompt = _PROMPTS.get(name)
    if prompt is None:
        raise KeyError(f"unknown prompt: {name}")
    return prompt.render(args or {})
