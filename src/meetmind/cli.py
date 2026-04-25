from __future__ import annotations

import click


@click.group()
def main() -> None:
    """MeetMind — local AI meeting assistant."""


@main.command()
def record() -> None:
    """Start capturing system audio for a new meeting."""
    click.echo("Recording... (not yet implemented)")


@main.command()
@click.argument("meeting_id")
def summarize(meeting_id: str) -> None:
    """Run transcript + summary pipeline on an existing recording."""
    click.echo(f"Summarizing {meeting_id}... (not yet implemented)")
