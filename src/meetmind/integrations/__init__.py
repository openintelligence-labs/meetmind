"""Outbound integrations.

Notion, Linear, Slack, Obsidian, GitHub Issues, Google Calendar,
Apple Reminders for v1.0. Presidio redaction profiles applied per export
target. Webhook bus for v1.2.

Module boundary: leaf package — pure transformations from internal types
to external APIs. Never imports from any other meetmind package except
`models` and `crypto`.
"""
