"""Slash command handlers for the AgentDesk CLI."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from rich import box
from rich.table import Table

import cli.ui as ui
from cli.picker import SessionPicker
from cli.theme import BORDER, CYAN, WHITE

if TYPE_CHECKING:
    from cli.app import AnnaCliApp


def handle_new(app: AnnaCliApp, argument: str) -> None:
    """Prepare a new session lazily; supports ``/new <title> --tools a,b,c``."""
    title, tools = _parse_new_argument(argument)
    app.reset_to_new_session(pending_title=title, pending_tools=tools)
    if title:
        ui.console.print(f"[green]Ready for session:[/green] [white]{title}[/white] [dim](Session will be created on your first message)[/dim]\n")
    else:
        ui.console.print("[green]Ready for a new session.[/green] [dim](Session will be created on your first message)[/dim]\n")


def handle_resume(app: AnnaCliApp, argument: str) -> None:
    """Switch or resume a session. Opens interactive picker if no argument is provided."""
    arg = argument.strip()
    sessions = app.client.list_sessions()

    if not sessions:
        ui.console.print("[dim]No sessions found. Use [cyan]/new[/cyan] to create one.[/dim]\n")
        return

    if not arg:
        cur_id = app.current_session.get("id") if app.current_session else None
        picker = SessionPicker(sessions, current_session_id=cur_id)
        selected = picker.run()
        if not selected:
            return
        app.open_session(selected)
        return

    target = _resolve_session(sessions, arg)
    if not target:
        ui.console.print(f"[red]Session not found: \"{arg}\". Type [cyan]/resume[/cyan] to select.[/red]\n")
        return

    app.open_session(target)


def handle_review(app: AnnaCliApp, argument: str) -> None:
    """Quick code review command."""
    prompt = argument.strip() or "Please review current git changes and code modifications in the workspace, highlighting potential issues and improvements."
    app.ask_agent(prompt, print_prompt=True)


def handle_sessions(app: AnnaCliApp, _argument: str) -> None:
    """List all sessions in a table."""
    sessions = app.client.list_sessions()
    ui.print_sessions_table(sessions, current_session_id=app.current_session.get("id") if app.current_session else None)


def handle_session(app: AnnaCliApp, _argument: str) -> None:
    """View current session details."""
    if not _require_current_session(app):
        return
    messages = app.client.list_messages(app.current_session["id"])
    ui.print_session_detail(app.current_session, len(messages))


def handle_subagents(app: AnnaCliApp, argument: str) -> None:
    """List and switch between subagents for the current main session."""
    if not _require_current_session(app):
        return

    sessions = app.client.list_sessions()
    current = app.current_session
    owner_id = current.get("main_session_id") or current.get("id")

    main_session = next((s for s in sessions if s.get("id") == owner_id), current)
    subs = [s for s in sessions if s.get("main_session_id") == owner_id and s.get("id") != owner_id]

    if not subs:
        ui.console.print("[dim](No subagents found for the current session)[/dim]\n")
        return

    arg = argument.strip()
    if arg:
        target = _resolve_session([main_session, *subs], arg)
        if not target:
            ui.console.print(f"[red]Session not found: \"{arg}\". Type [cyan]/subagent[/cyan] to select.[/red]\n")
            return
        app.open_session(target)
        return

    from cli.picker import SubagentPicker
    cur_id = current.get("id")
    picker = SubagentPicker(main_session=main_session, subagents=subs, current_session_id=cur_id)
    selected = picker.run()
    if not selected:
        return
    app.open_session(selected)


def handle_rename(app: AnnaCliApp, argument: str) -> None:
    """Rename current session or set pending session title."""
    title = argument.strip().strip("'\"")
    if not title:
        ui.console.print("[yellow]Usage: /rename <new_title>[/yellow]\n")
        return

    if not app.current_session:
        app._pending_title = title
        ui.console.print(f"[green]Session title set to:[/green] [white]{title}[/white] [dim](will be applied on your first message)[/dim]\n")
        return

    updated = app.client.rename_session(app.current_session["id"], title)
    app.current_session = updated
    ui.console.print(f"[green]Session renamed to:[/green] [white]{updated.get('title')}[/white]\n")


def handle_delete(app: AnnaCliApp, argument: str) -> None:
    """Delete a session (by index or session ID)."""
    arg = argument.strip()
    if not arg:
        ui.console.print("[yellow]Usage: /delete <index|SessionID>[/yellow]\n")
        return

    sessions = app.client.list_sessions()
    target = _resolve_session(sessions, arg)
    if not target:
        ui.console.print(f"[red]Session not found: \"{arg}\".[/red]\n")
        return

    try:
        confirm = ui.console.input(f"[yellow]Delete session {target.get('title')} ({target.get('id')})? Type 'y' to confirm: [/yellow]")
    except (KeyboardInterrupt, EOFError):
        ui.console.print("\n[dim]Deletion cancelled.[/dim]\n")
        return

    if confirm.strip().lower() not in {"y", "yes"}:
        ui.console.print("[dim]Deletion cancelled.[/dim]\n")
        return

    app.client.delete_session(target["id"])
    if app.current_session and app.current_session.get("id") == target["id"]:
        app.current_session = None
    ui.console.print(f"[green]Deleted session:[/green] [white]{target.get('title')}[/white]\n")


def handle_compact(app: AnnaCliApp, _argument: str) -> None:
    """Manually compact session context."""
    if not _require_current_session(app):
        return

    try:
        with ui.console.status("[cyan]Compacting session context...[/cyan]", spinner="dots"):
            result = app.client.compact_session(app.current_session["id"])
        ui.print_compact_result(result)
    except KeyboardInterrupt:
        ui.console.print("\n[dim]Context compaction cancelled.[/dim]\n")


def handle_health(app: AnnaCliApp, _argument: str) -> None:
    """Check backend service health status."""
    with ui.console.status("[cyan]Checking backend health...[/cyan]", spinner="dots"):
        health = app.client.health()
    if hasattr(app, "_health"):
        app._health = health
    ui.print_health(health, base_url=app.base_url)


def handle_mcps(app: AnnaCliApp, argument: str) -> None:
    """View or retry MCP servers."""
    parts = argument.strip().split()
    if not parts:
        mcps = app.client.list_mcps()
        ui.print_mcps_table(mcps)
        return

    if parts[0].lower() == "retry" and len(parts) == 2:
        server_id = parts[1]
        with ui.console.status(f"[cyan]Retrying MCP server {server_id}...[/cyan]", spinner="dots"):
            status = app.client.retry_mcp(server_id)
        ui.print_mcps_table([status])
        return

    ui.console.print("[yellow]Usage: /mcps or /mcps retry <server-id>[/yellow]\n")


def handle_skills(app: AnnaCliApp, argument: str) -> None:
    """View local skills catalog."""
    arg = argument.strip().lower()
    if not arg:
        skills = app.client.list_skills()
        ui.print_skills_table(skills)
        return

    ui.console.print("[yellow]Usage: /skills[/yellow]\n")


def handle_clear(app: AnnaCliApp, _argument: str) -> None:
    """Clear screen and redraw banner with session history."""
    ui.clear_screen()
    app.print_header()
    if app.current_session:
        try:
            messages = app.client.list_messages(app.current_session["id"])
            ui.print_history_list(messages, limit=10)
        except Exception:
            pass


def handle_history(app: AnnaCliApp, argument: str) -> None:
    """Display conversation history messages (usage: /history or /history <count>)."""
    if not _require_current_session(app):
        return
    limit: int | None = None
    arg = argument.strip()
    if arg.isdigit():
        limit = int(arg)
    try:
        messages = app.client.list_messages(app.current_session["id"])
    except Exception as exc:
        ui.print_error_message(f"Failed to fetch history: {exc}")
        return

    if not messages:
        ui.console.print("[dim]No messages in current session.[/dim]\n")
        return
    ui.print_history_list(messages, limit=limit)


def handle_help(_app: AnnaCliApp, _argument: str) -> None:
    """View commands cheatsheet."""
    from cli.completer import SLASH_COMMANDS

    table = Table(
        title="[bold white]Available Commands (Slash Commands)[/bold white]",
        border_style=BORDER,
        box=box.ROUNDED,
        title_justify="left",
        expand=True,
    )
    table.add_column("Command", style=f"bold {CYAN}", width=18)
    table.add_column("Description", style=WHITE)

    for cmd, desc in SLASH_COMMANDS:
        table.add_row(cmd, desc)

    ui.console.print()
    ui.console.print(table)
    ui.console.print()


def handle_rewind(app: AnnaCliApp, _argument: str) -> None:
    """Rewind past turns and fork a new session."""
    app.rewind_and_resume()


def handle_quit(app: AnnaCliApp, _argument: str) -> None:
    """Exit CLI."""
    app.running = False


def _parse_new_argument(argument: str) -> tuple[str | None, list[str] | None]:
    """Parse ``/new <title> --tools a,b,c`` arguments."""
    parts = argument.strip().split()
    title_words: list[str] = []
    tools: list[str] | None = None

    index = 0
    while index < len(parts):
        part = parts[index]
        if part.startswith("--tools="):
            tools = [t.strip() for t in part.split("=", 1)[1].split(",") if t.strip()]
        elif part == "--tools" and index + 1 < len(parts):
            tools = [t.strip() for t in parts[index + 1].split(",") if t.strip()]
            index += 1
        else:
            title_words.append(part)
        index += 1

    title = " ".join(title_words).strip().strip("'\"") or None
    return title, tools


def _resolve_session(sessions: list[dict[str, Any]], arg: str) -> dict[str, Any] | None:
    """Resolve session by 1-based index, ID prefix, or title substring."""
    arg_lower = arg.lower()

    if arg.isdigit():
        idx = int(arg)
        if 1 <= idx <= len(sessions):
            return sessions[idx - 1]
        return None

    for s in sessions:
        sid = (s.get("id") or "").lower()
        if sid == arg_lower or sid.startswith(arg_lower):
            return s

    title_matches = [s for s in sessions if arg_lower in (s.get("title") or "").lower()]
    return title_matches[0] if title_matches else None


def _require_current_session(app: AnnaCliApp) -> bool:
    """Ensure active session exists; otherwise print guidance and return False."""
    if app.current_session:
        return True
    if getattr(app, "_pending_title", None):
        ui.console.print(f"[dim]Pending session: [white]{app._pending_title}[/white] (will be created on your first message)[/dim]\n")
    else:
        ui.console.print("[dim]Ready for a new session. (Will be created on your first message)[/dim]\n")
    return False


COMMAND_MAP = {
    "review": handle_review,
    "resume": handle_resume,
    "rewind": handle_rewind,
    "fork": handle_rewind,
    "new": handle_new,
    "sessions": handle_sessions,
    "session": handle_session,
    "subagents": handle_subagents,
    "subagent": handle_subagents,
    "sub": handle_subagents,
    "rename": handle_rename,
    "delete": handle_delete,
    "compact": handle_compact,
    "health": handle_health,
    "mcps": handle_mcps,
    "skills": handle_skills,
    "history": handle_history,
    "clear": handle_clear,
    "help": handle_help,
    "quit": handle_quit,
    "exit": handle_quit,
}
