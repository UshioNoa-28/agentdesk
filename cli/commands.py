"""Slash command handlers for the AgentDesk CLI."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from rich import box
from rich.errors import LiveError
from rich.table import Table

import cli.ui as ui
from cli.picker import McpPicker, ModelPicker, SessionPicker
from cli.theme import BORDER, CYAN, WHITE

if TYPE_CHECKING:
    from cli.app import AnnaCliApp


@contextmanager
def _safe_status(message: str) -> Iterator[None]:
    """Use a spinner when the Console is free, otherwise render plainly."""
    status = ui.make_status(message)
    started = False
    try:
        try:
            status.start()
            started = True
        except LiveError:
            # A background turn owns Rich's single Live slot.  The command
            # still performs its request; only the animation is skipped.
            pass
        yield
    finally:
        if started:
            try:
                status.stop()
            except Exception:
                pass


def _turn_is_active(app: AnnaCliApp) -> bool:
    """Return whether a turn or interactive waiter owns the prompt."""
    count = getattr(app, "active_turn_count", 0)
    active = isinstance(count, int) and count > 0
    live_fn = getattr(app, "_has_live_waiters", None)
    if callable(live_fn):
        try:
            pending = live_fn() is True
        except Exception:
            pending = False
    else:
        pending_fn = getattr(app, "_has_pending_ask_users", None)
        pending = False
        if callable(pending_fn):
            try:
                # Test doubles may return MagicMock here; only the real boolean
                # result should block a command.
                pending = pending_fn() is True
            except Exception:
                pending = False
    return active or pending


def _reject_while_running(app: AnnaCliApp, action: str) -> bool:
    if not _turn_is_active(app):
        return False
    ui.console.print(
        f"[yellow]Cannot {action} while a turn is running. "
        "Keep typing, or wait for the current response to finish.[/yellow]\n"
    )
    return True


def handle_new(app: AnnaCliApp, argument: str) -> None:
    """Prepare a new session lazily; supports ``/new <title> --tools a,b,c``."""
    if _reject_while_running(app, "start a new session"):
        return
    title, tools = _parse_new_argument(argument)
    app.reset_to_new_session(pending_title=title, pending_tools=tools)
    if title:
        ui.console.print(
            f"[green]Ready for session:[/green] [white]{title}[/white] "
            "[dim](Session will be created on your first message)[/dim]\n"
        )
    else:
        ui.console.print(
            "[green]Ready for a new session.[/green] "
            "[dim](Session will be created on your first message)[/dim]\n"
        )


def handle_resume(app: AnnaCliApp, argument: str) -> None:
    """Switch or resume a session. Opens interactive picker if no argument is provided."""
    if _reject_while_running(app, "switch sessions"):
        return
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

    # The picker presents only main sessions, so numeric arguments must use
    # the same filtered ordering.  Explicit IDs/titles may still address a
    # subagent when requested directly.
    main_sessions = [s for s in sessions if not s.get("main_session_id")]
    candidates = (main_sessions or sessions) if arg.isdigit() else sessions
    target = _resolve_session(candidates, arg)
    if not target:
        ui.console.print(
            f'[red]Session not found: "{arg}". Type [cyan]/resume[/cyan] to select.[/red]\n'
        )
        return

    app.open_session(target)


def handle_memory(app: AnnaCliApp, argument: str) -> None:
    """View long-term memories or inspect a memory note."""
    arg = argument.strip()
    try:
        memories = app.client.list_memories()
    except Exception as exc:
        ui.print_error_message(f"Failed to list memories: {exc}")
        return

    if not arg or arg.lower() == "list":
        ui.print_memories_table(memories)
        return

    # Filter by layer: /memory user or /memory project
    if arg.lower() in {"user", "project"}:
        filtered = [m for m in memories if (m.get("layer") or "").lower() == arg.lower()]
        ui.print_memories_table(filtered)
        return

    # View note by title: /memory view <title> or /memory <title>
    title_target = arg
    if title_target.lower().startswith("view "):
        title_target = title_target[5:].strip()

    match = next(
        (m for m in memories if m.get("title") == title_target),
        next((m for m in memories if (m.get("title") or "").lower() == title_target.lower()), None),
    )

    if match:
        layer = match.get("layer") or "user"
        title = match.get("title") or title_target
        try:
            detail = app.client.get_memory(layer, title)
        except Exception as exc:
            ui.print_error_message(f"Failed to read memory note: {exc}")
            return
        if detail and detail.get("content"):
            ui.print_memory_note(layer, title, detail["content"])
            return

    # Filter memories containing arg in title or description
    matched = [
        m
        for m in memories
        if arg.lower() in (m.get("title") or "").lower()
        or arg.lower() in (m.get("description") or "").lower()
    ]
    if matched:
        ui.print_memories_table(matched)
    else:
        ui.console.print(
            f'[yellow]No memories matching "{arg}". '
            "Type [cyan]/memory[/cyan] to list all.[/yellow]\n"
        )


def handle_sessions(app: AnnaCliApp, _argument: str) -> None:
    """List all sessions in a table."""
    sessions = app.client.list_sessions()
    ui.print_sessions_table(
        sessions, current_session_id=app.current_session.get("id") if app.current_session else None
    )


def handle_session(app: AnnaCliApp, _argument: str) -> None:
    """View current session details."""
    if not _require_current_session(app):
        return
    messages = app.client.list_messages(app.current_session["id"])
    ui.print_session_detail(app.current_session, len(messages))


def handle_subagents(app: AnnaCliApp, argument: str) -> None:
    """List and switch between subagents for the current main session."""
    if _reject_while_running(app, "switch sessions"):
        return
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
            ui.console.print(
                f'[red]Session not found: "{arg}". Type [cyan]/subagent[/cyan] to select.[/red]\n'
            )
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
        ui.console.print(
            f"[green]Session title set to:[/green] [white]{title}[/white] "
            "[dim](will be applied on your first message)[/dim]\n"
        )
        return

    updated = app.client.rename_session(app.current_session["id"], title)
    app.current_session = updated
    ui.console.print(f"[green]Session renamed to:[/green] [white]{updated.get('title')}[/white]\n")


def handle_delete(app: AnnaCliApp, argument: str) -> None:
    """Delete the current session — the only one this window may hold."""
    if _reject_while_running(app, "delete a session"):
        return
    target = app.current_session
    if not target or not target.get("id"):
        ui.console.print("[yellow]No current session to delete.[/yellow]\n")
        return

    try:
        confirm = ui.console.input(
            f"[yellow]Delete session {target.get('title')} ({target.get('id')})? "
            "Type 'y' to confirm: [/yellow]"
        )
    except (KeyboardInterrupt, EOFError):
        ui.console.print("\n[dim]Deletion cancelled.[/dim]\n")
        return

    if confirm.strip().lower() not in {"y", "yes"}:
        ui.console.print("[dim]Deletion cancelled.[/dim]\n")
        return

    app.client.delete_session(target["id"])
    # Queued markers are local presentation state; deleting the session must
    # not let them leak into the next lazy session.
    if isinstance(getattr(app, "_queued_turns", None), dict) and hasattr(
        app, "_discard_queued_turns"
    ):
        app._discard_queued_turns(target["id"])
    if getattr(app, "_held_session_id", None) == target["id"]:
        app.client.release(target["id"])
        app._held_session_id = None
    app.current_session = None
    if hasattr(app, "_pending_title"):
        app._pending_title = None
    if hasattr(app, "_pending_tools"):
        app._pending_tools = None
    ui.console.print(f"[green]Deleted session:[/green] [white]{target.get('title')}[/white]\n")


def handle_compact(app: AnnaCliApp, _argument: str) -> None:
    """Manually compact session context."""
    if _reject_while_running(app, "compact context"):
        return
    if not _require_current_session(app):
        return

    try:
        with _safe_status("[cyan]Compacting session context...[/cyan]"):
            result = app.client.compact_session(app.current_session["id"])
        ui.print_compact_result(result)
    except KeyboardInterrupt:
        ui.console.print("\n[dim]Context compaction cancelled.[/dim]\n")


def _print_mcps_retry_summary(retried: dict[str, dict[str, Any]]) -> None:
    """Print a clean single-line status for retried MCP servers instead of a giant table."""
    if not retried:
        return
    for sid, res in retried.items():
        state = (res.get("state") or "").upper()
        if state == "CONNECTED":
            tools = res.get("tool_count", 0)
            ui.console.print(
                f"[green]✓ MCP server '{sid}' connected ({tools} tools).[/green]"
            )
        else:
            err = res.get("error") or "Failed to connect"
            first_err = err.split("\n")[0].strip()
            ui.console.print(f"[red]✗ MCP server '{sid}' failed: {first_err}[/red]")
    ui.console.print()


def handle_mcps(app: AnnaCliApp, argument: str) -> None:
    """View or retry MCP servers."""
    parts = argument.strip().split()
    if not parts:
        mcps = app.client.list_mcps()
        if not mcps:
            ui.console.print("[dim](No MCP servers configured)[/dim]\n")
            return
        picker = McpPicker(mcps, retry_fn=app.client.retry_mcp)
        picker.run()
        _print_mcps_retry_summary(picker.retried_results)
        return

    if parts[0].lower() == "list":
        mcps = app.client.list_mcps()
        ui.print_mcps_table(mcps)
        return

    if parts[0].lower() == "retry":
        mcps = app.client.list_mcps()
        if not mcps:
            ui.console.print("[dim](No MCP servers configured)[/dim]\n")
            return
        if len(parts) == 1:
            failed = [m["id"] for m in mcps if m.get("state") == "failed" and m.get("id")]
            if not failed:
                ui.console.print("[dim]No failed MCP servers to retry.[/dim]\n")
                return
            target_ids = failed
        else:
            target_ids = [parts[1]]

        if sys.stdin.isatty():
            picker = McpPicker(
                mcps, retry_fn=app.client.retry_mcp, auto_retry_ids=target_ids
            )
            picker.run()
            _print_mcps_retry_summary(picker.retried_results)
        else:
            statuses: dict[str, dict[str, Any]] = {}
            for server_id in target_ids:
                with _safe_status(f"[cyan]Retrying MCP server '{server_id}'...[/cyan]"):
                    status = app.client.retry_mcp(server_id)
                    statuses[server_id] = status
            _print_mcps_retry_summary(statuses)
        return

    ui.console.print(
        "[yellow]Usage: /mcps, /mcps list, /mcps retry, or /mcps retry <server-id>[/yellow]\n"
    )


def handle_skills(app: AnnaCliApp, argument: str) -> None:
    """View local skills catalog."""
    arg = argument.strip().lower()
    if not arg:
        skills = app.client.list_skills()
        ui.print_skills_table(skills)
        return

    ui.console.print("[yellow]Usage: /skills[/yellow]\n")


def handle_clear(app: AnnaCliApp, _argument: str) -> None:
    """Clear the screen and redraw history with compact subagent reports."""
    if _reject_while_running(app, "clear the screen"):
        return
    ui.clear_screen()
    app.print_header()

    if not _require_current_session(app):
        return
    try:
        messages = app.client.list_messages(app.current_session["id"])
    except Exception as exc:
        ui.print_error_message(f"Failed to load session history: {exc}")
        return
    ui.print_history_list(
        messages,
        limit=None,
        show_subagent_details=False,
    )
    if isinstance(getattr(app, "_queued_turns", None), dict) and hasattr(
        app, "_discard_queued_turns"
    ):
        app._discard_queued_turns(app.current_session["id"])


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
    if _reject_while_running(app, "rewind the session"):
        return
    app.rewind_and_resume()


def handle_skip(app: AnnaCliApp, _argument: str) -> None:
    """Skip / pause the current pending clarification or cancel running turn."""
    if hasattr(app, "_skip_pending_ask_user") and app._skip_pending_ask_user():
        return
    if getattr(app, "active_turn_count", 0) > 0 and hasattr(app, "cancel_active_turns"):
        app.cancel_active_turns()
        ui.console.print("\n[yellow]> Request cancelled[/yellow]\n")
        return
    ui.console.print("[dim]No pending clarification or running turn to cancel.[/dim]\n")


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


def _resolve_model(models: list[dict[str, Any]], arg: str) -> dict[str, Any] | None:
    """Resolve model by 1-based index, model ID prefix, or name substring."""
    arg_lower = arg.lower()

    if arg.isdigit():
        idx = int(arg)
        if 1 <= idx <= len(models):
            return models[idx - 1]
        return None

    for m in models:
        mid = (m.get("id") or m.get("model_id") or "").lower()
        if mid == arg_lower or mid.startswith(arg_lower):
            return m

    name_matches = [m for m in models if arg_lower in (m.get("name") or "").lower()]
    return name_matches[0] if name_matches else None


def handle_models(app: AnnaCliApp, _argument: str) -> None:
    """List all configured models in a table."""
    try:
        models = app.client.list_models()
    except Exception as exc:
        ui.print_error_message(f"Failed to list models: {exc}")
        return

    cur_id = getattr(app, "model_id", None)
    if not cur_id:
        try:
            cur = app.client.get_current_model()
            cur_id = cur.get("id") if cur else None
        except Exception:
            pass

    ui.print_models_table(models, current_model_id=cur_id)


def handle_model(app: AnnaCliApp, argument: str) -> None:
    """Switch model. Opens interactive picker if no argument is provided."""
    if _reject_while_running(app, "switch model"):
        return

    try:
        models = app.client.list_models()
    except Exception as exc:
        ui.print_error_message(f"Failed to list models: {exc}")
        return

    if not models:
        ui.console.print("[dim]No models configured.[/dim]\n")
        return

    arg = argument.strip()
    cur_id = getattr(app, "model_id", None)
    if not cur_id:
        try:
            cur = app.client.get_current_model()
            cur_id = cur.get("id") if cur else None
        except Exception:
            pass

    if not arg:
        picker = ModelPicker(models, current_model_id=cur_id)
        selected = picker.run()
        if not selected:
            return
        target_id = selected.get("id") or selected.get("model_id") or ""
        if target_id and hasattr(app, "select_model"):
            app.select_model(target_id)
        return

    target = _resolve_model(models, arg)
    if not target:
        ui.console.print(
            f'[red]Model not found: "{arg}". '
            "Type [cyan]/models[/cyan] to list available models.[/red]\n"
        )
        return

    target_id = target.get("id") or target.get("model_id") or ""
    if target_id and hasattr(app, "select_model"):
        app.select_model(target_id)


def handle_mode(app: AnnaCliApp, argument: str) -> None:
    """View or switch permission mode (default, dont_ask, bypass)."""
    arg = argument.strip().lower()
    valid_modes = {"default", "dont_ask", "bypass"}
    if not arg:
        cur = getattr(app, "permission_mode", "default")
        mode_color = "red" if cur == "bypass" else "yellow" if cur == "dont_ask" else "cyan"
        lines = [
            f"[dim]Current permission mode:[/dim] [{mode_color} bold]{cur}[/{mode_color} bold]"
            "  [dim](Shift+Tab to cycle)[/dim]\n",
        ]
        modes_desc = [
            ("default", "Ask confirmation for operations (default)"),
            ("dont_ask", "Auto-reject operations requiring permission without asking"),
            ("bypass", "Bypass permission checks and execute directly (caution)"),
        ]
        for name, desc in modes_desc:
            if name == cur:
                lines.append(
                    f"  [{mode_color}]›[/{mode_color}] [{mode_color} bold]{name:<9}"
                    f"[/{mode_color} bold]  [dim]{desc}[/dim]"
                )
            else:
                lines.append(f"    [white]{name:<9}[/white]  [dim]{desc}[/dim]")
        lines.append("\n[dim]Usage: /mode <default|dont_ask|bypass>[/dim]\n")
        ui.console.print("\n".join(lines))
        return
    if arg not in valid_modes:
        ui.console.print(
            f'[red]Invalid mode "{arg}". Valid modes: {", ".join(sorted(valid_modes))}[/red]\n'
        )
        return
    try:
        fn = getattr(app.client, "set_permission_mode", None)
        if callable(fn):
            fn(arg)
    except Exception as exc:
        ui.print_error_message(f"Failed to switch mode: {exc}")
        return
    app._permission_mode = arg
    if hasattr(app, "_invalidate_prompt"):
        app._invalidate_prompt()
    ui.console.print(f"[green]Permission mode set to:[/green] [cyan]{arg}[/cyan]\n")


def _require_current_session(app: AnnaCliApp) -> bool:
    """Ensure active session exists; otherwise print guidance and return False."""
    if app.current_session:
        return True
    if getattr(app, "_pending_title", None):
        ui.console.print(
            f"[dim]Pending session: [white]{app._pending_title}[/white] "
            "(will be created on your first message)[/dim]\n"
        )
    else:
        ui.console.print(
            "[dim]Ready for a new session. (Will be created on your first message)[/dim]\n"
        )
    return False


COMMAND_MAP = {
    "new": handle_new,
    "resume": handle_resume,
    "rewind": handle_rewind,
    "fork": handle_rewind,
    "rename": handle_rename,
    "delete": handle_delete,
    "compact": handle_compact,
    "memory": handle_memory,
    "memories": handle_memory,
    "mcp": handle_mcps,
    "mcps": handle_mcps,
    "skills": handle_skills,
    "model": handle_model,
    "models": handle_models,
    "mode": handle_mode,
    "subagents": handle_subagents,
    "subagent": handle_subagents,
    "sub": handle_subagents,
    "sessions": handle_sessions,
    "session": handle_session,
    "clear": handle_clear,
    "cancel": handle_skip,
    "skip": handle_skip,
    "help": handle_help,
    "exit": handle_quit,
    "quit": handle_quit,
}
