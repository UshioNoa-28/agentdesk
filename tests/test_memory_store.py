from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from agent.domain.memory import MemoryLayer, MemoryOperation, MemoryWriteError
from agent.infrastructure.memory import MemoryStore
from agent.infrastructure.memory.settings import load_memory_enabled


def _store(root: Path, *, enabled: bool = True, home: Path | None = None) -> MemoryStore:
    return MemoryStore(start=root, home=home, enabled=enabled)


def _record(
    store: MemoryStore,
    operation: MemoryOperation,
    content: str,
    *,
    session_id: str = "session-1",
    layer: MemoryLayer = MemoryLayer.PROJECT,
) -> None:
    store.record(operation, content, session_id=session_id, layer=layer)


class MemoryStoreTests(TestCase):
    def test_layers_resolve_to_home_and_workspace_agent_desk(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = _store(root / "workspace", home=root / "home")

            self.assertEqual(
                store.layer_root(MemoryLayer.USER),
                (root / "home").resolve() / ".agent-desk" / ".memory",
            )
            self.assertEqual(
                store.layer_root(MemoryLayer.PROJECT),
                (root / "workspace").resolve() / ".agent-desk" / ".memory",
            )

    def test_record_appends_one_line_per_call(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = _store(root / "workspace", home=root / "home")
            path = store.inbox_path(MemoryLayer.PROJECT, "session-1")

            _record(store, MemoryOperation.REMEMBER, "uses uv")
            _record(store, MemoryOperation.FORGET, "stale fact")

            self.assertEqual(path.suffix, ".md")
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertTrue(lines[0].startswith("- ") and lines[0].endswith(" remember uses uv"))
            self.assertTrue(lines[1].endswith(" forget stale fact"))
            self.assertEqual(
                [item.content for item in store.pending_records(MemoryLayer.PROJECT)],
                ["uses uv", "stale fact"],
            )

    def test_content_is_collapsed_to_a_single_line(self) -> None:
        with TemporaryDirectory() as directory:
            store = _store(Path(directory) / "workspace", home=Path(directory) / "home")

            _record(store, MemoryOperation.REMEMBER, " line one\n  line two\t")

            self.assertEqual(
                [item.content for item in store.pending_records(MemoryLayer.PROJECT)],
                ["line one line two"],
            )

    def test_sessions_and_layers_do_not_share_a_file(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = _store(root / "workspace", home=root / "home")

            _record(store, MemoryOperation.REMEMBER, "a", session_id="s1")
            _record(store, MemoryOperation.REMEMBER, "b", session_id="s2")
            _record(
                store,
                MemoryOperation.REMEMBER,
                "c",
                session_id="s1",
                layer=MemoryLayer.USER,
            )

            self.assertEqual(
                [item.content for item in store.pending_records(MemoryLayer.PROJECT)],
                ["a", "b"],
            )
            self.assertEqual(
                [item.session_id for item in store.pending_records(MemoryLayer.PROJECT)],
                ["s1", "s2"],
            )
            self.assertEqual(
                [item.content for item in store.pending_records(MemoryLayer.USER)],
                ["c"],
            )

    def test_a_missing_inbox_is_no_pending_records(self) -> None:
        with TemporaryDirectory() as directory:
            store = _store(Path(directory) / "workspace", home=Path(directory) / "home")

            self.assertEqual(store.pending_records(MemoryLayer.PROJECT), ())

    def test_a_malformed_pending_line_stops_the_reader(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = _store(root / "workspace", home=root / "home")
            _record(store, MemoryOperation.REMEMBER, "ok")
            path = store.inbox_path(MemoryLayer.PROJECT, "session-1")
            path.write_text("just some prose\n", encoding="utf-8")

            with self.assertRaises(MemoryWriteError):
                store.pending_records(MemoryLayer.PROJECT)

    def test_blank_and_oversized_content_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = _store(root / "workspace", home=root / "home")

            with self.assertRaises(MemoryWriteError):
                _record(store, MemoryOperation.REMEMBER, "   ")
            with self.assertRaises(MemoryWriteError):
                store.record(
                    MemoryOperation.REMEMBER,
                    "x" * 2_001,
                    session_id="s1",
                    layer=MemoryLayer.PROJECT,
                )
            self.assertEqual(list((root / "workspace" / ".agent-desk").rglob("*.md")), [])

    def test_session_id_cannot_escape_the_inbox_directory(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = _store(root / "workspace", home=root / "home")

            for hostile in ("../elsewhere", "a/b", "", ".", "x" * 65):
                with self.assertRaises(MemoryWriteError):
                    store.record(
                        MemoryOperation.REMEMBER,
                        "fact",
                        session_id=hostile,
                        layer=MemoryLayer.PROJECT,
                    )


class MemorySettingsTests(TestCase):
    def _write(self, root: Path, payload: dict[str, object]) -> None:
        settings_dir = root / ".agent-desk"
        settings_dir.mkdir(parents=True, exist_ok=True)
        (settings_dir / "settings.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_missing_file_or_section_defaults_to_disabled(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertFalse(load_memory_enabled(root, home=root))

            self._write(root, {"permissions": {"mode": "default"}})
            self.assertFalse(load_memory_enabled(root, home=root))

    def test_nearest_layer_wins(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            home, workspace = root / "home", root / "home" / "project"
            self._write(home, {"memory": {"enabled": True}})
            self._write(workspace, {"memory": {"enabled": False}})

            self.assertTrue(load_memory_enabled(home, home=home))
            self.assertFalse(load_memory_enabled(workspace, home=home))

    def test_bad_section_type_fails_even_when_shadowed(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            home, workspace = root / "home", root / "home" / "project"
            self._write(home, {"memory": {"enabled": "yes"}})
            self._write(workspace, {"memory": {"enabled": True}})

            with self.assertRaises(ValueError):
                load_memory_enabled(workspace, home=home)
