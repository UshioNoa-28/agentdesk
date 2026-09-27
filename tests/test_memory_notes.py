from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from agent.domain.memory import MemoryLayer, MemoryNotFound, MemoryWriteError
from agent.infrastructure.memory import MemoryStore
from agent.infrastructure.memory.notes import MEMORY_INDEX_FILE_NAME, NoteBook

MEMORY_DIR = ".memory"


def _notebook(root: Path) -> NoteBook:
    return NoteBook(MemoryStore(start=root / "workspace", home=root / "home", enabled=True))


class WriteNoteTests(TestCase):
    def test_body_file_and_index_line_are_written_together(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            book = _notebook(root)

            note = book.write_note(
                MemoryLayer.PROJECT,
                title="game preference",
                description="what type of game user like",
                body="User likes Slay the Spire.",
            )

            self.assertEqual(note.entry.title, "game-preference")
            self.assertEqual(
                book.read_index(MemoryLayer.PROJECT),
                f"game-preference\t{note.entry.timestamp.isoformat()}\t"
                "what type of game user like\n",
            )
            # 正文与 MEMORY.md 同级，标题就是文件名。
            self.assertEqual(
                (root / "workspace" / ".agent-desk" / MEMORY_DIR / "game-preference.md")
                .read_text("utf-8"),
                "User likes Slay the Spire.\n",
            )
            self.assertEqual(
                book.note_path(MemoryLayer.PROJECT, "game-preference").read_text("utf-8"),
                "User likes Slay the Spire.\n",
            )

    def test_duplicate_title_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            book = _notebook(Path(directory))
            book.write_note(MemoryLayer.USER, title="role", description="d", body="backend")

            with self.assertRaises(MemoryWriteError):
                book.write_note(MemoryLayer.USER, title="Role", description="d2", body="ml")

            self.assertEqual(
                [entry.title for entry in book.list_entries(MemoryLayer.USER)],
                ["role"],
            )
            self.assertEqual(book.read_note(MemoryLayer.USER, "role").body, "backend\n")

    def test_title_is_normalised_into_a_file_name(self) -> None:
        with TemporaryDirectory() as directory:
            book = _notebook(Path(directory))

            note = book.write_note(
                MemoryLayer.PROJECT,
                title="  Install   Deps  with  UV  ",
                description="one\nline\nonly",
                body="body",
            )

            self.assertEqual(note.entry.title, "install-deps-with-uv")
            self.assertEqual(note.entry.description, "one line only")

    def test_title_cannot_name_a_path(self) -> None:
        with TemporaryDirectory() as directory:
            book = _notebook(Path(directory))

            for hostile in ("../../etc/passwd", "a/b", "  ", ".hidden"):
                with self.subTest(title=hostile):
                    with self.assertRaises(MemoryWriteError):
                        book.write_note(
                            MemoryLayer.PROJECT, title=hostile, description="d", body="b"
                        )

    def test_oversized_title_is_rejected_before_any_write(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            book = _notebook(root)

            with self.assertRaises(MemoryWriteError):
                book.write_note(MemoryLayer.PROJECT, title="t" * 200, description="d", body="b")

            self.assertFalse((root / "workspace" / ".agent-desk" / MEMORY_DIR).exists())


class ReadNoteTests(TestCase):
    def test_round_trip_returns_entry_and_body(self) -> None:
        with TemporaryDirectory() as directory:
            book = _notebook(Path(directory))
            created = book.write_note(
                MemoryLayer.USER, title="Role", description="d", body="Backend engineer."
            )

            self.assertEqual(book.read_note(MemoryLayer.USER, "role"), created)

    def test_unknown_title_raises(self) -> None:
        with TemporaryDirectory() as directory:
            book = _notebook(Path(directory))

            with self.assertRaises(MemoryNotFound):
                book.read_note(MemoryLayer.USER, "nope")

    def test_indexed_note_without_a_body_file_raises(self) -> None:
        with TemporaryDirectory() as directory:
            book = _notebook(Path(directory))
            book.write_note(MemoryLayer.PROJECT, title="orphan", description="d", body="b")
            book.note_path(MemoryLayer.PROJECT, "orphan").unlink()

            with self.assertRaises(MemoryNotFound):
                book.read_note(MemoryLayer.PROJECT, "orphan")

    def test_malformed_index_line_raises_instead_of_being_skipped(self) -> None:
        with TemporaryDirectory() as directory:
            book = _notebook(Path(directory))
            book.write_note(MemoryLayer.PROJECT, title="good", description="d", body="b")
            book.index_path(MemoryLayer.PROJECT).write_text("a bare sentence\n", encoding="utf-8")

            with self.assertRaises(MemoryWriteError):
                book.list_entries(MemoryLayer.PROJECT)

    def test_layers_keep_their_own_index(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            book = _notebook(root)
            book.write_note(MemoryLayer.PROJECT, title="project fact", description="d", body="b")
            book.write_note(MemoryLayer.USER, title="user fact", description="d", body="b")

            self.assertEqual(
                [entry.title for entry in book.list_entries(MemoryLayer.PROJECT)],
                ["project-fact"],
            )
            self.assertTrue(
                (root / "home" / ".agent-desk" / MEMORY_DIR / MEMORY_INDEX_FILE_NAME).is_file()
            )


class UpdateNoteTests(TestCase):
    def test_partial_update_keeps_untouched_fields_and_bumps_timestamp(self) -> None:
        with TemporaryDirectory() as directory:
            book = _notebook(Path(directory))
            created = book.write_note(
                MemoryLayer.USER,
                title="prefers terse answers",
                description="Keep replies short.",
                body="Old body.",
            )

            updated = book.update_note(MemoryLayer.USER, "prefers-terse-answers", body="New body.")

            self.assertEqual(updated.entry.title, created.entry.title)
            self.assertEqual(updated.entry.description, created.entry.description)
            self.assertGreaterEqual(updated.entry.timestamp, created.entry.timestamp)
            self.assertEqual(updated.body, "New body.\n")
            self.assertEqual(len(book.list_entries(MemoryLayer.USER)), 1)

    def test_update_rewrites_the_index_line_in_place(self) -> None:
        with TemporaryDirectory() as directory:
            book = _notebook(Path(directory))
            book.write_note(MemoryLayer.PROJECT, title="first", description="d", body="b")
            book.write_note(MemoryLayer.PROJECT, title="second", description="d", body="b")

            book.update_note(
                MemoryLayer.PROJECT, "second", description="Rewritten.", body="c"
            )

            self.assertEqual(
                [
                    (entry.title, entry.description)
                    for entry in book.list_entries(MemoryLayer.PROJECT)
                ],
                [("first", "d"), ("second", "Rewritten.")],
            )

    def test_update_of_unknown_title_raises_without_writing(self) -> None:
        with TemporaryDirectory() as directory:
            book = _notebook(Path(directory))

            with self.assertRaises(MemoryNotFound):
                book.update_note(MemoryLayer.USER, "nope", body="b")

            self.assertFalse(book.index_path(MemoryLayer.USER).exists())


class DeleteNoteTests(TestCase):
    def test_index_line_and_file_both_go_away(self) -> None:
        with TemporaryDirectory() as directory:
            book = _notebook(Path(directory))
            book.write_note(MemoryLayer.USER, title="outdated", description="d", body="b")

            book.delete_note(MemoryLayer.USER, "outdated")

            self.assertEqual(book.list_entries(MemoryLayer.USER), ())
            self.assertFalse(book.note_path(MemoryLayer.USER, "outdated").exists())

    def test_delete_leaves_other_lines_intact(self) -> None:
        with TemporaryDirectory() as directory:
            book = _notebook(Path(directory))
            kept = book.write_note(MemoryLayer.USER, title="keep", description="d", body="b")
            book.write_note(MemoryLayer.USER, title="drop", description="d", body="b")

            book.delete_note(MemoryLayer.USER, "drop")

            self.assertEqual(
                book.read_index(MemoryLayer.USER),
                f"keep\t{kept.entry.timestamp.isoformat()}\td\n",
            )

    def test_delete_of_unknown_title_raises(self) -> None:
        with TemporaryDirectory() as directory:
            book = _notebook(Path(directory))

            with self.assertRaises(MemoryNotFound):
                book.delete_note(MemoryLayer.USER, "nope")

            self.assertFalse(book.index_path(MemoryLayer.USER).exists())
