from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, TestCase

from agent.application.context.projector import MessageContextProjector
from agent.application.graph.agent_graph import AgentGraph
from agent.application.graph.hook_registry import AgentHookRegistry
from agent.domain.context_settings import ContextSettings
from agent.domain.messages import Message, MessageKind
from agent.domain.model_messages import ModelMessage, ModelTurn, ToolCall
from agent.domain.tools import ToolContext
from agent.infrastructure.metatools import LoadSkillTool
from agent.infrastructure.project_settings import config_search_roots
from agent.infrastructure.skills import (
    AGENTDESK_DIR_NAME,
    SKILLS_DIR_NAME,
    SkillCatalog,
    discover_skill_dirs,
)
from agent.prompt.system import build_system_prompt
from tests.tokenizer_support import budget_token_service

# 投影器帽子：默认调度参数、单条工具结果 8000（等于不截断）。
_PROJECTOR_SETTINGS = ContextSettings(max_tool_result_tokens=8_000)


class _AllowPermissions:
    async def authorize(self, **_kwargs: object) -> str:
        return "allow"


def _write_skill(skill_dir: Path, *, name: str, description: str, body: str) -> None:
    """在 ``skill_dir`` 下写一个带 front matter 的 SKILL.md。"""

    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n",
        encoding="utf-8",
    )


def _skills_root(base: Path, name: str) -> Path:
    return base / AGENTDESK_DIR_NAME / SKILLS_DIR_NAME / name


class SkillDiscoveryTests(TestCase):
    def test_search_roots_walk_up_to_home_and_stop(self) -> None:
        home = Path("/home/anna")
        start = home / "code" / "projects" / "Agent"
        self.assertEqual(
            (
                start,
                home / "code" / "projects",
                home / "code",
                home,
            ),
            config_search_roots(start, home=home),
        )

    def test_search_roots_appends_home_when_start_outside_it(self) -> None:
        home = Path("/home/anna")
        start = Path("/opt/work/project")
        roots = config_search_roots(start, home=home)
        self.assertEqual(start, roots[0])
        self.assertEqual(home, roots[-1])

    def test_nearer_skill_shadows_outer_copy(self) -> None:
        with TemporaryDirectory() as directory:
            home = Path(directory) / "home"
            project = home / "work" / "repo"
            _write_skill(
                _skills_root(project, "review"), name="review", description="p", body="near"
            )
            _write_skill(_skills_root(home, "review"), name="review", description="f", body="far")
            _write_skill(_skills_root(home, "solo"), name="solo", description="s", body="only")

            skill_dirs = discover_skill_dirs(project, home=home)

            names = [path.name for path in skill_dirs]
            self.assertEqual(["review"], names[:1])
            winners = {path.name: path for path in skill_dirs}
            self.assertIn("solo", winners)
            self.assertEqual(project, winners["review"].parents[2])

    def test_non_directories_and_dot_entries_are_ignored(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            skills_root = _skills_root(root, "real")
            _write_skill(skills_root, name="real", description="ok", body="work")
            (skills_root.parent / "loose.md").write_text("x", encoding="utf-8")
            (skills_root.parent / ".DS_Store").write_text("x", encoding="utf-8")

            skill_dirs = discover_skill_dirs(root, home=root)

            self.assertEqual(("real",), tuple(path.name for path in skill_dirs))


class SkillCatalogTests(TestCase):
    def test_repository_skill_is_discoverable_and_loadable(self) -> None:
        repo_root = Path(__file__).parents[1]
        catalog = SkillCatalog.from_root(repo_root)

        self.assertIn("structured-answer", tuple(item.name for item in catalog.metadata()))
        document = catalog.get_document("structured-answer")
        self.assertIsNotNone(document)
        self.assertIn("Conclusion", document.instructions)
        self.assertTrue(document.metadata.path.endswith("SKILL.md"))

    def test_front_matter_is_authoritative(self) -> None:
        with TemporaryDirectory() as directory:
            skill_dir = _skills_root(Path(directory), "csv")
            _write_skill(
                skill_dir, name="csv", description="Parse CSV data.", body="Use csv.DictReader."
            )
            catalog = SkillCatalog([skill_dir])

            metadata = catalog.metadata()[0]

        self.assertEqual("csv", metadata.name)
        self.assertEqual("Parse CSV data.", metadata.description)
        self.assertIn("csv.DictReader", catalog.get_document("csv").instructions)

    def test_missing_description_raises(self) -> None:
        with TemporaryDirectory() as directory:
            skill_dir = _skills_root(Path(directory), "bad")
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: bad\n---\n\nDo the work.\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "description"):
                SkillCatalog([skill_dir])

    def test_missing_front_matter_raises(self) -> None:
        with TemporaryDirectory() as directory:
            skill_dir = _skills_root(Path(directory), "nofm")
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text("# No front matter\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "front matter"):
                SkillCatalog([skill_dir])

    def test_name_must_match_directory(self) -> None:
        with TemporaryDirectory() as directory:
            skill_dir = _skills_root(Path(directory), "actual-dir")
            _write_skill(
                skill_dir,
                name="other-name",
                description="Mismatched name.",
                body="Work.",
            )

            with self.assertRaisesRegex(ValueError, "must match its directory name"):
                SkillCatalog([skill_dir])

    def test_invalid_name_characters_raise(self) -> None:
        with TemporaryDirectory() as directory:
            skill_dir = _skills_root(Path(directory), "Bad_Name")
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: Bad_Name\ndescription: Uppercase.\n---\n\nWork.\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "must match"):
                SkillCatalog([skill_dir])

    def test_empty_instructions_raise(self) -> None:
        with TemporaryDirectory() as directory:
            skill_dir = _skills_root(Path(directory), "empty")
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: empty\ndescription: No body.\n---\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "instructions must not be empty"):
                SkillCatalog([skill_dir])

    def test_missing_skill_file_raises(self) -> None:
        with TemporaryDirectory() as directory:
            skill_dir = _skills_root(Path(directory), "nofile")
            skill_dir.mkdir(parents=True)

            with self.assertRaisesRegex(ValueError, "missing SKILL.md"):
                SkillCatalog([skill_dir])


class SkillToolAndPromptTests(IsolatedAsyncioTestCase):
    async def test_load_skill_returns_document_by_exact_name(self) -> None:
        with TemporaryDirectory() as directory:
            skill_dir = _skills_root(Path(directory), "csv")
            _write_skill(
                skill_dir, name="csv", description="Parse CSV data.", body="Use csv.DictReader."
            )
            result = await LoadSkillTool(SkillCatalog([skill_dir])).aexecute(
                {"name": "csv"}, context=ToolContext(caller_session_id="s-1")
            )

        payload = json.loads(result.content)
        self.assertTrue(payload["ok"])
        self.assertEqual("csv", payload["skill"]["name"])
        self.assertTrue(payload["skill"]["path"].endswith("SKILL.md"))
        self.assertIn("csv.DictReader", payload["skill"]["content"])

    async def test_load_skill_unknown_name_lists_available_skills(self) -> None:
        with TemporaryDirectory() as directory:
            skill_dir = _skills_root(Path(directory), "csv")
            _write_skill(
                skill_dir, name="csv", description="Parse CSV data.", body="Use csv.DictReader."
            )
            catalog = SkillCatalog([skill_dir])

        result = await LoadSkillTool(catalog).aexecute(
            {"name": "no-such-skill"}, context=ToolContext(caller_session_id="s-1")
        )

        payload = json.loads(result.content)
        self.assertFalse(payload["ok"])
        self.assertEqual("skill_not_found", payload["error"]["code"])
        self.assertIn("csv", payload["error"]["message"])

    def test_skill_metadata_in_prompt_hides_path_and_mentions_load_skill(self) -> None:
        prompt = build_system_prompt(
            (),
            (("csv", "Parse CSV data."),),
            ("load_skill",),
        )

        self.assertIn("<skills>", prompt)
        self.assertIn('skill: "csv"', prompt)
        self.assertIn("load_skill", prompt)
        self.assertNotIn("csv/SKILL.md", prompt)


class SkillGraphTests(IsolatedAsyncioTestCase):
    async def test_loaded_skill_stays_a_tool_result_and_is_replayed(self) -> None:
        with TemporaryDirectory() as directory:
            skill_dir = _skills_root(Path(directory), "csv")
            _write_skill(
                skill_dir, name="csv", description="Parse CSV data.", body="Use csv.DictReader."
            )
            tool = LoadSkillTool(SkillCatalog([skill_dir]))
            model = _SkillModel()
            persisted: list[tuple[ModelMessage, dict[str, object]]] = []
            projector = MessageContextProjector(
                token_counter=budget_token_service(),
                settings=_PROJECTOR_SETTINGS,
            )

            async def persist_tool(state) -> None:
                """测试请求级 Hook 负责保存并投影 Skill 结果。"""

                for index, message in enumerate(state["messages"]):
                    if message.role == "tool" and not any(m is message for m, _ in persisted):
                        persisted.append((message, {"kind": MessageKind.SKILL_RESULT}))
                        state["messages"][index] = ModelMessage.tool(
                            name=message.tool_name or "tool",
                            tool_call_id=message.tool_call_id or "unknown",
                            content=projector.wrap_skill_guidance(message.content),
                        )

            hook_registry = AgentHookRegistry()
            hook_registry.register_after_tool(persist_tool)
            result = await AgentGraph(
                max_turns=4,
                settings=ContextSettings(),
                hook_registry=hook_registry,
                permission_service=_AllowPermissions(),
            ).ainvoke(
                messages=[ModelMessage.human("处理 csv")],
                model=model,
                tools=(tool,),
                caller_session_id="s-1",
            )

        self.assertEqual("done", result["messages"][-1].content)
        self.assertFalse(any("csv.DictReader" in content for content in model.system_contexts))
        self.assertTrue(any("<skill_guidance>" in content for content in model.tool_contexts))
        self.assertFalse(any("<untrusted_content>" in content for content in model.tool_contexts))
        tool_message, metadata = next(
            (message, metadata)
            for message, metadata in persisted
            if message.role == "tool"
        )
        self.assertEqual(MessageKind.SKILL_RESULT, metadata["kind"])
        self.assertIn("csv.DictReader", tool_message.content)

        stored = Message(
            id="message-1",
            session_id="s",
            seq=2,
            role="tool",
            content=tool_message.content,
            tool_call_id=tool_message.tool_call_id,
            tool_name=tool_message.tool_name,
            metadata={"kind": MessageKind.SKILL_RESULT},
        )
        replay_assistant = Message.create(
            session_id="s",
            seq=1,
            message=ModelMessage.assistant(
                content="",
                tool_calls=(ToolCall("skill-1", "load_skill", {}),),
            ),
            metadata={"kind": MessageKind.ASSISTANT_TOOL_CALL},
        )
        replayed = MessageContextProjector(
            token_counter=budget_token_service(),
            settings=_PROJECTOR_SETTINGS,
        ).project([replay_assistant, stored])
        self.assertTrue(any("csv.DictReader" in message.content for message in replayed))


class _SkillModel:
    def __init__(self) -> None:
        self.turn = 0
        self.system_contexts: list[str] = []
        self.tool_contexts: list[str] = []

    async def ainvoke(
        self, *, messages, tools, tool_choice=None, max_output_tokens=None
    ) -> ModelTurn:
        self.turn += 1
        self.system_contexts.extend(
            message.content for message in messages if message.role == "system"
        )
        self.tool_contexts.extend(
            message.content for message in messages if message.role == "tool"
        )
        if self.turn == 1:
            call = ToolCall(
                id="skill-1",
                name="load_skill",
                arguments={"name": "csv"},
            )
            return ModelTurn(
                message=ModelMessage.assistant(content="", tool_calls=(call,)),
                tool_calls=(call,),
            )
        return ModelTurn(message=ModelMessage.assistant(content="done"), tool_calls=())


__all__ = ["SkillCatalogTests", "SkillDiscoveryTests", "SkillGraphTests", "SkillToolAndPromptTests"]
