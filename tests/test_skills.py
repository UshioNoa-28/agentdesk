from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, TestCase

from agent.application.context.projector import MessageContextProjector
from agent.domain.messages import Message, MessageKind
from agent.domain.model_messages import ModelMessage, ModelTurn, ToolCall
from agent.domain.skills import SkillState
from agent.domain.tools import ToolContext
from agent.graph.agent_graph import AgentGraph
from agent.graph.hook_registry import AgentHookRegistry
from agent.infrastructure.skills import (
    SkillCatalog,
    SkillConfig,
    load_skill_configuration,
)
from agent.metatools import LoadSkillTool
from agent.prompt.system import build_system_prompt
from tests.tokenizer_support import budget_token_service


class SkillCatalogTests(TestCase):
    def test_repository_skill_is_registered_and_loadable(self) -> None:
        root = Path(__file__).parents[1]
        configuration = load_skill_configuration(root / "agent" / "skills.yaml")
        catalog = SkillCatalog(
            configuration.skills,
            max_file_bytes=configuration.max_file_bytes,
        )

        self.assertEqual(("structured-answer",), tuple(item.name for item in catalog.metadata()))
        document = catalog.get_document("structured-answer")
        self.assertIsNotNone(document)
        self.assertIn("Conclusion", document.instructions)

    def test_loads_front_matter_and_searches_full_markdown(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "skills"
            skill_path = root / "scientific-writing" / "SKILL.md"
            skill_path.parent.mkdir(parents=True)
            skill_path.write_text(
                "---\n"
                "name: scientific-writing\n"
                "description: Improve academic writing and reviewer-facing structure.\n"
                "---\n\n"
                "# Scientific writing\n\nUse a clear claim-evidence structure.\n",
                encoding="utf-8",
            )

            catalog = SkillCatalog(
                [
                    SkillConfig(
                        name="scientific-writing",
                        description="Improve academic writing and reviewer-facing structure.",
                        file_path=str(skill_path),
                    )
                ]
            )
            metadata = catalog.metadata()
            document = catalog.get_document("scientific-writing")

        self.assertEqual(("scientific-writing",), tuple(item.name for item in metadata))
        self.assertIsNotNone(document)
        self.assertIn("claim-evidence", document.instructions)
        self.assertTrue(document.metadata.path.endswith("SKILL.md"))

    def test_yaml_metadata_is_authoritative_without_front_matter(self) -> None:
        with TemporaryDirectory() as directory:
            skill_path = Path(directory) / "instructions.md"
            skill_path.write_text(
                "# Instructions\n\nUse the configured metadata.\n", encoding="utf-8"
            )
            catalog = SkillCatalog(
                [SkillConfig("configured-name", "Configured description", str(skill_path))]
            )

        metadata = catalog.metadata()[0]
        self.assertEqual("configured-name", metadata.name)
        self.assertEqual("Configured description", metadata.description)
        self.assertIn(
            "configured metadata",
            catalog.get_document("configured-name").instructions,
        )

    def test_invalid_skill_is_skipped_without_hiding_valid_skill(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "valid" / "SKILL.md"
            valid.parent.mkdir()
            valid.write_text(
                "---\nname: valid\ndescription: A valid skill.\n---\n\nDo the work.\n",
                encoding="utf-8",
            )
            invalid = root / "invalid" / "SKILL.md"
            invalid.parent.mkdir()
            invalid.write_text("\n", encoding="utf-8")

            catalog = SkillCatalog(
                [
                    SkillConfig("valid", "A valid skill.", str(valid)),
                    SkillConfig("invalid", "An invalid skill.", str(invalid)),
                ]
            )

        self.assertEqual(("valid",), tuple(item.name for item in catalog.metadata()))
        self.assertEqual(1, len(catalog.errors))

    def test_statuses_include_loaded_failed_and_disabled_entries(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            loaded = root / "loaded.md"
            loaded.write_text("# Loaded\n\nUse this guidance.\n", encoding="utf-8")
            failed = root / "failed.md"
            failed.write_text("\n", encoding="utf-8")
            disabled = root / "disabled.md"
            catalog = SkillCatalog(
                [
                    SkillConfig("loaded", "Loaded skill.", str(loaded)),
                    SkillConfig("failed", "Failed skill.", str(failed)),
                    SkillConfig("disabled", "Disabled skill.", str(disabled), enabled=False),
                ]
            )

            statuses = catalog.list_statuses()
            metadata = catalog.metadata()
            failed_document = catalog.get_document("failed")
            disabled_document = catalog.get_document("disabled")

        self.assertEqual(
            (
                ("loaded", SkillState.LOADED),
                ("failed", SkillState.FAILED),
                ("disabled", SkillState.DISABLED),
            ),
            tuple((status.name, status.state) for status in statuses),
        )
        self.assertIsNone(statuses[0].error)
        self.assertIn("instructions must not be empty", statuses[1].error or "")
        self.assertIsNone(statuses[2].error)
        self.assertEqual(("loaded",), tuple(item.name for item in metadata))
        self.assertIsNone(failed_document)
        self.assertIsNone(disabled_document)

    def test_failed_skill_stays_failed_without_reload(self) -> None:
        """目录构造期加载一次；启动后文件修复也不影响本进程，重启即恢复。"""
        with TemporaryDirectory() as directory:
            skill_path = Path(directory) / "recoverable.md"
            catalog = SkillCatalog(
                [SkillConfig("recoverable", "Recoverable skill.", str(skill_path))]
            )
            self.assertEqual(SkillState.FAILED, catalog.list_statuses()[0].state)

            skill_path.write_text("# Recovered\n\nUse the recovered guidance.\n", encoding="utf-8")
            statuses = catalog.list_statuses()
            document = catalog.get_document("recoverable")

        self.assertEqual(SkillState.FAILED, statuses[0].state)
        self.assertIsNone(document)

    def test_configuration_reads_skill_metadata_and_size_limit(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "skills.yaml"
            path.write_text(
                "skills:\n"
                "  - {name: one, description: First skill, file_path: one/SKILL.md}\n"
                "  - {name: two, description: Second skill, "
                "file_path: two/SKILL.md, enabled: false}\n"
                "max_file_bytes: 4096\n",
                encoding="utf-8",
            )
            configuration = load_skill_configuration(path)

        self.assertEqual(("one", "two"), tuple(skill.name for skill in configuration.skills))
        self.assertEqual("one/SKILL.md", configuration.skills[0].file_path)
        self.assertFalse(configuration.skills[1].enabled)
        self.assertEqual(4096, configuration.max_file_bytes)

    def test_configuration_requires_name_description_and_file_path(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "skills.yaml"
            path.write_text(
                "skills:\n  - {name: only-name, file_path: skill.md}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "description"):
                load_skill_configuration(path)


class SkillToolAndPromptTests(IsolatedAsyncioTestCase):
    async def test_load_skill_returns_document_by_exact_name(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "skills" / "csv"
            root.mkdir(parents=True)
            (root / "SKILL.md").write_text(
                "---\nname: csv\ndescription: Parse CSV data.\n---\n\nUse csv.DictReader.\n",
                encoding="utf-8",
            )
            result = await LoadSkillTool(
                SkillCatalog([SkillConfig("csv", "Parse CSV data.", str(root / "SKILL.md"))]),
            ).aexecute({"name": "csv"}, context=ToolContext(caller_session_id="s-1"))

        payload = json.loads(result.content)
        self.assertTrue(payload["ok"])
        self.assertEqual("csv", payload["skill"]["name"])
        self.assertTrue(payload["skill"]["path"].endswith("SKILL.md"))
        self.assertIn("csv.DictReader", payload["skill"]["content"])

    async def test_load_skill_unknown_name_lists_available_skills(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "skills" / "csv"
            root.mkdir(parents=True)
            (root / "SKILL.md").write_text(
                "---\nname: csv\ndescription: Parse CSV data.\n---\n\nUse csv.DictReader.\n",
                encoding="utf-8",
            )
            catalog = SkillCatalog(
                [SkillConfig("csv", "Parse CSV data.", str(root / "SKILL.md"))]
            )

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
        self.assertNotIn("skills/csv/SKILL.md", prompt)


class SkillGraphTests(IsolatedAsyncioTestCase):
    async def test_loaded_skill_stays_a_tool_result_and_is_replayed(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "skills" / "csv"
            root.mkdir(parents=True)
            (root / "SKILL.md").write_text(
                "---\nname: csv\ndescription: Parse CSV data.\n---\n\nUse csv.DictReader.\n",
                encoding="utf-8",
            )
            tool = LoadSkillTool(
                SkillCatalog([SkillConfig("csv", "Parse CSV data.", str(root / "SKILL.md"))]),
            )
            model = _SkillModel()
            persisted: list[tuple[ModelMessage, dict[str, object]]] = []
            projector = MessageContextProjector(token_counter=budget_token_service())

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
                hook_registry=hook_registry,
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


__all__ = ["SkillCatalogTests", "SkillGraphTests", "SkillToolAndPromptTests"]
