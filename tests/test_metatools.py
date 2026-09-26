from __future__ import annotations

import importlib
import json
import pkgutil
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock

from jsonschema import Draft202012Validator, ValidationError, validate

from agent.domain.entities import Session
from agent.domain.interruption import (
    AskUserRequest,
    InterruptionResult,
    InterruptionStatus,
)
from agent.domain.multi_agent import RESERVED_AGENT_NAMES, BusMessage
from agent.domain.tools import (
    ALL_META_TOOL_NAMES,
    MAIN_AGENT_ONLY_TOOL_NAMES,
    ToolContext,
    ToolDefinition,
)
from agent.infrastructure.metatools import MetaToolRegistry
from agent.infrastructure.metatools.ask_user import ASK_USER_DEFINITION, AskUserTool
from agent.infrastructure.metatools.define_subagent import (
    DEFINE_SUBAGENT_DEFINITION,
    GRANT_TOOL_DEFINITIONS,
    DefineSubagentTool,
)
from agent.infrastructure.metatools.execute_mcp import ExecuteMcpTool
from agent.infrastructure.metatools.list_subagents import ListSubagentsTool
from agent.infrastructure.metatools.load_skill import LoadSkillTool
from agent.infrastructure.metatools.search_mcp import SearchMcpTool
from agent.infrastructure.metatools.send_message import SendMessageTool
from agent.infrastructure.metatools.wait_for_replies import WaitForRepliesTool
from agent.infrastructure.runtime.status_registry import InMemoryAgentStatusRegistry


def _assert_rejected_by_schema(
    test: TestCase,
    definition,
    arguments: dict[str, object],
) -> None:
    """参数在进入工具前就该被 definition.parameters 拒掉。"""

    with test.assertRaises(ValidationError):
        validate(instance=arguments, schema=dict(definition.parameters))


def _all_definitions() -> dict[str, ToolDefinition]:
    """收集 ``agent.infrastructure.metatools`` 包内声明的全部 ToolDefinition 常量。"""

    package = importlib.import_module("agent.infrastructure.metatools")
    definitions: dict[str, ToolDefinition] = {}
    for module_info in pkgutil.walk_packages(package.__path__, "agent.infrastructure.metatools."):
        module = importlib.import_module(module_info.name)
        for value in vars(module).values():
            if isinstance(value, ToolDefinition):
                definitions[value.name] = value
    return definitions


class _CapturingSessionService:
    """记录 define_subagent 传给 SessionService.create 的全部参数。"""

    def __init__(
        self,
        main_title: str = "Main Session",
        fail_on_get: bool = False,
    ) -> None:
        self.main = Session.create(title=main_title)
        self.fail_on_get = fail_on_get
        self.calls: list[dict] = []

    async def get(self, session_id: str) -> Session:
        if self.fail_on_get:
            raise RuntimeError("owner session vanished")
        return self.main

    async def create(self, **kwargs) -> Session:
        self.calls.append(kwargs)
        return Session.create(
            title=kwargs["title"],
            main_session_id=kwargs.get("main_session_id"),
            allowed_tools=kwargs.get("allowed_tools", ()),
        )


class _IdentitySessionService:
    """按固定标题返回会话，供 SendMessageTool 推导发送方身份与归属。"""

    def __init__(self, session: Session) -> None:
        self.session = session

    async def get(self, session_id: str) -> Session:
        return self.session


class MetaToolRegistryTests(TestCase):
    def test_registry_binds_fixed_tools_in_stable_order(self) -> None:
        session_service = MagicMock()
        registry = MetaToolRegistry(
            tools=[
                SearchMcpTool(MagicMock()),
                LoadSkillTool(MagicMock()),
                ExecuteMcpTool(MagicMock()),
                AskUserTool(MagicMock()),
                DefineSubagentTool(session_service=session_service),
                SendMessageTool(runtime=MagicMock(), session_service=MagicMock()),
                ListSubagentsTool(
                    session_service=session_service,
                    status_registry=InMemoryAgentStatusRegistry(),
                ),
                WaitForRepliesTool(
                    session_service=MagicMock(),
                    default_timeout=60,
                    max_timeout=600,
                ),
            ]
        )

        tools = registry.tools
        self.assertEqual(
            (
                "search_mcp",
                "load_skill",
                "execute_mcp",
                "ask_user",
                "define_subagent",
                "send_message",
                "list_subagents",
                "wait_for_replies",
            ),
            tuple(tool.definition.name for tool in tools),
        )
        self.assertIn("mcp", tools[0].definition.parameters["required"])
        self.assertIn("name", tools[1].definition.parameters["required"])
        self.assertIn("arguments", tools[2].definition.parameters["required"])
        self.assertIn("question", tools[3].definition.parameters["required"])
        self.assertIn("name", tools[4].definition.parameters["required"])
        self.assertIn("message", tools[5].definition.parameters["required"])


class MetaToolSchemaContractTests(TestCase):
    """参数校验只在 graph 一处发生，schema 自身合法就是整套约定的前提。

    ``definition.parameters`` 不是合法 JSON Schema 时，``_execute_tool`` 会抛
    ``AgentGraphExecutionError`` 打断整轮，非法参数也会在无人拦截下直达工具。
    """

    def test_every_meta_tool_schema_is_valid(self) -> None:
        definitions = _all_definitions()

        self.assertEqual(set(ALL_META_TOOL_NAMES), set(definitions))
        for name, definition in sorted(definitions.items()):
            with self.subTest(tool=name):
                Draft202012Validator.check_schema(dict(definition.parameters))

    def test_schema_alone_rejects_blank_required_strings(self) -> None:
        """工具内不判空的必填字符串参数，空白必须由 schema 单独拒掉。

        只校验单个 property 的子 schema，避开同工具其他必填参数；空白即空的判断
        依赖 ``pattern: "\\S"`` 而非 ``minLength``，后者只数字符数。workspace 的
        path 与 search_text 的 query 不在列，它们的空白值本身是有意义的字面量。
        """

        definitions = _all_definitions()
        blank = ("", "   ", "\t\n")
        non_blank_arguments = {
            "ask_user": ("question",),
            "execute_mcp": ("mcp", "tool"),
            "load_skill": ("name",),
            "search_mcp": ("mcp", "query"),
            "send_message": ("recipient", "message"),
        }
        for name, properties in sorted(non_blank_arguments.items()):
            schema = definitions[name].parameters["properties"]
            for argument in properties:
                with self.subTest(tool=name, argument=argument):
                    property_schema = dict(schema[argument])
                    self.assertEqual("string", property_schema["type"])
                    validator = Draft202012Validator(property_schema)
                    for value in blank:
                        self.assertFalse(
                            validator.is_valid(value),
                            msg=f"{name}.{argument} accepted {value!r}",
                        )
                    self.assertTrue(validator.is_valid("value"))


class _RecordingRuntime:
    """记录 SendMessageTool 派发的消息，验证自环拦截不产生投递。"""

    def __init__(self) -> None:
        self.dispatched: list[BusMessage] = []

    async def dispatch(self, message: BusMessage) -> None:
        self.dispatched.append(message)


class DefineSubagentToolTests(IsolatedAsyncioTestCase):
    """验证 define_subagent 的默认工具白名单与主控专用工具拦截。"""

    def setUp(self) -> None:
        self.service = _CapturingSessionService()
        self.tool = DefineSubagentTool(session_service=self.service)

    def test_allowed_tools_schema_is_derived_from_grant_definitions(self) -> None:
        """enum 与描述从 GRANT_TOOL_DEFINITIONS 派生，新增 worker 工具时自动同步。"""
        schema = DEFINE_SUBAGENT_DEFINITION.parameters["properties"]["allowed_tools"]

        self.assertEqual(list(GRANT_TOOL_DEFINITIONS), schema["items"]["enum"])
        for name in GRANT_TOOL_DEFINITIONS:
            self.assertIn(f"'{name}'", schema["description"])
        for name, definition in GRANT_TOOL_DEFINITIONS.items():
            self.assertIn(definition.description, schema["description"])

    async def test_omitted_allowed_tools_defaults_to_send_message(self) -> None:
        """C2：省略 allowed_tools 时默认只授予 send_message。"""
        result = await self.tool.aexecute(
            {"name": "coder", "description": "writes code", "system_prompt": "be a coder"},
            context=self._context(),
        )

        payload = json.loads(result.content)
        self.assertTrue(payload["ok"])
        self.assertEqual(("send_message",), self.service.calls[0]["allowed_tools"])

    async def test_empty_allowed_tools_falls_back_to_default(self) -> None:
        """C2：显式空列表与省略等价，同样回退到 send_message 默认值。"""
        result = await self.tool.aexecute(
            {
                "name": "coder",
                "description": "d",
                "system_prompt": "p",
                "allowed_tools": [],
            },
            context=self._context(),
        )

        payload = json.loads(result.content)
        self.assertTrue(payload["ok"])
        self.assertEqual(("send_message",), self.service.calls[0]["allowed_tools"])

    async def test_explicit_allowed_tools_are_honored(self) -> None:
        result = await self.tool.aexecute(
            {
                "name": "coder",
                "description": "d",
                "system_prompt": "p",
                "allowed_tools": ["edit_file", "send_message"],
            },
            context=self._context(),
        )

        payload = json.loads(result.content)
        self.assertTrue(payload["ok"])
        self.assertEqual(
            ("edit_file", "send_message"), self.service.calls[0]["allowed_tools"]
        )

    async def test_main_agent_only_tools_are_rejected_without_creating_session(self) -> None:
        result_content = []
        for tool_name in MAIN_AGENT_ONLY_TOOL_NAMES:
            result = await self.tool.aexecute(
                {
                    "name": "bad",
                    "description": "d",
                    "system_prompt": "p",
                    "allowed_tools": [tool_name],
                },
                context=self._context(),
            )
            result_content.append(json.loads(result.content))

        self.assertTrue(all(payload["ok"] is False for payload in result_content))
        for tool_name, payload in zip(MAIN_AGENT_ONLY_TOOL_NAMES, result_content, strict=True):
            error = payload["error"]
            self.assertEqual("main_agent_only_tools", error["code"])
            self.assertIn(tool_name, error["message"])
        self.assertEqual([], self.service.calls)

    async def test_unknown_tools_are_rejected_without_creating_session(self) -> None:
        """未知工具名必须拒绝：放行会被 registry 静默丢弃，授权方无从察觉。"""
        result = await self.tool.aexecute(
            {
                "name": "coder",
                "description": "d",
                "system_prompt": "p",
                "allowed_tools": ["serach_mcp", "web_search"],
            },
            context=self._context(),
        )

        payload = json.loads(result.content)
        self.assertFalse(payload["ok"])
        error = payload["error"]
        self.assertEqual("unknown_tools", error["code"])
        self.assertIn("serach_mcp", error["message"])
        self.assertIn("web_search", error["message"])
        self.assertIn("edit_file", error["message"])
        self.assertEqual([], self.service.calls)

    async def test_created_subagent_session_uses_owner_session_id_and_prompt(self) -> None:
        """caller_session_id 即主会话：标题前缀与 main_session_id 均取自它。"""
        service = _CapturingSessionService(main_title="Main Session")
        tool = DefineSubagentTool(session_service=service)

        result = await tool.aexecute(
            {"name": "coder", "description": "writes code", "system_prompt": "be a coder"},
            context=ToolContext(caller_session_id=service.main.id),
        )

        payload = json.loads(result.content)
        self.assertTrue(payload["ok"])
        call = service.calls[0]
        self.assertEqual(f"{service.main.id}_subagent_coder", call["title"])
        self.assertEqual(service.main.id, call["main_session_id"])
        # 回报契约必须压过 persona：放在 raw prompt 之前，且写明纯文本不可达。
        self.assertTrue(
            call["custom_system_prompt"].startswith("<reporting_contract>")
        )
        self.assertIn("Plain text replies reach NOBODY", call["custom_system_prompt"])
        self.assertIn("be a coder", call["custom_system_prompt"])
        self.assertIn("send_message", call["custom_system_prompt"])

    def _context(self) -> ToolContext:
        return ToolContext(caller_session_id=self.service.main.id)

    async def test_owner_session_lookup_failure_is_reported(self) -> None:
        """C3：主会话查询失败必须上报，禁止吞异常继续创建子会话。"""
        service = _CapturingSessionService(fail_on_get=True)
        tool = DefineSubagentTool(session_service=service)

        result = await tool.aexecute(
            {"name": "coder", "description": "d", "system_prompt": "p"},
            context=ToolContext(caller_session_id=service.main.id),
        )

        payload = json.loads(result.content)
        self.assertFalse(payload["ok"])
        error = payload["error"]
        self.assertEqual("owner_session_unavailable", error["code"])
        self.assertIn("Could not load owner session", error["message"])
        self.assertIn("owner session vanished", error["message"])
        self.assertEqual([], service.calls)

    async def test_reserved_agent_names_are_rejected(self) -> None:
        """C5：main_agent / user 是保留名，不许被子代理占用。"""
        for reserved in RESERVED_AGENT_NAMES:
            result = await self.tool.aexecute(
                {"name": reserved, "description": "d", "system_prompt": "p"},
                context=self._context(),
            )
            payload = json.loads(result.content)
            self.assertFalse(payload["ok"])
            self.assertEqual("reserved_name", payload["error"]["code"])
            self.assertIn("reserved", payload["error"]["message"])
        self.assertEqual([], self.service.calls)


class SendMessageToolTests(IsolatedAsyncioTestCase):
    """验证 send_message 的身份推导、自环拦截与正常派发。"""

    def setUp(self) -> None:
        self.runtime = _RecordingRuntime()

    def _tool(self, session: Session) -> SendMessageTool:
        return SendMessageTool(
            runtime=self.runtime,
            session_service=_IdentitySessionService(session),
        )

    async def test_main_agent_cannot_send_message_to_itself(self) -> None:
        """C7：主控给自己发消息必须显式拒绝，而不是报成功后静默落库。"""
        main_session = Session.create(title="Main Session")
        result = await self._tool(main_session).aexecute(
            {"recipient": "main_agent", "message": "note to self"},
            context=ToolContext(caller_session_id=main_session.id),
        )

        payload = json.loads(result.content)
        self.assertFalse(payload["ok"])
        self.assertEqual("self_message_not_allowed", payload["error"]["code"])
        self.assertIn("Cannot send a message to yourself", payload["error"]["message"])
        self.assertEqual([], self.runtime.dispatched)

    async def test_dispatch_to_subagent_still_works(self) -> None:
        main_session = Session.create(title="Main Session")
        result = await self._tool(main_session).aexecute(
            {"recipient": "coder", "message": "write tests"},
            context=ToolContext(caller_session_id=main_session.id),
        )

        payload = json.loads(result.content)
        self.assertTrue(payload["ok"])
        self.assertEqual(1, len(self.runtime.dispatched))
        dispatched = self.runtime.dispatched[0]
        self.assertEqual("coder", dispatched.recipient)
        self.assertEqual("main_agent", dispatched.sender)
        self.assertEqual(main_session.id, dispatched.session_id)

    async def test_subagent_cannot_send_message_to_itself(self) -> None:
        sub_session = Session.create(
            title="owner-1_subagent_coder", main_session_id="owner-1"
        )
        result = await self._tool(sub_session).aexecute(
            {"recipient": "coder", "message": "note to self"},
            context=ToolContext(caller_session_id=sub_session.id),
        )

        payload = json.loads(result.content)
        self.assertFalse(payload["ok"])
        self.assertEqual([], self.runtime.dispatched)

    async def test_subagent_report_targets_owner_session_with_identity(self) -> None:
        """子代理回报：sender 取标题后缀，信封归属固定为其主会话。"""
        sub_session = Session.create(
            title="owner-1_subagent_coder", main_session_id="owner-1"
        )
        result = await self._tool(sub_session).aexecute(
            {"recipient": "main_agent", "message": "all done"},
            context=ToolContext(caller_session_id=sub_session.id),
        )

        payload = json.loads(result.content)
        self.assertTrue(payload["ok"])
        self.assertEqual(1, len(self.runtime.dispatched))
        dispatched = self.runtime.dispatched[0]
        self.assertEqual("main_agent", dispatched.recipient)
        self.assertEqual("coder", dispatched.sender)
        self.assertEqual("owner-1", dispatched.session_id)

    async def test_subagent_cannot_message_another_subagent(self) -> None:
        sub_session = Session.create(
            title="owner-1_subagent_coder", main_session_id="owner-1"
        )
        result = await self._tool(sub_session).aexecute(
            {"recipient": "reviewer", "message": "ping"},
            context=ToolContext(caller_session_id=sub_session.id),
        )

        payload = json.loads(result.content)
        self.assertFalse(payload["ok"])
        self.assertEqual("star_topology_violation", payload["error"]["code"])
        self.assertIn("star topology", payload["error"]["message"])
        self.assertEqual([], self.runtime.dispatched)


class AskUserToolTests(IsolatedAsyncioTestCase):
    """验证 ask_user 的 schema 与工具侧规则。

    参数形状由 ``ASK_USER_DEFINITION.parameters`` 声明，graph 在 ``aexecute``
    之前就用它校验并直接回失败信封；工具只保留 schema 表达不了的
    strip 归一化与 recommended/options 跨字段一致性。工具本体经注入的
    broker 发起一次 :class:`AskUserRequest` 中断，把用户答复作为工具结果返回。
    """

    def setUp(self) -> None:
        self.broker = AsyncMock()
        self.broker.request.return_value = InterruptionResult(
            status=InterruptionStatus.RESOLVED, payload={"answer": "PostgreSQL"}
        )
        self.tool = AskUserTool(self.broker)
        self.context = ToolContext(caller_session_id="sess-main")

    async def test_valid_arguments_publish_request_and_return_answer(self) -> None:
        result = await self.tool.aexecute(
            {
                "question": "用哪个数据库？",
                "options": ["PostgreSQL", "SQLite"],
                "recommended": "PostgreSQL",
            },
            context=self.context,
        )

        request = self.broker.request.await_args.args[0]
        self.assertIsInstance(request, AskUserRequest)
        self.assertEqual("用哪个数据库？", request.question)
        self.assertEqual(("PostgreSQL", "SQLite"), request.options)
        self.assertEqual("PostgreSQL", request.recommended)
        self.assertEqual("sess-main", request.session_id)

        payload = json.loads(result.content)
        self.assertTrue(payload["ok"])
        self.assertEqual("PostgreSQL", payload["answer"])

    async def test_whitespace_is_trimmed_and_recommendation_omission_is_fine(self) -> None:
        await self.tool.aexecute(
            {"question": " q ", "options": [" a ", "b"]},
            context=self.context,
        )

        request = self.broker.request.await_args.args[0]
        self.assertEqual("q", request.question)
        self.assertEqual(("a", "b"), request.options)
        self.assertIsNone(request.recommended)

    async def test_unavailable_interrupt_reports_error(self) -> None:
        self.broker.request.return_value = InterruptionResult(
            status=InterruptionStatus.UNAVAILABLE
        )
        payload = await self._run(question="q", options=["a", "b"])
        self.assertFalse(payload["ok"])
        self.assertEqual("ask_user_unavailable", payload["error"]["code"])

    async def test_cancelled_interrupt_reports_error(self) -> None:
        self.broker.request.return_value = InterruptionResult(
            status=InterruptionStatus.CANCELLED
        )
        payload = await self._run(question="q", options=["a", "b"])
        self.assertFalse(payload["ok"])
        self.assertEqual("ask_user_cancelled", payload["error"]["code"])

    def test_question_and_options_shape_is_a_schema_contract(self) -> None:
        _assert_rejected_by_schema(
            self, ASK_USER_DEFINITION, {"question": "   ", "options": ["a", "b"]}
        )
        _assert_rejected_by_schema(self, ASK_USER_DEFINITION, {"question": "q"})
        for bad_options in ([], ["only-one"], ["a", "b", "c", "d", "e"], ["a", 1], ["a", " "]):
            with self.subTest(options=bad_options):
                _assert_rejected_by_schema(
                    self, ASK_USER_DEFINITION, {"question": "q", "options": bad_options}
                )
        _assert_rejected_by_schema(
            self, ASK_USER_DEFINITION, {"question": "q", "options": ["a", "a"]}
        )

    async def test_recommended_must_match_an_option(self) -> None:
        payload = await self._run(question="q", options=["a", "b"], recommended="c")
        self.assertFalse(payload["ok"])
        self.assertEqual("invalid_arguments", payload["error"]["code"])
        self.assertIn("'a'", payload["error"]["message"])
        self.assertIn("'b'", payload["error"]["message"])

    def test_definition_is_main_agent_only_and_allow_free_text(self) -> None:
        """ask_user 必须落在主控专用白名单里，且描述声明自由输入永远可用。"""
        self.assertEqual("ask_user", ASK_USER_DEFINITION.name)
        self.assertIn("ask_user", MAIN_AGENT_ONLY_TOOL_NAMES)
        self.assertEqual(
            ["question", "options"], list(ASK_USER_DEFINITION.parameters["required"])
        )
        options_schema = ASK_USER_DEFINITION.parameters["properties"]["options"]
        self.assertEqual(2, options_schema["minItems"])
        self.assertEqual(4, options_schema["maxItems"])
        self.assertEqual(
            {"type": "string", "pattern": r"\S"}, dict(options_schema["items"])
        )
        self.assertIs(True, options_schema["uniqueItems"])
        self.assertIn("Free-form answers are always possible", options_schema["description"])

    async def _run(self, **arguments: object) -> dict:
        result = await self.tool.aexecute(arguments, context=self.context)
        return json.loads(result.content)


__all__ = [
    "AskUserToolTests",
    "DefineSubagentToolTests",
    "MetaToolRegistryTests",
    "MetaToolSchemaContractTests",
    "SendMessageToolTests",
]
