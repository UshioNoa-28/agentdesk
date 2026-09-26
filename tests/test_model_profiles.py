from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, TestCase

from agent.domain.model_messages import ModelMessage
from agent.domain.model_profile import ModelProfile
from agent.infrastructure.model import ModelCatalog
from agent.infrastructure.project_settings import project_settings_path

VALID = json.dumps(
    {
        "model": {
            "model-fast": {
                "name": "Fast",
                "base_url": "https://f.test/v1",
                "api_key": "sk-f",
                "context_window": 128_000,
                "max_output_tokens": 4096,
            },
            "model-smart": {
                "name": "Smart",
                "base_url": "https://s.test/v1",
                "api_key": "sk-s",
                "max_output_tokens": 8192,
                "context_window": 400_000,
                "max_retries": 1,
            },
        }
    }
)

MINIMAL = {
    "name": "n",
    "base_url": "u",
    "api_key": "k",
    "context_window": 200_000,
    "max_output_tokens": 8192,
}


def _write_settings(base: Path, payload: object) -> None:
    path = project_settings_path(base)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = payload if isinstance(payload, str) else json.dumps(payload)
    path.write_text(text, encoding="utf-8")


class ModelCatalogLoadingTests(TestCase):
    def test_absent_model_section_raises(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_settings(root, {"permissions": {"allow": ["read_file"]}})
            with self.assertRaisesRegex(ValueError, "at least one model"):
                ModelCatalog.from_root(root, home=root)

    def test_first_model_is_current(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_settings(root, VALID)
            catalog = ModelCatalog.from_root(root, home=root)

            self.assertEqual("model-fast", catalog.current_id)
            self.assertEqual("Fast", catalog.current_profile().name)
            self.assertEqual("https://f.test/v1", catalog.current_profile().base_url)
            # 列出的 id 顺序 = 发现顺序，展示名走 name 字段。
            self.assertEqual(
                [("model-fast", "Fast"), ("model-smart", "Smart")],
                [(i.model_id, i.name) for i in catalog.list_models()],
            )

    def test_optional_profile_fields_default_when_absent(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fields = {k: v for k, v in MINIMAL.items() if k != "max_retries"}
            _write_settings(root, {"model": {"m": fields}})
            profile = ModelCatalog.from_root(root, home=root).current_profile()

            self.assertEqual("m", profile.model_id)
            self.assertEqual(8192, profile.max_output_tokens)
            self.assertEqual(200_000, profile.context_window)
            self.assertEqual(2, profile.max_retries)

    def test_missing_keys_take_profile_defaults(self) -> None:
        """不校验：缺什么键吃什么字段的默认值，空对象也能启动。"""

        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_settings(root, {"model": {"m": {}}})
            profile = ModelCatalog.from_root(root, home=root).current_profile()

            self.assertEqual("m", profile.model_id)
            self.assertEqual(200_000, profile.context_window)
            self.assertEqual(8_192, profile.max_output_tokens)
            self.assertEqual("", profile.api_key)

    def test_unknown_fields_are_ignored(self) -> None:
        """未知键（老的嵌套 profile、策略字段、无关参数）一律静默忽略。"""

        for key, value in (
            ("profile", {"max_retries": 5}),
            ("compact_threshold_ratio", 1.5),
            ("temperature", 2),
        ):
            with self.subTest(key=key), TemporaryDirectory() as directory:
                root = Path(directory)
                fields = {**MINIMAL, key: value}
                _write_settings(root, {"model": {"m": fields}})
                catalog = ModelCatalog.from_root(root, home=root)

                self.assertEqual(200_000, catalog.current_profile().context_window)
                # 被忽略的键不得改写已知字段（max_retries 仍是默认 2）。
                self.assertEqual(2, catalog.current_profile().max_retries)

    def test_empty_model_section_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_settings(root, {"model": {}})
            with self.assertRaisesRegex(ValueError, "at least one model"):
                ModelCatalog.from_root(root, home=root)

    def test_nearer_layer_shadows_same_model_id(self) -> None:
        with TemporaryDirectory() as directory:
            home = Path(directory) / "home"
            project = home / "work" / "repo"
            project.mkdir(parents=True)
            _write_settings(
                project,
                {
                    "model": {
                        "model-fast": {
                            "name": "Project",
                            "base_url": "u",
                            "api_key": "k",
                            "context_window": 200_000,
                            "max_output_tokens": 8192,
                        }
                    }
                },
            )
            _write_settings(
                home,
                {
                    "model": {
                        "model-fast": {
                            "name": "Home",
                            "base_url": "u",
                            "api_key": "k",
                            "context_window": 200_000,
                            "max_output_tokens": 8192,
                        },
                        "model-far": {
                            "name": "Far",
                            "base_url": "u",
                            "api_key": "k",
                            "context_window": 200_000,
                            "max_output_tokens": 8192,
                        },
                    }
                },
            )

            catalog = ModelCatalog.from_root(project, home=home)

            self.assertEqual("Project", catalog.current_profile().name)
            self.assertEqual(
                ["model-fast", "model-far"], [i.model_id for i in catalog.list_models()]
            )


class ModelCatalogSelectionTests(TestCase):
    def _catalog(self) -> ModelCatalog:
        return ModelCatalog(
            [
                (
                    "model-fast",
                    ModelProfile(
                        model_id="model-fast",
                        name="Fast",
                        base_url="u",
                        api_key="k",
                        context_window=128_000,
                        max_output_tokens=4_096,
                    ),
                ),
                (
                    "model-smart",
                    ModelProfile(
                        model_id="model-smart",
                        name="Smart",
                        base_url="u",
                        api_key="k",
                        context_window=400_000,
                        max_output_tokens=8192,
                    ),
                ),
            ]
        )

    def test_select_moves_current(self) -> None:
        catalog = self._catalog()
        self.assertEqual("model-smart", catalog.select_model("model-smart"))
        self.assertEqual("model-smart", catalog.current_id)
        self.assertEqual("Smart", catalog.current_profile().name)

    def test_current_profile_follows_selection(self) -> None:
        catalog = self._catalog()
        self.assertEqual(128_000, catalog.current_profile().context_window)
        catalog.select_model("model-smart")
        self.assertEqual(400_000, catalog.current_profile().context_window)
        self.assertEqual(8192, catalog.current_profile().max_output_tokens)

    def test_unknown_select_raises_and_keeps_current(self) -> None:
        catalog = self._catalog()
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            catalog.select_model("ghost")
        self.assertEqual("model-fast", catalog.current_id)

    def test_empty_catalog_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one model"):
            ModelCatalog([])

    def test_listing_never_carries_credentials(self) -> None:
        listing = self._catalog().list_models()
        self.assertEqual(
            {"model_id", "name"},
            {field.name for field in dataclasses.fields(listing[0])},
        )


class _RecordingCatalog:
    """每次 current_profile() 依次吐出给定 model_id，用于验证"每轮现取"。"""

    def __init__(self, ids: Sequence[str]) -> None:
        self._ids = list(ids)
        self.pulled = 0

    def current_profile(self) -> ModelProfile:
        model_id = self._ids[min(self.pulled, len(self._ids) - 1)]
        self.pulled += 1
        return ModelProfile(
            model_id=model_id,
            name=model_id,
            base_url="u",
            api_key="k",
            context_window=200_000,
        )


class LangChainAgentModelTurnTests(IsolatedAsyncioTestCase):
    async def test_reads_current_model_each_turn(self) -> None:
        from unittest.mock import AsyncMock, Mock, patch

        from agent.infrastructure.model import chat_model as cm

        built: list[str] = []

        class _FakeChatOpenAI:
            configs: list[object] = []

            def __init__(self, **kwargs: object) -> None:
                built.append(str(kwargs.get("model")))

            async def ainvoke(self, _messages: object, config=None):  # noqa: ANN201, ANN202
                _FakeChatOpenAI.configs.append(config)

                class _Resp:
                    content = "ok"
                    tool_calls: list = []
                    usage_metadata = None

                return _Resp()

        catalog = _RecordingCatalog(["model-a", "model-b"])
        model = cm.LangChainAgentModel(
            catalog=catalog,
            tool_adapter=Mock(),
        )
        with (
            patch.object(cm, "ChatOpenAI", _FakeChatOpenAI),
            patch.object(cm, "build_provider_async_http_client", return_value=AsyncMock()),
        ):
            await model.ainvoke(messages=[ModelMessage.human("one")])
            await model.ainvoke(messages=[ModelMessage.human("two")])

        # 每轮向 catalog 现取模型：两次调用分别落到 model-a、model-b（用 id 发送）。
        self.assertEqual(["model-a", "model-b"], built)
        # 默认 CHAT 调用不带任何 run config——正常参与用户可见流。
        self.assertEqual([None, None], _FakeChatOpenAI.configs)

        await model.aclose()


class LangChainAgentModelStreamTagTests(IsolatedAsyncioTestCase):
    """purpose 到流抑制 tag 的映射：摘要增量不得进入 messages 流。"""

    async def _capture_configs(self, **purpose_kwargs: object) -> list[object]:
        from unittest.mock import AsyncMock, Mock, patch

        from agent.infrastructure.model import chat_model as cm

        configs: list[object] = []

        class _FakeChatOpenAI:
            def __init__(self, **_kwargs: object) -> None:
                pass

            async def ainvoke(self, _messages: object, config=None):  # noqa: ANN201, ANN202
                configs.append(config)

                class _Resp:
                    content = "ok"
                    tool_calls: list = []
                    usage_metadata = None

                return _Resp()

        model = cm.LangChainAgentModel(
            catalog=_RecordingCatalog(["model-a"]),
            tool_adapter=Mock(),
        )
        with (
            patch.object(cm, "ChatOpenAI", _FakeChatOpenAI),
            patch.object(cm, "build_provider_async_http_client", return_value=AsyncMock()),
        ):
            await model.ainvoke(messages=[ModelMessage.human("x")], **purpose_kwargs)
        await model.aclose()
        return configs

    async def test_chat_purpose_carries_no_suppression_tag(self) -> None:
        from agent.ports.model import ModelCallPurpose

        configs = await self._capture_configs(purpose=ModelCallPurpose.CHAT)
        self.assertEqual([None], configs)

    async def test_summary_purpose_tags_run_nostream(self) -> None:
        from agent.ports.model import ModelCallPurpose

        configs = await self._capture_configs(purpose=ModelCallPurpose.SUMMARY)
        self.assertEqual([{"tags": ["nostream"]}], configs)


__all__ = [
    "LangChainAgentModelStreamTagTests",
    "LangChainAgentModelTurnTests",
    "ModelCatalogLoadingTests",
    "ModelCatalogSelectionTests",
]
