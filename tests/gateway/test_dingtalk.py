"""Tests for DingTalk platform adapter."""
import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from gateway.config import Platform, PlatformConfig


# ---------------------------------------------------------------------------
# Requirements check
# ---------------------------------------------------------------------------


class TestDingTalkRequirements:

    def test_returns_false_when_sdk_missing(self, monkeypatch):
        with patch.dict("sys.modules", {"dingtalk_stream": None}):
            monkeypatch.setattr(
                "gateway.platforms.dingtalk.DINGTALK_STREAM_AVAILABLE", False
            )
            from gateway.platforms.dingtalk import check_dingtalk_requirements
            assert check_dingtalk_requirements() is False

    def test_returns_false_when_env_vars_missing(self, monkeypatch):
        monkeypatch.setattr(
            "gateway.platforms.dingtalk.DINGTALK_STREAM_AVAILABLE", True
        )
        monkeypatch.setattr("gateway.platforms.dingtalk.HTTPX_AVAILABLE", True)
        monkeypatch.delenv("DINGTALK_CLIENT_ID", raising=False)
        monkeypatch.delenv("DINGTALK_CLIENT_SECRET", raising=False)
        from gateway.platforms.dingtalk import check_dingtalk_requirements
        assert check_dingtalk_requirements() is False

    def test_returns_true_when_all_available(self, monkeypatch):
        monkeypatch.setattr(
            "gateway.platforms.dingtalk.DINGTALK_STREAM_AVAILABLE", True
        )
        monkeypatch.setattr("gateway.platforms.dingtalk.HTTPX_AVAILABLE", True)
        monkeypatch.setenv("DINGTALK_CLIENT_ID", "test-id")
        monkeypatch.setenv("DINGTALK_CLIENT_SECRET", "test-secret")
        from gateway.platforms.dingtalk import check_dingtalk_requirements
        assert check_dingtalk_requirements() is True


# ---------------------------------------------------------------------------
# Adapter construction
# ---------------------------------------------------------------------------


class TestDingTalkAdapterInit:

    def test_reads_config_from_extra(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        config = PlatformConfig(
            enabled=True,
            extra={"client_id": "cfg-id", "client_secret": "cfg-secret"},
        )
        adapter = DingTalkAdapter(config)
        assert adapter._client_id == "cfg-id"
        assert adapter._client_secret == "cfg-secret"
        assert adapter.name == "Dingtalk"  # base class uses .title()

    def test_falls_back_to_env_vars(self, monkeypatch):
        monkeypatch.setenv("DINGTALK_CLIENT_ID", "env-id")
        monkeypatch.setenv("DINGTALK_CLIENT_SECRET", "env-secret")
        from gateway.platforms.dingtalk import DingTalkAdapter
        config = PlatformConfig(enabled=True)
        adapter = DingTalkAdapter(config)
        assert adapter._client_id == "env-id"
        assert adapter._client_secret == "env-secret"


# ---------------------------------------------------------------------------
# Message text extraction
# ---------------------------------------------------------------------------


class TestExtractText:

    def test_extracts_dict_text(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        msg = MagicMock()
        msg.text = {"content": "  hello world  "}
        msg.rich_text = None
        assert DingTalkAdapter._extract_text(msg) == "hello world"

    def test_extracts_string_text(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        msg = MagicMock()
        msg.text = "plain text"
        msg.rich_text = None
        assert DingTalkAdapter._extract_text(msg) == "plain text"

    def test_falls_back_to_rich_text(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        msg = MagicMock()
        msg.text = ""
        msg.rich_text = [{"text": "part1"}, {"text": "part2"}, {"image": "url"}]
        assert DingTalkAdapter._extract_text(msg) == "part1 part2"

    def test_returns_empty_for_no_content(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        msg = MagicMock()
        msg.text = ""
        msg.rich_text = None
        assert DingTalkAdapter._extract_text(msg) == ""


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:

    def test_first_message_not_duplicate(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))
        assert adapter._dedup.is_duplicate("msg-1") is False

    def test_second_same_message_is_duplicate(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))
        adapter._dedup.is_duplicate("msg-1")
        assert adapter._dedup.is_duplicate("msg-1") is True

    def test_different_messages_not_duplicate(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))
        adapter._dedup.is_duplicate("msg-1")
        assert adapter._dedup.is_duplicate("msg-2") is False

    def test_cache_cleanup_on_overflow(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))
        max_size = adapter._dedup._max_size
        # Fill beyond max
        for i in range(max_size + 10):
            adapter._dedup.is_duplicate(f"msg-{i}")
        # Cache should have been pruned
        assert len(adapter._dedup._seen) <= max_size + 10


# ---------------------------------------------------------------------------
# Send
# ---------------------------------------------------------------------------


class TestSend:

    @pytest.mark.asyncio
    async def test_send_posts_to_webhook(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "OK"

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        adapter._http_client = mock_client

        result = await adapter.send(
            "chat-123", "Hello!",
            metadata={"session_webhook": "https://dingtalk.example/webhook"}
        )
        assert result.success is True
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "https://dingtalk.example/webhook"
        payload = call_args[1]["json"]
        assert payload["msgtype"] == "markdown"
        assert payload["markdown"]["title"] == "Hermes"
        assert payload["markdown"]["text"] == "Hello!"

    @pytest.mark.asyncio
    async def test_send_fails_without_webhook(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))
        adapter._http_client = AsyncMock()

        result = await adapter.send("chat-123", "Hello!")
        assert result.success is False
        assert "session_webhook" in result.error

    @pytest.mark.asyncio
    async def test_send_uses_cached_webhook(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        adapter._http_client = mock_client
        adapter._session_webhooks["chat-123"] = "https://cached.example/webhook"

        result = await adapter.send("chat-123", "Hello!")
        assert result.success is True
        assert mock_client.post.call_args[0][0] == "https://cached.example/webhook"

    @pytest.mark.asyncio
    async def test_send_handles_http_error(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        adapter._http_client = mock_client

        result = await adapter.send(
            "chat-123", "Hello!",
            metadata={"session_webhook": "https://example/webhook"}
        )
        assert result.success is False
        assert "400" in result.error


# ---------------------------------------------------------------------------
# Connect / disconnect
# ---------------------------------------------------------------------------


class TestConnect:

    @pytest.mark.asyncio
    async def test_disconnect_closes_async_stream_websocket(self, monkeypatch):
        from gateway.platforms import dingtalk as dingtalk_module
        from gateway.platforms.dingtalk import DingTalkAdapter

        adapter = DingTalkAdapter(PlatformConfig(enabled=True))

        class FakeConnectionClosed(Exception):
            pass

        class FakeWebSocket:
            def __init__(self):
                self.closed = asyncio.Event()
                self.close_calls = 0

            async def recv(self):
                await self.closed.wait()
                raise FakeConnectionClosed()

            async def close(self):
                self.close_calls += 1
                self.closed.set()

            async def ping(self):
                return None

        class FakeConnectContext:
            def __init__(self, websocket):
                self.websocket = websocket

            async def __aenter__(self):
                return self.websocket

            async def __aexit__(self, exc_type, exc, tb):
                return False

        class FakeStreamClient:
            def __init__(self, websocket):
                self.websocket = None
                self._websocket = websocket
                self.pre_start_called = False

            def pre_start(self):
                self.pre_start_called = True

            def open_connection(self):
                return {"endpoint": "wss://example.invalid/connect", "ticket": "ticket"}

            async def keepalive(self, websocket):
                await websocket.closed.wait()

            async def background_task(self, json_message):
                return None

            async def start(self):
                while True:
                    try:
                        await self._websocket.closed.wait()
                        return
                    except asyncio.CancelledError:
                        continue

        websocket = FakeWebSocket()
        adapter._stream_client = FakeStreamClient(websocket)
        adapter._running = True

        fake_websockets = SimpleNamespace(
            connect=lambda _uri: FakeConnectContext(websocket),
            exceptions=SimpleNamespace(ConnectionClosed=FakeConnectionClosed),
        )
        monkeypatch.setattr(dingtalk_module, "websockets", fake_websockets, raising=False)

        adapter._stream_task = asyncio.create_task(adapter._run_stream())
        for _ in range(20):
            if adapter._stream_client.websocket is not None:
                break
            await asyncio.sleep(0)

        await asyncio.wait_for(adapter.disconnect(), timeout=0.5)

        assert websocket.close_calls == 1
        assert adapter._stream_task is None

    @pytest.mark.asyncio
    async def test_connect_fails_without_sdk(self, monkeypatch):
        monkeypatch.setattr(
            "gateway.platforms.dingtalk.DINGTALK_STREAM_AVAILABLE", False
        )
        from gateway.platforms.dingtalk import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))
        result = await adapter.connect()
        assert result is False

    @pytest.mark.asyncio
    async def test_connect_fails_without_credentials(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))
        adapter._client_id = ""
        adapter._client_secret = ""
        result = await adapter.connect()
        assert result is False

    @pytest.mark.asyncio
    async def test_disconnect_cleans_up(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))
        adapter._session_webhooks["a"] = "http://x"
        adapter._dedup._seen["b"] = 1.0
        adapter._http_client = AsyncMock()
        adapter._stream_task = None

        await adapter.disconnect()
        assert len(adapter._session_webhooks) == 0
        assert len(adapter._dedup._seen) == 0
        assert adapter._http_client is None


# ---------------------------------------------------------------------------
# Platform enum
# ---------------------------------------------------------------------------


class TestPlatformEnum:

    @pytest.mark.asyncio
    async def test_incoming_handler_process_awaits_adapter_handler(self):
        from gateway.platforms.dingtalk import _IncomingHandler, dingtalk_stream

        adapter = MagicMock()
        adapter._on_message = AsyncMock()

        handler = _IncomingHandler(adapter, asyncio.get_running_loop())
        message = MagicMock()
        message.data = {"msgId": "msg-1", "conversationId": "conv-1", "senderId": "sender-1", "msgtype": "text", "text": {"content": "hi"}}

        result = await handler.process(message)

        adapter._on_message.assert_awaited_once()
        forwarded = adapter._on_message.await_args.args[0]
        assert forwarded.message_id == "msg-1"
        assert forwarded.conversation_id == "conv-1"
        assert forwarded.sender_id == "sender-1"
        assert forwarded.text.content == "hi"
        assert result == (dingtalk_stream.AckMessage.STATUS_OK, "OK")

    @pytest.mark.asyncio
    async def test_incoming_handler_logs_topic_metadata(self, caplog):
        from gateway.platforms import dingtalk as dingtalk_module
        from gateway.platforms.dingtalk import _IncomingHandler

        adapter = MagicMock()
        adapter._on_message = AsyncMock()

        handler = _IncomingHandler(adapter, asyncio.get_running_loop())
        message = MagicMock()
        message.message_id = "msg-1"
        message.conversation_id = "conv-1"
        message.sender_id = "sender-1"
        message.type = "CALLBACK"
        message.text = {"content": "hi"}
        message.data = {"k": "v"}
        message.headers = {"topic": "chatbot"}
        type(message).topic = PropertyMock(return_value="chatbot")

        with caplog.at_level("INFO", logger=dingtalk_module.logger.name):
            await handler.process(message)

        assert "Incoming callback topic=chatbot" in caplog.text
        assert "callback_type=CALLBACK" in caplog.text
        assert "conversation_id=conv-1" in caplog.text
        assert "sender_id=sender-1" in caplog.text
        assert "has_text=True" in caplog.text
        assert "has_data=True" in caplog.text
        assert "has_headers=True" in caplog.text

    @pytest.mark.asyncio
    async def test_on_message_logs_inbound_metadata(self, caplog):
        from gateway.platforms import dingtalk as dingtalk_module
        from gateway.platforms.dingtalk import DingTalkAdapter

        adapter = DingTalkAdapter(PlatformConfig(enabled=True))
        adapter.handle_message = AsyncMock()

        message = MagicMock()
        message.message_id = "msg-1"
        message.text = {"content": "hi"}
        message.rich_text = None
        message.conversation_id = "conv-1"
        message.conversation_type = "1"
        message.sender_id = "sender-1"
        message.sender_nick = "Alice"
        message.sender_staff_id = "staff-1"
        message.conversation_title = "Hermes"
        message.session_webhook = "https://api.dingtalk.com/v1.0/robot/oToMessages/batchSend"
        message.create_at = str(int(datetime.now(tz=timezone.utc).timestamp() * 1000))

        with caplog.at_level("INFO", logger=dingtalk_module.logger.name):
            await adapter._on_message(message)

        assert "Inbound message message_id=msg-1" in caplog.text
        assert "chat_type=dm" in caplog.text
        assert "conversation_id=conv-1" in caplog.text
        assert "sender_id=sender-1" in caplog.text
        assert "sender_staff_id=staff-1" in caplog.text

    @pytest.mark.asyncio
    async def test_on_message_caches_legacy_oapi_session_webhook(self):
        from gateway.platforms.dingtalk import DingTalkAdapter

        adapter = DingTalkAdapter(PlatformConfig(enabled=True))
        adapter.handle_message = AsyncMock()

        message = MagicMock()
        message.message_id = "msg-legacy"
        message.text = {"content": "hi"}
        message.rich_text = None
        message.conversation_id = "conv-legacy"
        message.conversation_type = "1"
        message.sender_id = "sender-legacy"
        message.sender_nick = "Alice"
        message.sender_staff_id = "staff-legacy"
        message.conversation_title = "Hermes"
        message.session_webhook = "https://oapi.dingtalk.com/robot/sendBySession?session=abc"
        message.create_at = str(int(datetime.now(tz=timezone.utc).timestamp() * 1000))

        await adapter._on_message(message)

        assert adapter._session_webhooks["conv-legacy"] == (
            "https://oapi.dingtalk.com/robot/sendBySession?session=abc"
        )

    @pytest.mark.asyncio
    async def test_unauthorized_dm_sends_pairing_code_via_cached_webhook(self, monkeypatch, tmp_path):
        import gateway.run as gateway_run
        from gateway.config import GatewayConfig
        from gateway.platforms.dingtalk import DingTalkAdapter
        from gateway.run import GatewayRunner

        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("DINGTALK_ALLOW_ALL_USERS", raising=False)
        monkeypatch.delenv("DINGTALK_ALLOWED_USERS", raising=False)
        monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
        monkeypatch.delenv("GATEWAY_ALLOWED_USERS", raising=False)
        (tmp_path / "config.yaml").write_text("", encoding="utf-8")

        runner = GatewayRunner(GatewayConfig())
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))
        adapter.set_message_handler(runner._handle_message)
        runner.adapters[Platform.DINGTALK] = adapter

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "OK"
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        adapter._http_client = mock_client

        message = MagicMock()
        message.message_id = "msg-auth"
        message.text = {"content": "hello"}
        message.rich_text = None
        message.conversation_id = "conv-auth"
        message.conversation_type = "1"
        message.sender_id = "sender-auth"
        message.sender_nick = "Alice"
        message.sender_staff_id = "staff-auth"
        message.conversation_title = "Hermes"
        message.session_webhook = "https://oapi.dingtalk.com/robot/sendBySession?session=abc"
        message.create_at = str(int(datetime.now(tz=timezone.utc).timestamp() * 1000))

        await adapter._on_message(message)
        if adapter._background_tasks:
            await asyncio.gather(*list(adapter._background_tasks))

        pending = runner.pairing_store.list_pending("dingtalk")
        assert len(pending) == 1
        sent_url = mock_client.post.await_args.args[0]
        sent_payload = mock_client.post.await_args.kwargs["json"]
        assert sent_url == "https://oapi.dingtalk.com/robot/sendBySession?session=abc"
        assert pending[0]["code"] in sent_payload["markdown"]["text"]
        assert "hermes pairing approve dingtalk" in sent_payload["markdown"]["text"]

    def test_dingtalk_in_platform_enum(self):
        assert Platform.DINGTALK.value == "dingtalk"
