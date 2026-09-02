"""Config, pipeline orchestration and report rendering.

The pipeline is exercised with stub providers so the whole run is offline and
deterministic, including the failure paths that matter most: one pair broken, all
pairs broken, no LLM key, no notification channel.
"""

from __future__ import annotations

import json

import pytest

from forex.config import Config, LLMConfig, TelegramConfig
from forex.instruments import parse_symbol
from forex.pipeline import (
    add_ai_commentary,
    build_payload,
    resolve_instruments,
    run,
    write_report,
)
from forex.providers import ProviderManager
from forex.report import render, render_markdown, telegram_summary

from tests.helpers import StubProvider

FX_ENV_VARS = (
    "FOREX_SYMBOLS", "FOREX_TIMEFRAMES", "FOREX_BARS", "FOREX_PROVIDER",
    "FOREX_OUTPUT_DIR", "FOREX_REPORT_FORMAT", "LLM_API_KEY", "OPENAI_API_KEY",
    "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY", "LLM_MODEL",
    "LLM_BASE_URL", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "LOG_LEVEL",
    "SIGNAL_TRACKING_DB", "SIGNAL_STRATEGY_VERSION", "SIGNAL_TIMEOUT_MINUTES",
    "AI_PROVIDER_ORDER", "GROQ_MODEL", "GEMINI_MODEL", "DEEPSEEK_MODEL",
)


@pytest.fixture
def clean_env(monkeypatch):
    """Remove all recognised env vars so tests never inherit the real machine."""
    for name in FX_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.fixture
def offline_config(clean_env, tmp_path):
    config = Config.from_env(dotenv_path=None)
    config.symbols = ["EURUSD", "USDJPY"]
    config.timeframes = ["H1", "D1"]
    config.bars = 120
    config.output_dir = str(tmp_path / "reports")
    return config


@pytest.fixture
def stub_manager():
    return ProviderManager([StubProvider("stub")])


class TestConfig:
    def test_defaults_are_safe_without_env(self, clean_env):
        config = Config.from_env(dotenv_path=None)
        assert config.symbols == []
        assert config.timeframes == ["H1", "H4", "D1"]
        assert config.llm.enabled is False
        assert config.telegram.enabled is False

    def test_symbols_parsed_from_env(self, clean_env):
        clean_env.setenv("FOREX_SYMBOLS", "EURUSD, GBPUSD ")
        assert Config.from_env(dotenv_path=None).symbols == ["EURUSD", "GBPUSD"]

    def test_invalid_int_falls_back_to_default(self, clean_env):
        clean_env.setenv("FOREX_BARS", "not-a-number")
        assert Config.from_env(dotenv_path=None).bars == 300

    @pytest.mark.parametrize(
        "var", ["LLM_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY"]
    )
    def test_any_vendor_key_enables_llm(self, clean_env, var):
        """Users should not have to rename their existing key."""
        clean_env.setenv(var, "sk-test")
        assert Config.from_env(dotenv_path=None).llm.enabled is True

    def test_base_url_trailing_slash_stripped(self, clean_env):
        clean_env.setenv("LLM_BASE_URL", "https://openrouter.ai/api/v1/")
        assert Config.from_env(dotenv_path=None).llm.base_url.endswith("/v1")

    def test_telegram_needs_both_values(self, clean_env):
        clean_env.setenv("TELEGRAM_BOT_TOKEN", "token")
        assert Config.from_env(dotenv_path=None).telegram.enabled is False
        clean_env.setenv("TELEGRAM_CHAT_ID", "123")
        assert Config.from_env(dotenv_path=None).telegram.enabled is True

    def test_describe_does_not_leak_secrets(self, clean_env):
        clean_env.setenv("LLM_API_KEY", "sk-super-secret-value")
        clean_env.setenv("TELEGRAM_BOT_TOKEN", "bot-secret")
        clean_env.setenv("TELEGRAM_CHAT_ID", "42")
        text = Config.from_env(dotenv_path=None).describe()
        assert "sk-super-secret-value" not in text
        assert "bot-secret" not in text

    def test_ai_candidates_use_free_first_then_paid(self, clean_env):
        clean_env.setenv("GROQ_API_KEY", "g")
        clean_env.setenv("GEMINI_API_KEY", "m")
        clean_env.setenv("DEEPSEEK_API_KEY", "d")
        candidates = LLMConfig.candidates_from_env()
        assert [item.provider for item in candidates] == ["groq", "gemini", "deepseek"]
        assert candidates[0].base_url == "https://api.groq.com/openai/v1"

    def test_dotenv_does_not_override_real_env(self, clean_env, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("FOREX_BARS=999\n", encoding="utf-8")
        monkeypatch.setenv("FOREX_BARS", "150")
        assert Config.from_env(dotenv_path=str(env_file)).bars == 150

    def test_dotenv_values_are_loaded(self, clean_env, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text('FOREX_SYMBOLS="EURUSD"\nFOREX_BARS=77\n', encoding="utf-8")
        config = Config.from_env(dotenv_path=str(env_file))
        assert config.bars == 77
        assert config.symbols == ["EURUSD"]

    def test_missing_dotenv_is_not_an_error(self, clean_env, tmp_path):
        assert Config.from_env(dotenv_path=str(tmp_path / "absent.env")).bars == 300


class TestResolveInstruments:
    def test_uses_configured_symbols(self, offline_config):
        assert [i.symbol for i in resolve_instruments(offline_config)] == ["EURUSD", "USDJPY"]

    def test_falls_back_to_default_watchlist(self, offline_config):
        offline_config.symbols = []
        assert len(resolve_instruments(offline_config)) >= 7


class TestBuildPayload:
    def test_payload_shape(self, offline_config, stub_manager):
        payload = build_payload(offline_config, manager=stub_manager)
        assert payload["summary"] == {"requested": 2, "succeeded": 2, "failed": 0}
        assert {p["symbol"] for p in payload["pairs"]} == {"EURUSD", "USDJPY"}
        assert payload["session_summary"]

    def test_each_pair_has_alignment_and_timeframes(self, offline_config, stub_manager):
        for pair in build_payload(offline_config, manager=stub_manager)["pairs"]:
            assert set(pair["timeframes"]) == {"H1", "D1"}
            assert pair["alignment"]["verdict"] in {"up", "down", "sideways", "conflicted"}

    def test_payload_is_json_serialisable(self, offline_config, stub_manager):
        payload = build_payload(offline_config, manager=stub_manager)
        assert json.loads(json.dumps(payload, default=str))["pairs"]

    def test_one_failing_pair_does_not_abort_run(self, offline_config):
        """A broken symbol must be recorded, not fatal."""
        class SelectiveStub(StubProvider):
            def fetch(self, instrument, timeframe, bars):
                if instrument.symbol == "USDJPY":
                    raise RuntimeError("simulated outage")
                return super().fetch(instrument, timeframe, bars)

        payload = build_payload(offline_config, manager=ProviderManager([SelectiveStub("sel")]))
        assert payload["summary"] == {"requested": 2, "succeeded": 1, "failed": 1}
        failed = [p for p in payload["pairs"] if p.get("error")]
        assert failed[0]["symbol"] == "USDJPY"

    def test_all_pairs_failing_yields_zero_success(self, offline_config):
        manager = ProviderManager([StubProvider("broken", fail_on=["*"])])
        payload = build_payload(offline_config, manager=manager)
        assert payload["summary"]["succeeded"] == 0
        assert all(p.get("error") for p in payload["pairs"])

    def test_price_display_uses_instrument_precision(self, offline_config, stub_manager):
        payload = build_payload(offline_config, manager=stub_manager)
        by_symbol = {p["symbol"]: p for p in payload["pairs"]}
        assert len(by_symbol["EURUSD"]["last_price_display"].split(".")[1]) == 5
        assert len(by_symbol["USDJPY"]["last_price_display"].split(".")[1]) == 3


class TestReportRendering:
    def test_markdown_contains_key_sections(self, offline_config, stub_manager):
        text = render_markdown(build_payload(offline_config, manager=stub_manager))
        for expected in ("# Daily Forex Analysis", "EUR/USD", "USD/JPY", "Timeframe", "Not investment advice"):
            assert expected in text

    def test_markdown_notes_absent_commentary(self, offline_config, stub_manager):
        payload = build_payload(offline_config, manager=stub_manager)
        payload["commentary"] = None
        assert "AI commentary tidak diminta" in render_markdown(payload)

    def test_markdown_includes_commentary_when_present(self, offline_config, stub_manager):
        payload = build_payload(offline_config, manager=stub_manager)
        payload["commentary"] = "Range-bound into the London open."
        assert "Range-bound into the London open." in render_markdown(payload)

    def test_failed_pair_gets_failures_section(self, offline_config):
        manager = ProviderManager([StubProvider("broken", fail_on=["*"])])
        text = render_markdown(build_payload(offline_config, manager=manager))
        assert "## Failures" in text
        assert "Data unavailable" in text

    def test_json_format_round_trips(self, offline_config, stub_manager):
        payload = build_payload(offline_config, manager=stub_manager)
        assert json.loads(render(payload, "json"))["summary"]["succeeded"] == 2

    def test_unsupported_format_raises(self, offline_config, stub_manager):
        with pytest.raises(ValueError, match="unsupported report format"):
            render(build_payload(offline_config, manager=stub_manager), "pdf")

    def test_telegram_summary_is_truncated(self, offline_config, stub_manager):
        payload = build_payload(offline_config, manager=stub_manager)
        payload["commentary"] = "x" * 6000
        text = telegram_summary(payload, limit=500)
        assert len(text) <= 500
        assert text.endswith("...")

    def test_telegram_summary_lists_every_pair(self, offline_config, stub_manager):
        text = telegram_summary(build_payload(offline_config, manager=stub_manager))
        assert "EUR/USD" in text and "USD/JPY" in text


class TestRunPipeline:
    def test_dry_run_writes_nothing(self, offline_config, stub_manager, tmp_path):
        import os

        result = run(offline_config, dry_run=True, manager=stub_manager)
        assert result["path"] is None
        assert result["report"]
        assert not os.path.exists(offline_config.output_dir)

    def test_dry_run_skips_llm_even_with_key(self, offline_config, stub_manager, monkeypatch):
        """A configured key must not trigger a network call during a dry run."""
        offline_config.llm = LLMConfig(api_key="sk-test")

        def explode(*args, **kwargs):
            raise AssertionError("LLM must not be called during a dry run")

        monkeypatch.setattr("forex.pipeline.generate_commentary", explode)
        assert run(offline_config, dry_run=True, manager=stub_manager)["payload"]["commentary"] is None

    def test_writes_report_file(self, offline_config, stub_manager):
        import os

        result = run(offline_config, dry_run=False, push=False, manager=stub_manager)
        assert os.path.isfile(result["path"])
        assert result["path"].endswith(".md")
        with open(result["path"], encoding="utf-8") as handle:
            assert "Daily Forex Analysis" in handle.read()

    def test_json_format_writes_json_extension(self, offline_config, stub_manager):
        offline_config.report_format = "json"
        result = run(offline_config, dry_run=False, push=False, manager=stub_manager)
        assert result["path"].endswith(".json")

    def test_llm_failure_does_not_break_report(self, offline_config, stub_manager, monkeypatch):
        """Losing optional prose must not discard a computed report."""
        offline_config.llm = LLMConfig(api_key="sk-test")
        monkeypatch.setattr("forex.pipeline.generate_commentary", lambda *a, **k: None)
        result = run(offline_config, dry_run=False, push=False, manager=stub_manager)
        assert result["payload"]["commentary"] is None
        assert "Daily Forex Analysis" in result["report"]

    def test_no_notification_when_telegram_unconfigured(self, offline_config, stub_manager):
        assert run(offline_config, dry_run=False, push=True, manager=stub_manager)["notifications"] == {}

    def test_notification_sent_when_configured(self, offline_config, stub_manager, monkeypatch):
        offline_config.telegram = TelegramConfig(bot_token="t", chat_id="1")
        sent = {}

        def fake_send(text, config, timeout=20):
            sent["text"] = text
            return True

        monkeypatch.setattr("forex.notify.send_telegram", fake_send)
        result = run(offline_config, dry_run=False, push=True, manager=stub_manager)
        assert result["notifications"] == {"telegram": True}
        assert "EUR/USD" in sent["text"]

    def test_push_false_skips_notification(self, offline_config, stub_manager, monkeypatch):
        offline_config.telegram = TelegramConfig(bot_token="t", chat_id="1")
        monkeypatch.setattr(
            "forex.notify.send_telegram",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not send")),
        )
        assert run(offline_config, dry_run=False, push=False, manager=stub_manager)["notifications"] == {}

    def test_write_report_creates_directory(self, offline_config):
        import os

        path = write_report("hello", offline_config)
        assert os.path.isfile(path)
        assert os.path.isdir(offline_config.output_dir)


class TestAIOnDemand:
    def test_default_run_does_not_call_llm(self, offline_config, stub_manager, monkeypatch):
        """AI is opt-in: a normal run with a key configured must not call the LLM."""
        offline_config.llm = LLMConfig(api_key="sk-test")

        def explode(*args, **kwargs):
            raise AssertionError("LLM must not be called unless generate=True")

        monkeypatch.setattr("forex.pipeline.generate_commentary", explode)
        result = run(offline_config, dry_run=False, push=False, manager=stub_manager)
        assert result["payload"]["commentary"] is None
        assert "Daily Forex Analysis" in result["report"]

    def test_generate_true_calls_llm(self, offline_config, stub_manager, monkeypatch):
        offline_config.llm = LLMConfig(api_key="sk-test")
        monkeypatch.setattr(
            "forex.pipeline.generate_commentary",
            lambda *a, **k: "Pasar emas dalam uptrend H4.",
        )
        result = run(
            offline_config,
            dry_run=False,
            push=False,
            manager=stub_manager,
            generate=True,
        )
        assert result["payload"]["commentary"] == "Pasar emas dalam uptrend H4."

    def test_dry_run_ignores_generate(self, offline_config, stub_manager, monkeypatch):
        offline_config.llm = LLMConfig(api_key="sk-test")

        def explode(*args, **kwargs):
            raise AssertionError("LLM must not be called during a dry run")

        monkeypatch.setattr("forex.pipeline.generate_commentary", explode)
        result = run(
            offline_config,
            dry_run=True,
            push=False,
            manager=stub_manager,
            generate=True,
        )
        assert result["payload"]["commentary"] is None

    def test_add_ai_commentary_helper(self, offline_config, stub_manager, monkeypatch):
        offline_config.llm = LLMConfig(api_key="sk-test")
        payload = build_payload(offline_config, manager=stub_manager)
        captured = {}

        def fake_generate(payload, config, metadata=None):
            if metadata is not None:
                metadata.update({"enabled": True, "provider": "test"})
            return "Ringkasan on-demand"

        monkeypatch.setattr("forex.pipeline.generate_commentary", fake_generate)
        updated = add_ai_commentary(payload, offline_config)
        assert updated["commentary"] == "Ringkasan on-demand"
        assert updated["ai"]["enabled"] is True
        assert updated["ai"]["provider"] == "test"


class TestNotifyGuards:
    def test_disabled_telegram_returns_false(self):
        from forex.notify import send_telegram

        assert send_telegram("hi", TelegramConfig()) is False

    def test_network_error_is_swallowed(self, monkeypatch):
        """A push failure must never raise into the pipeline."""
        import forex.notify as notify_module

        class FakeRequests:
            @staticmethod
            def post(*args, **kwargs):
                raise ConnectionError("no network")

        monkeypatch.setitem(__import__("sys").modules, "requests", FakeRequests)
        assert notify_module.send_telegram("hi", TelegramConfig(bot_token="t", chat_id="1")) is False

    def test_network_error_log_does_not_leak_bot_token(self, monkeypatch, caplog):
        import forex.notify as notify_module

        secret = "123456:do-not-log-this-token"

        class FakeRequests:
            @staticmethod
            def post(*args, **kwargs):
                raise ConnectionError(f"failed URL https://api.telegram.org/bot{secret}/sendMessage")

        monkeypatch.setitem(__import__("sys").modules, "requests", FakeRequests)
        assert notify_module.send_telegram("hi", TelegramConfig(bot_token=secret, chat_id="1")) is False
        assert secret not in caplog.text

    def test_tracking_adds_statistics_button(self, monkeypatch, tmp_path):
        import forex.notify as notify_module

        captured = {}

        class Response:
            status_code = 200

        class FakeRequests:
            @staticmethod
            def post(url, json, timeout):
                captured.update(json)
                return Response()

        monkeypatch.setenv("SIGNAL_TRACKING_DB", str(tmp_path / "signals.db"))
        monkeypatch.setitem(__import__("sys").modules, "requests", FakeRequests)
        assert notify_module.send_telegram("hi", TelegramConfig(bot_token="t", chat_id="1"))
        assert captured["reply_markup"]["inline_keyboard"][0][0]["callback_data"] == "ai"
        assert captured["reply_markup"]["inline_keyboard"][0][1]["callback_data"] == "stats"


class TestLLMGuards:
    def test_returns_none_without_key(self):
        from forex.llm import generate_commentary

        assert generate_commentary({"pairs": []}, LLMConfig()) is None

    def test_prompt_includes_payload_json(self):
        from forex.llm import build_user_prompt

        prompt = build_user_prompt({"pairs": [{"symbol": "EURUSD"}]})
        assert "EURUSD" in prompt and "```json" in prompt

    def test_http_error_returns_none(self, monkeypatch):
        import forex.llm as llm_module

        def explode(*args, **kwargs):
            raise llm_module.LLMError("500 from provider")

        monkeypatch.setattr(llm_module, "_post_chat_completion", explode)
        assert llm_module.generate_commentary({"pairs": []}, LLMConfig(api_key="sk-x")) is None

    def test_successful_call_returns_text(self, monkeypatch):
        import forex.llm as llm_module

        monkeypatch.setattr(llm_module, "_post_chat_completion", lambda *a, **k: "  Commentary.  ")
        assert llm_module.generate_commentary({"pairs": []}, LLMConfig(api_key="sk-x")) == "Commentary."

    def test_empty_response_becomes_none(self, monkeypatch):
        import forex.llm as llm_module

        monkeypatch.setattr(llm_module, "_post_chat_completion", lambda *a, **k: "   ")
        assert llm_module.generate_commentary({"pairs": []}, LLMConfig(api_key="sk-x")) is None

    def test_provider_failure_falls_back(self, clean_env, monkeypatch):
        import forex.llm as llm_module

        clean_env.setenv("GROQ_API_KEY", "g")
        clean_env.setenv("GEMINI_API_KEY", "m")
        attempts = []

        def fake_call(config, messages):
            attempts.append(config.provider)
            if config.provider == "groq":
                raise llm_module.LLMError("temporary")
            return "Ringkasan Gemini"

        monkeypatch.setattr(llm_module, "_post_chat_completion", fake_call)
        metadata = {}
        result = llm_module.generate_commentary(
            {"pairs": []}, LLMConfig.from_env(), metadata=metadata
        )
        assert result == "Ringkasan Gemini"
        assert attempts == ["groq", "gemini"]
        assert metadata["provider"] == "gemini"
