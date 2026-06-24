from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from worldcup_prediction.data_sources import SampleDataSource
from worldcup_prediction.engine import run_predict, run_review
from worldcup_prediction.ledger import LedgerManager
from worldcup_prediction.providers import DEFAULT_USER_AGENT, OpenAICompatibleProvider, ProviderError, get_provider


class WorldCupPredictionTests(unittest.TestCase):
    def test_sample_data_source_loads_matches(self):
        snapshot = SampleDataSource().load_matches("2026-06-23", "Asia/Shanghai")
        self.assertGreaterEqual(len(snapshot.matches), 1)
        self.assertEqual(snapshot.matches[0].match_id, "wc2026-043-arg-aut")
        self.assertEqual(snapshot.matches[0].home_team, "阿根廷")
        self.assertEqual(snapshot.sources[0].source_id, "schedule_guardian_2026_06_22")

    def test_predict_generates_required_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = run_predict(
                date="2026-06-23",
                source="sample",
                timezone_name="Asia/Shanghai",
                llm_provider="mock",
                project_root=tmp_path,
            )
            self.assertTrue((run_dir / "daily_report.md").exists())
            self.assertTrue((run_dir / "matches.json").exists())
            self.assertTrue((run_dir / "predictions.json").exists())
            self.assertTrue((run_dir / "sources.json").exists())
            self.assertTrue((run_dir / "run.log").exists())
            predictions = json.loads((run_dir / "predictions.json").read_text(encoding="utf-8"))
            first_match_id = predictions["matches"][0]["match_id"]
            self.assertTrue((run_dir / "agents" / first_match_id / "data_analyst.md").exists())
            self.assertTrue((run_dir / "agents" / first_match_id / "summarizer.md").exists())

            self.assertEqual(len(predictions["predictions"]), len(predictions["matches"]))
            self.assertEqual(len(predictions["role_analyses"]), len(predictions["matches"]) * 4)

    def test_per_match_provider_failure_still_generates_daily_report(self):
        class FailsFirstMatchProvider:
            name = "fails-first"
            supports_daily_batch = False

            def __init__(self):
                self.calls = 0

            def generate_role_analysis(self, role, match, sources, lessons):
                self.calls += 1
                if match.match_id == "sample-01-BRA-CRO":
                    raise ProviderError("simulated timeout")
                return type(
                    "Result",
                    (),
                    {
                        "payload": {
                            "role": role,
                            "match_id": match.match_id,
                            "predicted_outcome": "HOME_WIN",
                            "conclusion": "ok",
                            "confidence": "medium",
                            "key_evidence": ["ok"],
                            "risks": [],
                            "citations": [],
                            "one_line": "ok",
                        },
                        "raw_text": "{}",
                        "parse_error": None,
                    },
                )()

            def generate_summary(self, match, role_payloads, sources, lessons):
                return type(
                    "Result",
                    (),
                    {
                        "payload": {
                            "match_id": match.match_id,
                            "predicted_outcome": "HOME_WIN",
                            "confidence": "medium",
                            "key_evidence": ["ok"],
                            "max_risk": "none",
                            "role_summaries": {"data_analyst": "ok"},
                            "agreements": ["ok"],
                            "conflicts": [],
                            "decision": "ok",
                            "citations": [],
                        },
                        "raw_text": "{}",
                        "parse_error": None,
                    },
                )()

        from worldcup_prediction import engine

        original_get_provider = engine.get_provider
        try:
            engine.get_provider = lambda _name: FailsFirstMatchProvider()
            with tempfile.TemporaryDirectory() as tmp:
                run_dir = run_predict(
                    date="2026-06-24",
                    source="sample",
                    timezone_name="Asia/Shanghai",
                    llm_provider="auto",
                    project_root=Path(tmp),
                )
                predictions = json.loads((run_dir / "predictions.json").read_text(encoding="utf-8"))
                self.assertEqual(len(predictions["matches"]), 3)
                self.assertEqual(len(predictions["predictions"]), 3)
                self.assertEqual(predictions["predictions"][0]["parse_error"], "simulated timeout")
                self.assertTrue((run_dir / "daily_report.md").exists())
        finally:
            engine.get_provider = original_get_provider

    def test_review_computes_hits_and_misses(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_predict(
                date="2026-06-23",
                source="sample",
                timezone_name="Asia/Shanghai",
                llm_provider="mock",
                project_root=tmp_path,
            )
            results_path = tmp_path / "results.json"
            results_path.write_text(
                json.dumps(
                    {
                        "results": [
                            {"match_id": "wc2026-043-arg-aut", "home_score": 2, "away_score": 0},
                            {"match_id": "wc2026-042-fra-irq", "home_score": 3, "away_score": 0},
                            {"match_id": "wc2026-041-nor-sen", "home_score": 3, "away_score": 2},
                            {"match_id": "wc2026-044-jor-alg", "home_score": 1, "away_score": 2},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            summary = run_review("2026-06-23", results_path, project_root=tmp_path)
            self.assertEqual(summary.reviewed, 4)
            self.assertGreaterEqual(summary.hits, 1)
            self.assertGreaterEqual(summary.misses, 0)
            self.assertTrue((tmp_path / "ledger" / "results_ledger.jsonl").exists())
            lessons = (tmp_path / "ledger" / "lessons.md").read_text(encoding="utf-8")
            self.assertTrue("Lesson:" in lessons or "All reviewed predictions hit" in lessons)

    def test_lessons_are_loaded_into_next_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ledger = LedgerManager(tmp_path)
            ledger.ledger_dir.mkdir(parents=True, exist_ok=True)
            ledger.lessons_path.write_text(
                "## prior review\n\n- Low-block teams can force a draw; do not underestimate 逼平 probability.\n",
                encoding="utf-8",
            )

            run_dir = run_predict(
                date="2026-06-24",
                source="sample",
                timezone_name="Asia/Shanghai",
                llm_provider="mock",
                project_root=tmp_path,
            )
            report = (run_dir / "daily_report.md").read_text(encoding="utf-8")
            predictions = json.loads((run_dir / "predictions.json").read_text(encoding="utf-8"))
            self.assertIn("逼平", report)
            self.assertTrue(predictions["lessons_used"])

    def test_openai_compatible_requests_include_user_agent(self):
        provider = OpenAICompatibleProvider(
            api_key="test-key",
            model="gpt-5.5",
            base_url="https://code.ddsst.online/v1",
            wire_api="responses",
        )

        request = provider._request("/responses", {"model": "gpt-5.5"})

        self.assertEqual(request.get_header("User-agent"), DEFAULT_USER_AGENT)
        self.assertEqual(request.get_header("Content-type"), "application/json")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")

    def test_provider_user_agent_can_be_overridden(self):
        original_env = os.environ.copy()
        try:
            os.environ.clear()
            os.environ.update(
                {
                    "DDSS_API_KEY": "test-key",
                    "WCP_OPENAI_USER_AGENT": "custom-agent/1.0",
                }
            )

            provider = get_provider("auto")

            self.assertIsInstance(provider, OpenAICompatibleProvider)
            self.assertEqual(provider.user_agent, "custom-agent/1.0")
        finally:
            os.environ.clear()
            os.environ.update(original_env)

    def test_provider_reads_local_config_file(self):
        original_env = os.environ.copy()
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "ddss.env"
            config_path.write_text(
                "\n".join(
                    [
                        "DDSS_API_KEY=file-key",
                        "WCP_OPENAI_MODEL=gpt-5.5",
                        "WCP_OPENAI_BASE_URL=https://code.ddsst.online/v1",
                        "WCP_OPENAI_WIRE_API=responses",
                        "WCP_OPENAI_USER_AGENT=file-agent/1.0",
                    ]
                ),
                encoding="utf-8",
            )
            try:
                os.environ.clear()
                os.environ["WCP_CONFIG_ENV_PATH"] = str(config_path)

                provider = get_provider("auto")

                self.assertIsInstance(provider, OpenAICompatibleProvider)
                self.assertEqual(provider.api_key, "file-key")
                self.assertEqual(provider.model, "gpt-5.5")
                self.assertEqual(provider.base_url, "https://code.ddsst.online/v1")
                self.assertEqual(provider.wire_api, "responses")
                self.assertEqual(provider.user_agent, "file-agent/1.0")
                self.assertEqual(provider.reasoning_effort, "high")
                self.assertFalse(provider.supports_daily_batch)
            finally:
                os.environ.clear()
                os.environ.update(original_env)

    def test_responses_request_includes_reasoning_effort_when_configured(self):
        provider = OpenAICompatibleProvider(
            api_key="test-key",
            model="gpt-5.5",
            base_url="https://code.ddsst.online/v1",
            wire_api="responses",
            reasoning_effort="high",
        )

        body = provider._responses_body("Return {}")

        self.assertEqual(body["reasoning"], {"effort": "high"})

    def test_ddss_reasoning_effort_is_fixed_to_high(self):
        original_env = os.environ.copy()
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "ddss.env"
            config_path.write_text(
                "\n".join(
                    [
                        "DDSS_API_KEY=file-key",
                        "WCP_OPENAI_MODEL=gpt-5.5",
                        "WCP_OPENAI_BASE_URL=https://code.ddsst.online/v1",
                        "WCP_OPENAI_WIRE_API=responses",
                        "WCP_OPENAI_REASONING_EFFORT=ultra",
                    ]
                ),
                encoding="utf-8",
            )
            try:
                os.environ.clear()
                os.environ["WCP_CONFIG_ENV_PATH"] = str(config_path)

                provider = get_provider("ddss")

                self.assertEqual(provider.reasoning_effort, "high")
            finally:
                os.environ.clear()
                os.environ.update(original_env)

    def test_auto_provider_prefers_ddss_over_qwen_local_config(self):
        original_env = os.environ.copy()
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "combined.env"
            config_path.write_text(
                "\n".join(
                    [
                        "DDSS_API_KEY=ddss-key",
                        "WCP_OPENAI_MODEL=gpt-5.5",
                        "WCP_OPENAI_BASE_URL=https://code.ddsst.online/v1",
                        "WCP_OPENAI_WIRE_API=responses",
                        "AUTODL_API_KEY=qwen-key",
                        "AUTODL_MODEL=Qwen3.5-397B-A17B",
                        "AUTODL_BASE_URL=https://www.autodl.art/api/v1",
                    ]
                ),
                encoding="utf-8",
            )
            try:
                os.environ.clear()
                os.environ["WCP_CONFIG_ENV_PATH"] = str(config_path)

                provider = get_provider("auto")

                self.assertIsInstance(provider, OpenAICompatibleProvider)
                self.assertEqual(provider.api_key, "ddss-key")
                self.assertEqual(provider.model, "gpt-5.5")
                self.assertEqual(provider.base_url, "https://code.ddsst.online/v1")
                self.assertEqual(provider.wire_api, "responses")
            finally:
                os.environ.clear()
                os.environ.update(original_env)

    def test_qwen_provider_uses_autodl_config_without_ddss_wire_api(self):
        original_env = os.environ.copy()
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "combined.env"
            config_path.write_text(
                "\n".join(
                    [
                        "DDSS_API_KEY=ddss-key",
                        "WCP_OPENAI_WIRE_API=responses",
                        "AUTODL_API_KEY=qwen-key",
                        "AUTODL_MODEL=Qwen3.5-397B-A17B",
                        "AUTODL_BASE_URL=https://www.autodl.art/api/v1",
                    ]
                ),
                encoding="utf-8",
            )
            try:
                os.environ.clear()
                os.environ["WCP_CONFIG_ENV_PATH"] = str(config_path)

                provider = get_provider("qwen")

                self.assertIsInstance(provider, OpenAICompatibleProvider)
                self.assertEqual(provider.api_key, "qwen-key")
                self.assertEqual(provider.model, "Qwen3.5-397B-A17B")
                self.assertEqual(provider.base_url, "https://www.autodl.art/api/v1")
                self.assertEqual(provider.wire_api, "chat_completions")
            finally:
                os.environ.clear()
                os.environ.update(original_env)

    def test_environment_overrides_local_config_file(self):
        original_env = os.environ.copy()
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "ddss.env"
            config_path.write_text(
                "DDSS_API_KEY=file-key\nWCP_OPENAI_MODEL=gpt-5.5\n",
                encoding="utf-8",
            )
            try:
                os.environ.clear()
                os.environ.update(
                    {
                        "WCP_CONFIG_ENV_PATH": str(config_path),
                        "DDSS_API_KEY": "env-key",
                        "WCP_OPENAI_MODEL": "env-model",
                    }
                )

                provider = get_provider("auto")

                self.assertIsInstance(provider, OpenAICompatibleProvider)
                self.assertEqual(provider.api_key, "env-key")
                self.assertEqual(provider.model, "env-model")
            finally:
                os.environ.clear()
                os.environ.update(original_env)


if __name__ == "__main__":
    unittest.main()
