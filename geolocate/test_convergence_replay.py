"""Tests for the standalone Geolocate convergence replay."""

import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

from clemcore.backends import ModelSpec
from geolocate.scripts import run_convergence_replay as replay

from geolocate.scripts.run_convergence_replay import (
    build_closure_prompt,
    discover_timed_out_episodes,
    extract_visible_game_context,
    load_evidence,
    score_prediction,
)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_replay_prompt_excludes_internal_game_master_evaluation(tmp_path):
    interactions = {
        "turns": [[
            {
                "from": "GM",
                "to": "Player 1",
                "action": {"type": "send message", "content": "Locate these views."},
            },
            {
                "from": "Player 1",
                "to": "GM",
                "action": {"type": "get message", "content": "LATITUDE: 1"},
            },
            {
                "from": "GM",
                "to": "GM",
                "action": {
                    "type": "geolocation evaluation",
                    "content": {"target_latitude": 99.12345, "distance_km": 2.0},
                },
            },
        ], [
            {
                "from": "GM",
                "to": "Player 1",
                "action": {"type": "send message", "content": "DISTANCE_FEEDBACK: under 25 km"},
            },
            {
                "from": "Player 1",
                "to": "GM",
                "action": {"type": "get message", "content": "AGENT_EPISODE_TIMEOUT: stopped"},
            },
        ]],
    }

    initial, history = extract_visible_game_context(interactions)
    prompt = build_closure_prompt(initial, history, "Investigator saw a station sign.")

    assert "Locate these views." in prompt
    assert "LATITUDE: 1" in prompt
    assert "DISTANCE_FEEDBACK: under 25 km" in prompt
    assert "99.12345" not in prompt
    assert "distance_km" not in prompt
    assert "AGENT_EPISODE_TIMEOUT" not in prompt


def test_discovers_only_timed_out_geolocate_episodes(tmp_path):
    timed_out = tmp_path / "agent-a/geolocate/civic_public/instance_00002"
    completed = tmp_path / "agent-a/geolocate/civic_public/instance_00003"
    for directory, is_timeout, game_id in ((timed_out, True, 2), (completed, False, 3)):
        write_json(directory / "agent_trace_meta.json", {
            "agent": "agent-a",
            "experiment_name": "civic_public",
            "game_id": game_id,
            "episode_timed_out": is_timeout,
        })
        write_json(directory / "instance.json", {})
        write_json(directory / "interactions.json", {"Aborted": int(is_timeout)})

    episodes = discover_timed_out_episodes(tmp_path)

    assert [episode.game_id for episode in episodes] == [2]


def test_evidence_is_bounded_and_game_calls_are_not_repeated(tmp_path):
    write_json(tmp_path / "agent_loop.json", {
        "events": [
            {"type": "tool_call", "name": "game__start_game", "arguments": {}},
            {"type": "reasoning", "content": "Baltic hypothesis " * 100},
            {"type": "tool_call", "name": "web_search", "arguments": {"q": "station"}},
            {"type": "tool_result", "name": "web_search", "content": "useful result"},
            {"type": "tool_call", "name": "game__submit_response", "arguments": {}},
        ]
    })

    evidence, statistics = load_evidence(tmp_path, evidence_char_limit=500, event_char_limit=200)

    assert len(evidence) <= 500
    assert "Baltic hypothesis" in evidence
    assert "web_search" in evidence
    assert "start_game" not in evidence
    assert "submit_response" not in evidence
    assert statistics["auxiliary_tool_calls"] == 1
    assert statistics["game_action_calls"] == 2


def test_scores_prediction_with_geolocate_metrics():
    prediction = {"latitude": 52.52, "longitude": 13.405, "country": "Germany"}
    instance = {
        "target_latitude": 52.52,
        "target_longitude": 13.405,
        "target_country": "Germany",
        "target_country_code": "DE",
    }
    config = {
        "earth_radius_km": 6371.009,
        "inverse_distance_half_score_km": 100.0,
        "exponential_distance_half_score_km": 100.0,
        "close_range_half_score_km": 5.0,
    }

    score = score_prediction(prediction, instance, config)

    assert score["distance_km"] == 0
    assert score["inverse_distance_quality"] == 100
    assert score["country_correct"] is True


class TestConvergenceReplay(unittest.TestCase):
    """Offline fixtures only: never construct a provider client or start a harness."""

    def test_original_regressions(self):
        for test in (
            test_replay_prompt_excludes_internal_game_master_evaluation,
            test_discovers_only_timed_out_geolocate_episodes,
            test_evidence_is_bounded_and_game_calls_are_not_repeated,
        ):
            with self.subTest(test=test.__name__), tempfile.TemporaryDirectory() as directory:
                test(Path(directory))
        test_scores_prediction_with_geolocate_metrics()

    def test_per_event_clipping_is_reported_without_total_clipping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "agent_loop.json", {"events": [
                {"type": "reasoning", "content": "x" * 10000},
            ]})
            evidence, stats = load_evidence(root, 250000, 3000)
            self.assertTrue(stats["evidence_truncated"])
            self.assertFalse(stats["total_limit_truncated"])
            self.assertEqual(stats["event_items_truncated"], 1)
            self.assertGreater(stats["evidence_characters_before_event_limits"], len(evidence))

    def test_media_encodings_do_not_fill_evidence_budget(self):
        event = {"type": "tool_result", "name": "inspect_image", "content": [
            {"type": "image", "data": "FAKE_BASE64" * 1000, "mimeType": "image/jpeg"},
            {"type": "text", "text": "Visible station sign"},
        ]}
        evidence = replay.evidence_line(event, 1000)
        self.assertIn("Visible station sign", evidence)
        self.assertNotIn("FAKE_BASE64", evidence)

    def test_no_completed_game_or_fourth_guess_is_replayed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for game_id, aborted, guesses in [(1, False, 1), (2, True, 3), (3, True, 1)]:
                episode = root / f"agent/geolocate/test/instance_{game_id:05d}"
                write_json(episode / "agent_trace_meta.json", {
                    "game_id": game_id, "agent": "agent", "experiment_name": "test", "episode_timed_out": True,
                })
                write_json(episode / "instance.json", {})
                write_json(episode / "interactions.json", {
                    "Aborted": aborted, "episode_result": {"predictions": [{}] * guesses, "max_rounds": 3},
                })
            self.assertEqual([item.game_id for item in discover_timed_out_episodes(root)], [3])

    def test_source_wire_settings_are_not_substituted_with_closer_settings(self):
        events = [
            {"type": "model_request", "payload": {"model": "qwen", "messages": ["private content"],
                                                    "reasoning": {"enabled": True, "effort": "high"}}},
            {"type": "model_request", "payload": {"model": "metadata-only"}},
        ]
        settings = replay.observed_request_settings(events)
        self.assertEqual(settings, [{"model": "qwen", "reasoning": {"enabled": True, "effort": "high"}}])
        self.assertNotIn("private content", json.dumps(settings))

    def test_native_fallback_is_delegated_to_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "agent_trace_meta.json", {"agent": "new-unseen-harness"})
            (root / "agent_trace.log").write_text("native format", encoding="utf-8")
            adapter = MagicMock()
            adapter.parse_agent_trace.return_value = {"events": [
                {"type": "reasoning", "content": "Evidence from the new harness"},
            ]}
            with patch("clemagents.adapters.harness_class_for_agent", return_value=adapter) as resolve:
                evidence, stats = load_evidence(root, 10000, 3000)
            self.assertIn("Evidence from the new harness", evidence)
            resolve.assert_called_once()
            adapter.parse_agent_trace.assert_called_once()
            self.assertIn("in memory", stats["evidence_source"])
            self.assertFalse((root / "agent_loop.json").exists())

    def test_counterfactual_uses_full_cohort_and_pending_is_not_zero(self):
        cohort = [
            {"agent": "a", "source_episode": "a/1", "source_quality": 60.0},
            {"agent": "a", "source_episode": "a/2", "source_quality": None},
            {"agent": "a", "source_episode": "a/3", "source_quality": None},
        ]
        selected = {"a/2"}
        pending = replay.aggregate_counterfactual(cohort, selected, [])['a']
        self.assertIsNone(pending["counterfactual"])
        self.assertEqual(pending["pending_replays"], 1)
        completed = replay.aggregate_counterfactual(cohort, selected, [{
            "source_episode": "a/2", "status": "completed", "valid_response": True,
            "score": {"inverse_distance_quality": 100.0},
        }])["a"]
        self.assertEqual(completed["baseline"]["clemscore"], 20.0)
        self.assertAlmostEqual(completed["counterfactual"]["clemscore"], 160 / 3)
        self.assertEqual(completed["counterfactual"]["quality"], 80.0)
        self.assertAlmostEqual(completed["counterfactual"]["percent_played"], 200 / 3)

    def make_fixture(self, root):
        config = {
            "earth_radius_km": 6371.009, "inverse_distance_half_score_km": 100.0,
            "exponential_distance_half_score_km": 100.0, "close_range_half_score_km": 5.0,
        }
        write_json(root / "geolocate/resources/experiment_config.json", config)
        registry = [{"agent_name": f"harness-{suffix}", "agent_config": {"clem_model": f"model-{suffix}"}}
                    for suffix in ("a", "b")]
        write_json(root / "agent_registry.json", registry)
        for suffix in ("a", "b"):
            for game_id in (1, 2):
                episode = root / f"source/harness-{suffix}/geolocate/civic_public/instance_{game_id:05d}"
                write_json(episode / "agent_trace_meta.json", {
                    "agent": f"harness-{suffix}", "experiment_name": "civic_public", "game_id": game_id,
                    "episode_timed_out": game_id == 1, "episode_elapsed_seconds": 1000.0,
                    "episode_terminal_reason": "episode_timeout" if game_id == 1 else "game_done",
                })
                write_json(episode / "interactions.json", {
                    "Aborted": int(game_id == 1),
                    "turns": [[{"from": "GM", "to": "Player 1", "action": {
                        "type": "send message", "content": "Locate the supplied views; you have three guesses.",
                    }}]],
                    "episode_result": {"max_rounds": 3, "target_latitude": 52.500321,
                                       "predictions": [] if game_id == 1 else [{"inverse_quality": 60.0}]},
                })
                write_json(episode / "instance.json", {
                    "image_paths": [f"images/{i}.jpg" for i in range(4)],
                    "target_latitude": 52.500321, "target_longitude": 13.4,
                    "target_country": "Germany", "target_country_code": "DE",
                })
                write_json(episode / "agent_loop.json", {"events": [
                    {"type": "reasoning", "content": "The images suggest a German station."},
                    {"type": "model_request", "payload": {"model": f"model-{suffix}", "messages": [],
                        "reasoning": {"enabled": True, "effort": "high"}}},
                ]})
        (root / "images").mkdir()
        for i in range(4):
            (root / f"images/{i}.jpg").write_bytes(f"offline image fixture {i}".encode())
        model_registry = MagicMock()
        model_registry.get_first_model_spec_that_unify_with.side_effect = lambda name: ModelSpec.from_dict({
            "model_name": name, "model_id": name, "backend": "offline-test",
            "context_size": 100000,
            "model_config": {"multimodality": {"multiple_images": True},
                             "extra_body": {"reasoning": {"enabled": True, "effort": "xhigh"}}},
        })
        return model_registry

    def run_fixture(self, root, registry, load, *extra):
        with patch.object(replay, "REPOSITORY_ROOT", root), patch.object(
            replay.ModelRegistry, "from_packaged_and_cwd_files", return_value=registry,
        ), patch.object(replay, "load_closer_model", side_effect=load), redirect_stdout(io.StringIO()):
            replay.main([
                "--results-dir", str(root / "source"), "--output-dir", str(root / "output"),
                "--run-name", "fixture", *extra,
            ])

    @staticmethod
    def fake_model():
        model = MagicMock()
        model.generate_response.return_value = (
            ["encoded fixture"], {"choices": [{"finish_reason": "stop"}], "usage": {"completion_tokens": 40}},
            "EXPLANATION: German station\nLATITUDE: 52.500321\nLONGITUDE: 13.4\nCOUNTRY: Germany",
        )
        return model

    def test_mixed_models_resume_and_originals_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = self.make_fixture(root)
            originals = {str(path): path.read_bytes() for path in (root / "source").rglob("*") if path.is_file()}
            models = {name: self.fake_model() for name in ("model-a", "model-b")}
            calls = []

            def load(spec, gen_args):
                calls.append(spec.model_name)
                self.assertEqual(gen_args, {"temperature": 1.0, "max_tokens": 2000})
                return models[spec.model_name]

            self.run_fixture(root, registry, load)
            self.assertEqual(calls, ["model-a", "model-b"])
            summary = replay.load_json(root / "output/fixture/summary.json")
            for cell in summary["by_agent"].values():
                self.assertEqual(cell["baseline"]["clemscore"], 30.0)
                self.assertEqual(cell["counterfactual"]["clemscore"], 80.0)
            for model in models.values():
                messages = model.generate_response.call_args.args[0]
                self.assertNotIn("52.500321", messages[0]["content"])
                self.assertEqual(len(messages[0]["image"]), 4)
            self.run_fixture(root, registry, lambda *args, **kwargs: self.fail("resume reloaded a completed model"), "--resume")
            self.assertEqual(originals, {str(path): path.read_bytes() for path in (root / "source").rglob("*") if path.is_file()})
            with self.assertRaises(FileExistsError):
                self.run_fixture(root, registry, load)
            with self.assertRaisesRegex(ValueError, "Cannot resume"):
                self.run_fixture(root, registry, load, "--resume", "--max-tokens", "10000")

    def test_dry_run_never_loads_model_or_writes_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = self.make_fixture(root)
            self.run_fixture(root, registry, lambda *args, **kwargs: self.fail("model loaded"), "--dry-run")
            self.assertFalse((root / "output").exists())

    def test_remaining_guesses_cli_preserves_cohort_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = self.make_fixture(root)
            config = replay.load_json(Path(__file__).parent / 'resources/experiment_config.json')
            write_json(root / 'geolocate/resources/experiment_config.json', config)
            template = root / 'geolocate/resources/feedback_prompts/en/geolocate.template'
            template.parent.mkdir(parents=True)
            template.write_text((Path(__file__).parent / 'resources/feedback_prompts/en/geolocate.template').read_text())
            originals = {str(p): p.read_bytes() for p in (root / 'source').rglob('*') if p.is_file()}
            self.run_fixture(root, registry, lambda *a, **kw: self.fail('model loaded'),
                             '--guess-mode', 'remaining-guesses', '--dry-run')
            self.assertFalse((root / 'output').exists())
            models = []
            def load(*args, **kwargs):
                model = self.fake_model()
                model.generate_response.return_value = ({}, {'choices': [{'finish_reason': 'stop'}]},
                    'EXPLANATION: Candidate\nLATITUDE: 51\nLONGITUDE: 13.4\nCOUNTRY: Germany')
                models.append(model)
                return model
            self.run_fixture(root, registry, load, '--guess-mode', 'remaining-guesses')
            self.assertEqual([m.generate_response.call_count for m in models], [3, 3])
            summary = replay.load_json(root / 'output/fixture/summary.json')
            self.assertEqual(summary['configuration']['guess_mode'], 'remaining-guesses')
            self.assertEqual(len(summary['episodes']), 2)
            for row in summary['episodes']:
                self.assertEqual(len(row['recovery_rounds']), 3)
                self.assertEqual(row['termination'], 'guesses_exhausted')
            for row in summary['by_agent'].values():
                self.assertEqual(row['counterfactual']['episodes'], 2)
            self.run_fixture(root, registry, lambda *a, **kw: self.fail('model reloaded'),
                             '--guess-mode', 'remaining-guesses', '--resume')
            with self.assertRaisesRegex(ValueError, 'Cannot resume'):
                self.run_fixture(root, registry, load, '--resume')
            self.assertEqual(originals, {str(p): p.read_bytes() for p in (root / 'source').rglob('*') if p.is_file()})

    def test_request_error_checkpoints_and_resume_retries_only_that_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = self.make_fixture(root)
            bad_model = self.fake_model()
            bad_model.generate_response.side_effect = RuntimeError("offline simulated HTTP 402")
            with self.assertRaises(SystemExit):
                self.run_fixture(root, registry, lambda spec, **kwargs: self.fake_model() if spec.model_name == "model-a" else bad_model)
            summary = replay.load_json(root / "output/fixture/summary.json")
            self.assertIsNone(summary["by_agent"]["harness-b"]["counterfactual"])
            self.assertEqual(summary["by_agent"]["harness-b"]["pending_replays"], 1)
            calls = []

            def retry(spec, **kwargs):
                calls.append(spec.model_name)
                return self.fake_model()

            self.run_fixture(root, registry, retry, "--resume")
            self.assertEqual(calls, ["model-b"])
            destination = root / "output/fixture/harness-b/civic_public/instance_00001"
            self.assertEqual(replay.load_json(destination / "attempt_001/result.json")["status"], "request_error")
            self.assertEqual(replay.load_json(destination / "attempt_002/result.json")["status"], "completed")

    def test_output_limit_is_distinct_from_request_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = self.make_fixture(root)
            model = self.fake_model()
            model.generate_response.return_value = ([], {"choices": [{"finish_reason": "length"}]}, "EXPLANATION: incomplete")
            self.run_fixture(root, registry, lambda *args, **kwargs: model)
            summary = replay.load_json(root / "output/fixture/summary.json")
            self.assertTrue(all(item["status"] == "output_limit" for item in summary["episodes"]))
            self.assertTrue(all(item["counterfactual_clemscore"] == 0 for item in summary["episodes"]))

    def test_resume_recovers_saved_exchange_without_another_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = self.make_fixture(root)
            self.run_fixture(root, registry, lambda *args, **kwargs: self.fake_model())
            destination = root / "output/fixture/harness-a/civic_public/instance_00001"
            (destination / "result.json").unlink()
            (destination / "attempt_001/result.json").unlink()
            self.run_fixture(root, registry, lambda *args, **kwargs: self.fail("already saved response regenerated"), "--resume")
            self.assertEqual(replay.load_json(destination / "result.json")["status"], "completed")
            self.assertFalse((destination / "attempt_002").exists())

    def test_changed_scoring_configuration_is_rejected(self):
        interactions = {"episode_result": {"predictions": [{"distance_km": 100.0, "inverse_quality": 50.0}]}}
        config = {"inverse_distance_half_score_km": 100.0}
        replay.validate_scoring_config(interactions, config)
        config["inverse_distance_half_score_km"] = 50.0
        with self.assertRaisesRegex(ValueError, "Scoring configuration differs"):
            replay.validate_scoring_config(interactions, config)

    def test_post_timeout_messages_are_not_evidence(self):
        interactions = {"turns": [[
            {"from": "GM", "to": "Player 1", "action": {"type": "send message", "content": "Locate these images"}},
            {"from": "Player 1", "to": "GM", "action": {"type": "get message", "content": "AGENT_EPISODE_TIMEOUT: stopped"}},
            {"from": "GM", "to": "Player 1", "action": {"type": "send message", "content": "HIDDEN ANSWER REVEALED AFTER END"}},
        ]]}
        initial, history = extract_visible_game_context(interactions)
        self.assertEqual(initial, "Locate these images")
        self.assertEqual(history, [])

    def test_game_tool_match_requires_a_name_boundary(self):
        for name in ("start_game", "mcp__game__start_game", "game.submit_response", "game__submit_response"):
            self.assertTrue(replay.is_game_action(name))
        self.assertFalse(replay.is_game_action("restart_game"))

    def test_exact_closer_settings_reach_serialized_http_request(self):
        # The actual SDK/backends are exercised, but transport is entirely fake.
        import httpx
        import openai
        from clemcore.backends.openai_api import OpenAIModel
        from clemcore.backends.openrouter_api import OpenRouterModel

        cases = [
            ("gemma4-26b-a4b", OpenAIModel, {"reasoning_effort": "none"}),
            ("qwen3.8-27b-reasoning", OpenRouterModel, {"reasoning": {"enabled": False}}),
            ("glm-5.3-flash", OpenRouterModel, {"reasoning": {"enabled": True, "effort": "low"}}),
        ]
        for name, model_class, extra_body in cases:
            with self.subTest(model=name):
                captured = []

                def respond(request):
                    captured.append(json.loads(request.content))
                    return httpx.Response(200, json={
                        "id": "offline", "object": "chat.completion", "created": 0, "model": name,
                        "choices": [{"index": 0, "finish_reason": "stop", "message": {
                            "role": "assistant", "content": "EXPLANATION: test\nLATITUDE: 1\nLONGITUDE: 2\nCOUNTRY: test",
                        }}],
                    })

                with openai.OpenAI(api_key="offline-test", base_url="https://offline.invalid/v1",
                                   http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
                    backend = MagicMock()
                    backend.get_model_for.side_effect = lambda spec: model_class(client, spec)
                    backend_registry = MagicMock()
                    backend_registry.get_backend_for.return_value = backend
                    spec = ModelSpec.from_dict({
                        "model_name": name, "model_id": name, "backend": "offline",
                        "model_config": {"extra_body": extra_body},
                    })
                    original = json.dumps(spec.to_dict(), sort_keys=True)
                    with patch.object(replay.BackendRegistry, "from_packaged_and_cwd_files", return_value=backend_registry), patch.object(
                        replay.ModelRegistry, "from_packaged_and_cwd_files",
                        side_effect=AssertionError("must not re-unify the configured spec"),
                    ):
                        model = replay.load_closer_model(spec, {"temperature": 1.0, "max_tokens": 50000})
                        model.generate_response([{"role": "user", "content": "offline test"}])
                    self.assertEqual(len(captured), 1)
                    wire = captured[0]
                    for key, value in extra_body.items():
                        self.assertEqual(wire[key], value)
                    if name.startswith("gemma"):
                        self.assertNotIn("reasoning", wire)
                    if name.startswith("qwen"):
                        self.assertNotIn("effort", wire["reasoning"])
                    self.assertEqual(wire["max_tokens"], 50000)
                    self.assertEqual(json.dumps(spec.to_dict(), sort_keys=True), original)

    def test_cli_overrides_are_saved_and_passed_to_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = self.make_fixture(root)
            settings = {"model-a": {"reasoning_effort": "none"}, "model-b": {"reasoning": {"enabled": False}}}

            def load(spec, **kwargs):
                self.assertEqual(spec.model_config["extra_body"], settings[spec.model_name])
                return self.fake_model()

            self.run_fixture(root, registry, load,
                             "--model-extra-body", 'model-a={"reasoning_effort":"none"}',
                             "--model-extra-body", 'model-b={"reasoning":{"enabled":false}}')
            for request_path in (root / "output").glob("**/attempt_001/request.json"):
                request = replay.load_json(request_path)
                self.assertEqual(request["closer_model_spec"]["model_config"]["extra_body"], settings[request["closer_model"]])
                self.assertEqual(request["model_loading"], "exact_spec_v1")


if __name__ == "__main__":
    unittest.main()
