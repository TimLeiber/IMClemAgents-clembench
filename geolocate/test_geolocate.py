"""Unit tests for the Geolocate game."""

import json
import math
import unittest
from pathlib import Path

from clemcore.backends import CustomResponseModel
from clemcore.clemgame import GameSpec
from clemcore.clemgame.master import Outcome
from clemcore.clemgame.metrics import BENCH_SCORE

from instancegenerator import GeolocateGameInstanceGenerator
from master import (
    GeolocateGameBenchmark,
    GeolocateGameMaster,
    GeolocateGameScorer,
    GeolocateGameState,
)
from utils import (
    close_range_quality,
    country_is_correct,
    distance_feedback_band,
    exponential_distance_quality,
    haversine_distance_km,
    inverse_distance_quality,
    parse_geolocation_response,
)


GAME_PATH = Path(__file__).resolve().parent
LABELS = {
    "explanation_label": "EXPLANATION",
    "latitude_label": "LATITUDE",
    "longitude_label": "LONGITUDE",
    "country_label": "COUNTRY",
}


def create_master() -> GeolocateGameMaster:
    """Create a one-instance master backed by the programmatic player."""
    game_spec = GameSpec.from_dict({
        "game_name": "geolocate",
        "description": "Single-player multimodal geolocation game",
        "main_game": "geolocate",
        "players": 1,
        "image": "multi",
        "languages": ["en"],
        "benchmark": ["3.0"],
        "game_path": str(GAME_PATH),
    })
    master = GeolocateGameMaster(
        game_spec,
        {"name": "test", "lang": "en"},
        [CustomResponseModel()],
    )
    master.setup(
        game_id=1,
        image_paths=[
            f"geolocate/resources/images/location_002/view_{index:02d}.jpg"
            for index in range(4)
        ],
        target_latitude=-6.193977,
        target_longitude=106.849348,
        target_country="Indonesia",
        target_country_code="ID",
    )
    master.before_game()
    return master


def prediction(latitude: float, longitude: float, country: str) -> str:
    """Build one valid test prediction."""
    return (
        "EXPLANATION: Test prediction based on the available views.\n"
        f"LATITUDE: {latitude}\n"
        f"LONGITUDE: {longitude}\n"
        f"COUNTRY: {country}"
    )


class GeolocateTestCase(unittest.TestCase):
    def test_required_classes_are_defined(self):
        self.assertIsNotNone(GeolocateGameBenchmark)
        self.assertIsNotNone(GeolocateGameInstanceGenerator)
        self.assertIsNotNone(GeolocateGameMaster)
        self.assertIsNotNone(GeolocateGameScorer)
        self.assertIsNotNone(GeolocateGameState)

    def test_game_metadata_declares_multiple_images(self):
        metadata = json.loads((GAME_PATH / "clemgame.json").read_text())
        self.assertEqual(metadata["game_name"], "geolocate")
        self.assertEqual(metadata["players"], 1)
        self.assertEqual(metadata["image"], "multi")

    def test_parse_valid_response(self):
        parsed = parse_geolocation_response(
            "EXPLANATION: Urban form and German road signs.\n"
            "LATITUDE: 52.52\n"
            "LONGITUDE: 13.405\n"
            "COUNTRY: Germany",
            LABELS,
        )
        self.assertEqual(parsed["latitude"], 52.52)
        self.assertEqual(parsed["longitude"], 13.405)
        self.assertEqual(parsed["country"], "Germany")

    def test_parse_rejects_wrapped_or_extra_output(self):
        with self.assertRaisesRegex(ValueError, "exactly four"):
            parse_geolocation_response(
                "EXPLANATION: first line\nsecond line\n"
                "LATITUDE: 1\nLONGITUDE: 2\nCOUNTRY: X",
                LABELS,
            )

    def test_parse_rejects_out_of_range_coordinates(self):
        with self.assertRaisesRegex(ValueError, "Latitude"):
            parse_geolocation_response(
                "EXPLANATION: clue\nLATITUDE: 91\nLONGITUDE: 2\nCOUNTRY: X",
                LABELS,
            )

    def test_haversine_distance(self):
        berlin_to_paris = haversine_distance_km(
            52.5200, 13.4050, 48.8566, 2.3522
        )
        self.assertAlmostEqual(berlin_to_paris, 877, delta=2)
        self.assertEqual(haversine_distance_km(1, 2, 1, 2), 0)

    def test_quality_scores_share_configured_half_point(self):
        self.assertEqual(inverse_distance_quality(0, 100), 100)
        self.assertEqual(inverse_distance_quality(100, 100), 50)
        self.assertEqual(exponential_distance_quality(0, 100), 100)
        self.assertTrue(math.isclose(exponential_distance_quality(100, 100), 50))

    def test_close_range_quality_separates_short_distances(self):
        self.assertEqual(close_range_quality(0, 5), 100)
        self.assertTrue(math.isclose(close_range_quality(5, 5), 50))
        self.assertTrue(math.isclose(close_range_quality(10, 5), 25))
        # Sub-kilometre predictions stay well separated at this half point.
        self.assertGreater(close_range_quality(1, 5), 85)
        self.assertLess(close_range_quality(15, 5), 13)

    def test_distance_feedback_bands_use_exclusive_upper_bounds(self):
        bands = [
            {"upper_bound_km": 25, "label": "under 25 km"},
            {"upper_bound_km": 100, "label": "25-100 km"},
            {"upper_bound_km": None, "label": "over 100 km"},
        ]
        self.assertEqual(distance_feedback_band(24.9, bands), (0, "under 25 km"))
        self.assertEqual(distance_feedback_band(25, bands), (1, "25-100 km"))
        self.assertEqual(distance_feedback_band(100, bands), (2, "over 100 km"))

    def test_country_name_or_code_is_accepted(self):
        self.assertTrue(country_is_correct("Germany", "Germany", "DE"))
        self.assertTrue(country_is_correct("de", "Germany", "DE"))
        self.assertTrue(country_is_correct("USA", "United States", "US"))
        self.assertFalse(country_is_correct("Austria", "Germany", "DE"))

    def test_programmatic_episode_receives_four_images_and_scores_exactly(self):
        master = create_master()
        context = master.get_context_for(master.locator)
        self.assertEqual(len(context["image"]), 4)
        self.assertIn("LATITUDE:", context["content"])
        self.assertIn("photo sequence", context["content"])
        self.assertNotIn("capabilities available to you", context["content"])
        self.assertNotIn("$", context["content"])

        response = master.locator(context)
        done, _ = master.step(response)

        self.assertTrue(done)
        self.assertEqual(master.state.outcome, Outcome.SUCCESS)
        self.assertEqual(master.state.distance_km, 0)
        self.assertEqual(master.state.inverse_quality, 100)
        self.assertTrue(master.state.country_correct)
        self.assertEqual(len(master.state.predictions), 1)

    def test_three_turn_episode_returns_banded_programmatic_feedback(self):
        master = create_master()

        done, _ = master.step(prediction(0, 0, "Unknown"))
        self.assertFalse(done)
        feedback = master.get_context_for(master.locator)["content"]
        self.assertIn("DISTANCE_FEEDBACK: over 2,000 km", feedback)
        self.assertIn("COUNTRY_FEEDBACK: incorrect", feedback)
        self.assertIn("PROGRESS_FEEDBACK: first prediction", feedback)
        self.assertIn("PREDICTIONS_REMAINING: 2", feedback)
        self.assertNotIn(str(master.state.distance_km), feedback)

        done, _ = master.step(prediction(-6, 100, "Indonesia"))
        self.assertFalse(done)
        feedback = master.get_context_for(master.locator)["content"]
        self.assertIn("DISTANCE_FEEDBACK: 500-2,000 km", feedback)
        self.assertIn("COUNTRY_FEEDBACK: correct", feedback)
        self.assertIn("closer than the previous prediction", feedback)
        self.assertIn("PREDICTIONS_REMAINING: 1", feedback)

        done, _ = master.step(prediction(-6, 106, "Indonesia"))
        self.assertTrue(done)
        self.assertEqual(master.state.outcome, Outcome.SUCCESS)
        self.assertEqual(len(master.state.predictions), 3)
        self.assertEqual(len(master.state.feedbacks), 2)

    def test_scorer_uses_inverse_quality_as_main_score(self):
        scorer = GeolocateGameScorer("geolocate", {}, {})
        scorer.compute_episode_scores({
            "Aborted": 0,
            "episode_result": {
                "distance_km": 100,
                "inverse_quality": 50,
                "exponential_quality": 50,
                "close_range_quality": 0.0,
                "country_correct": True,
                "predictions": [
                    {
                        "distance_km": 300,
                        "inverse_quality": 25,
                        "exponential_quality": 12.5,
                        "close_range_quality": 0.0,
                        "country_correct": False,
                    },
                    {
                        "distance_km": 25,
                        "inverse_quality": 80,
                        "exponential_quality": 84,
                        "close_range_quality": 3.125,
                        "country_correct": True,
                    },
                    {
                        "distance_km": 100,
                        "inverse_quality": 50,
                        "exponential_quality": 50,
                        "close_range_quality": 0.0,
                        "country_correct": True,
                    },
                ],
            },
        })
        episode_scores = scorer.scores["episode scores"]
        self.assertEqual(episode_scores[BENCH_SCORE], 50)
        self.assertEqual(episode_scores["Distance km"], 100)
        self.assertEqual(episode_scores["Close Range Quality"], 0.0)
        self.assertEqual(episode_scores["Country Accuracy"], 1)
        self.assertEqual(episode_scores["Initial Inverse Distance Quality"], 25)
        self.assertEqual(episode_scores["Best Inverse Distance Quality"], 80)
        self.assertEqual(episode_scores["Final Improvement"], 25)
        self.assertEqual(episode_scores["Predictions Used"], 3)

    def test_scorer_records_exact_round_diagnostics(self):
        scorer = GeolocateGameScorer("geolocate", {}, {})
        scorer.compute_round_score(0, [{
            "action": {
                "type": "geolocation evaluation",
                "content": {
                    "distance_km": 742.5,
                    "inverse_quality": 11.87,
                    "exponential_quality": 0.58,
                    "close_range_quality": 0.0,
                    "country_correct": False,
                    "distance_band_index": 3,
                },
            },
        }])
        round_scores = scorer.scores["round scores"][0]
        self.assertEqual(round_scores["Distance km"], 742.5)
        self.assertEqual(round_scores["Inverse Distance Quality"], 11.87)
        self.assertEqual(round_scores["Close Range Quality"], 0.0)
        self.assertEqual(round_scores["Country Accuracy"], 0)
        self.assertEqual(round_scores["Distance Band Index"], 3)


if __name__ == "__main__":
    unittest.main()
