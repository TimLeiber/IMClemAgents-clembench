"""Single-player multimodal Geolocate game."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from clemcore.backends import Model
from clemcore.clemgame import GameBenchmark, GameScorer, GameSpec, ParseError, Player
from clemcore.clemgame.master import DialogueGameMaster, GameState, Outcome
from clemcore.clemgame.metrics import BENCH_SCORE, METRIC_ABORTED, METRIC_LOSE, METRIC_SUCCESS

from utils import (
    close_range_quality,
    country_is_correct,
    distance_feedback_band,
    exponential_distance_quality,
    haversine_distance_km,
    inverse_distance_quality,
    parse_geolocation_response,
)


GAME_NAME = "geolocate"


@dataclass
class GeolocateGameState(GameState):
    """Store the reference location and iterative predictions."""

    image_paths: Optional[List[str]] = None
    target_latitude: Optional[float] = None
    target_longitude: Optional[float] = None
    target_country: Optional[str] = None
    target_country_code: Optional[str] = None
    predicted_latitude: Optional[float] = None
    predicted_longitude: Optional[float] = None
    predicted_country: Optional[str] = None
    explanation: Optional[str] = None
    distance_km: Optional[float] = None
    inverse_quality: Optional[float] = None
    exponential_quality: Optional[float] = None
    close_range_quality: Optional[float] = None
    country_correct: Optional[bool] = None
    max_rounds: int = 3
    predictions: List[Dict] = field(default_factory=list)
    feedbacks: List[Dict] = field(default_factory=list)
    invalid_response: bool = False

    def __post_init__(self):
        super().__init__()


class GeolocatePlayer(Player):
    """Single player responsible for locating the supplied views."""

    def __init__(self,
                 model: Model,
                 target_latitude: float,
                 target_longitude: float,
                 target_country: str,
                 response_format: Dict[str, str]):
        super().__init__(model, game_role="Locator")
        self.target_latitude = target_latitude
        self.target_longitude = target_longitude
        self.target_country = target_country
        self.response_format = response_format

    def _custom_response(self, context: Dict) -> str:
        """Return a valid deterministic response for programmatic test models."""
        return (
            f"{self.response_format['explanation_label']}: Programmatic reference prediction.\n"
            f"{self.response_format['latitude_label']}: {self.target_latitude}\n"
            f"{self.response_format['longitude_label']}: {self.target_longitude}\n"
            f"{self.response_format['country_label']}: {self.target_country}"
        )


class GeolocateGameMaster(DialogueGameMaster):
    """Run an iterative four-view geolocation episode."""

    def __init__(self,
                 game_spec: GameSpec,
                 experiment: Dict,
                 player_models: List[Model]):
        super().__init__(game_spec, experiment, player_models)

    def _on_setup(self, **game_instance):
        config = self.load_json("resources/experiment_config")
        lang_config = self.load_json("resources/langconfig")[self.experiment["lang"]]
        response_format = lang_config["response_format"]
        prompt = self.load_template(
            config["initial_prompt"].format(lang=self.experiment["lang"])
        )
        feedback_template = self.load_template(
            config["feedback_prompt"].format(lang=self.experiment["lang"])
        )

        for key, label in response_format.items():
            prompt = prompt.replace(f"${key.upper()}$", label)
            feedback_template = feedback_template.replace(f"${key.upper()}$", label)

        prompt = prompt.replace("$MAX_ROUNDS$", str(config["max_rounds"]))

        image_paths = list(game_instance["image_paths"])

        if len(image_paths) != 4:
            raise ValueError("Each Geolocate instance must contain exactly four views.")

        self.game_instance = game_instance
        self.response_format = response_format
        self.feedback_template = feedback_template
        self.config = config
        self.state = GeolocateGameState(
            image_paths=image_paths,
            target_latitude=float(game_instance["target_latitude"]),
            target_longitude=float(game_instance["target_longitude"]),
            target_country=game_instance["target_country"],
            target_country_code=game_instance.get("target_country_code"),
            max_rounds=int(config["max_rounds"]),
        )
        self.locator = GeolocatePlayer(
            self.player_models[0],
            self.state.target_latitude,
            self.state.target_longitude,
            self.state.target_country,
            response_format,
        )
        self.add_player(
            self.locator,
            initial_context={
                "role": "user",
                "content": prompt,
                "image": image_paths,
            },
        )

    def _parse_response(self, player: Player, response: str) -> dict:
        try:
            prediction = parse_geolocation_response(response, self.response_format)
        except ValueError as error:
            raise ParseError(
                str(error),
                response=response,
                key="invalid_response",
            ) from error

        self.log_to_self("submitted geolocation", {
            "response": response,
            **prediction,
        })
        return prediction

    def _advance_game(self, player: Player, parsed_response: dict):
        self.state.predicted_latitude = parsed_response["latitude"]
        self.state.predicted_longitude = parsed_response["longitude"]
        self.state.predicted_country = parsed_response["country"]
        self.state.explanation = parsed_response["explanation"]
        self.state.distance_km = haversine_distance_km(
            self.state.target_latitude,
            self.state.target_longitude,
            self.state.predicted_latitude,
            self.state.predicted_longitude,
            self.config["earth_radius_km"],
        )
        self.state.inverse_quality = inverse_distance_quality(
            self.state.distance_km,
            self.config["inverse_distance_half_score_km"],
        )
        self.state.exponential_quality = exponential_distance_quality(
            self.state.distance_km,
            self.config["exponential_distance_half_score_km"],
        )
        self.state.close_range_quality = close_range_quality(
            self.state.distance_km,
            self.config["close_range_half_score_km"],
        )
        self.state.country_correct = country_is_correct(
            self.state.predicted_country,
            self.state.target_country,
            self.state.target_country_code,
        )
        band_index, band_label = distance_feedback_band(
            self.state.distance_km,
            self.config["distance_feedback_bands"],
        )
        previous_distance = (
            self.state.predictions[-1]["distance_km"]
            if self.state.predictions
            else None
        )

        if previous_distance is None:
            progress_feedback = "first prediction"
        elif self.state.distance_km < previous_distance:
            progress_feedback = "closer than the previous prediction"
        elif self.state.distance_km > previous_distance:
            progress_feedback = "farther from the target than the previous prediction"
        else:
            progress_feedback = "the same distance as the previous prediction"

        evaluation = {
            "round": self.current_round + 1,
            "latitude": self.state.predicted_latitude,
            "longitude": self.state.predicted_longitude,
            "country": self.state.predicted_country,
            "explanation": self.state.explanation,
            "distance_km": self.state.distance_km,
            "inverse_quality": self.state.inverse_quality,
            "exponential_quality": self.state.exponential_quality,
            "close_range_quality": self.state.close_range_quality,
            "country_correct": self.state.country_correct,
            "distance_band_index": band_index,
            "distance_band": band_label,
            "progress_feedback": progress_feedback,
        }
        self.state.predictions.append(evaluation)
        self.log_to_self("geolocation evaluation", evaluation)

        reached_target = (
            self.state.distance_km <= float(self.config["early_stop_distance_km"])
            and self.state.country_correct
        )
        exhausted_rounds = self.current_round + 1 >= self.state.max_rounds

        if reached_target or exhausted_rounds:
            self.state.succeed()
            return

        feedback = {
            "distance_band": band_label,
            "country_feedback": "correct" if self.state.country_correct else "incorrect",
            "progress_feedback": progress_feedback,
            "predictions_remaining": self.state.max_rounds - len(self.state.predictions),
        }
        self.state.feedbacks.append(feedback)
        content = (
            self.feedback_template
            .replace("$DISTANCE_FEEDBACK$", feedback["distance_band"])
            .replace("$COUNTRY_FEEDBACK$", feedback["country_feedback"])
            .replace("$PROGRESS_FEEDBACK$", feedback["progress_feedback"])
            .replace("$PREDICTIONS_REMAINING$", str(feedback["predictions_remaining"]))
        )
        self.set_context_for(self.locator, content)

    def _on_parse_error(self, error: ParseError):
        self.state.invalid_response = True
        self.log_to_self("invalid response", {
            "response": error.response,
            "error": error.reason,
        })
        self.state.abort()

    def _does_game_proceed(self) -> bool:
        return super()._does_game_proceed()

    def _on_after_game(self):
        self.log_key(METRIC_ABORTED, int(self.state.outcome == Outcome.ABORTED))
        self.log_key(METRIC_LOSE, int(self.state.outcome == Outcome.FAILURE))
        self.log_key(METRIC_SUCCESS, int(self.state.outcome == Outcome.SUCCESS))
        self.log_key("episode_result", {
            "outcome": self.state.outcome.value,
            "target_latitude": self.state.target_latitude,
            "target_longitude": self.state.target_longitude,
            "target_country": self.state.target_country,
            "target_country_code": self.state.target_country_code,
            "predicted_latitude": self.state.predicted_latitude,
            "predicted_longitude": self.state.predicted_longitude,
            "predicted_country": self.state.predicted_country,
            "explanation": self.state.explanation,
            "distance_km": self.state.distance_km,
            "inverse_quality": self.state.inverse_quality,
            "exponential_quality": self.state.exponential_quality,
            "close_range_quality": self.state.close_range_quality,
            "country_correct": self.state.country_correct,
            "max_rounds": self.state.max_rounds,
            "predictions": self.state.predictions,
            "feedbacks": self.state.feedbacks,
            "invalid_response": self.state.invalid_response,
        })


class GeolocateGameScorer(GameScorer):
    """Score coordinate distance, country accuracy, and format adherence."""

    def compute_round_score(self,
                            round_idx: int,
                            round_events: List[Dict]) -> None:
        evaluation = None

        for event in round_events:
            if event["action"]["type"] == "geolocation evaluation":
                evaluation = event["action"]["content"]

        if evaluation is None:
            return

        self.log_round_score(round_idx, "Distance km", evaluation["distance_km"])
        self.log_round_score(
            round_idx,
            "Inverse Distance Quality",
            evaluation["inverse_quality"],
        )
        self.log_round_score(
            round_idx,
            "Exponential Distance Quality",
            evaluation["exponential_quality"],
        )
        self.log_round_score(
            round_idx,
            "Close Range Quality",
            evaluation["close_range_quality"],
        )
        self.log_round_score(
            round_idx,
            "Country Accuracy",
            int(evaluation["country_correct"]),
        )
        self.log_round_score(
            round_idx,
            "Distance Band Index",
            evaluation["distance_band_index"],
        )

    def compute_episode_scores(self, interactions: Dict) -> None:
        episode = interactions["episode_result"]
        predictions = episode["predictions"]

        if interactions[METRIC_ABORTED] or not predictions:
            main_score = np.nan
            distance_km = np.nan
            inverse_quality = np.nan
            exponential_quality = np.nan
            close_range = np.nan
            country_correct = np.nan
            initial_quality = np.nan
            best_quality = np.nan
            improvement = np.nan
        else:
            final_prediction = predictions[-1]
            main_score = final_prediction["inverse_quality"]
            distance_km = final_prediction["distance_km"]
            inverse_quality = final_prediction["inverse_quality"]
            exponential_quality = final_prediction["exponential_quality"]
            close_range = final_prediction["close_range_quality"]
            country_correct = int(final_prediction["country_correct"])
            initial_quality = predictions[0]["inverse_quality"]
            best_quality = max(item["inverse_quality"] for item in predictions)
            improvement = inverse_quality - initial_quality

        self.log_episode_score(BENCH_SCORE, main_score)
        self.log_episode_score("Distance km", distance_km)
        self.log_episode_score("Inverse Distance Quality", inverse_quality)
        self.log_episode_score("Exponential Distance Quality", exponential_quality)
        self.log_episode_score("Close Range Quality", close_range)
        self.log_episode_score("Country Accuracy", country_correct)
        self.log_episode_score("Initial Inverse Distance Quality", initial_quality)
        self.log_episode_score("Best Inverse Distance Quality", best_quality)
        self.log_episode_score("Final Improvement", improvement)
        self.log_episode_score("Predictions Used", len(predictions))


class GeolocateGameBenchmark(GameBenchmark):
    """Register Geolocate with the clembench runner."""

    def create_game_master(self,
                           experiment: Dict,
                           player_models: List[Model]) -> DialogueGameMaster:
        return GeolocateGameMaster(self.game_spec, experiment, player_models)

    def create_game_scorer(self,
                           experiment: Dict,
                           game_instance: Dict) -> GameScorer:
        return GeolocateGameScorer(GAME_NAME, experiment, game_instance)
