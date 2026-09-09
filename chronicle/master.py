"""
Chronicle game master.

Two-player historical deduction game: the Narrator describes a famous event
across up to 5 rounds using circumlocution (no forbidden proper nouns), while
the Detective attempts to identify it.  All scoring is fully programmatic.
"""

import re
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from clemcore.clemgame import GameBenchmark, Player, GameSpec
from clemcore.clemgame.legacy.master import DialogueGameMaster
from clemcore.clemgame.legacy.scorer import GameScorer
from clemcore.clemgame.master import GameState, Outcome
from clemcore.clemgame.metrics import METRIC_ABORTED, METRIC_SUCCESS, METRIC_LOSE, BENCH_SCORE

GAME_NAME = "chronicle"
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------

@dataclass
class ChronicleState(GameState):
    target_event: str = None
    event_category: str = None
    difficulty: str = None
    event_summary: str = None
    forbidden_keywords: list = field(default_factory=list)
    acceptable_answers: list = field(default_factory=list)
    max_detective_words: int = 100
    max_turns: int = 5

    # Per-turn tracking
    paragraphs_accumulated: list = field(default_factory=list)
    violations_per_turn: list = field(default_factory=list)   # 0 or 1 per narrator turn
    guesses: list = field(default_factory=list)
    success_turn: Optional[int] = None

    # Transient: set in _validate, consumed in _on_valid
    _current_paragraph: str = None
    _current_guess: str = None

    def __post_init__(self):
        super().__init__()


# ---------------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------------

class ChronicleNarrator(Player):
    def __init__(self, model):
        super().__init__(model)
        self._custom_responses = [
            "PARAGRAPH: A significant conflict unfolded between two opposing sides at a crucial moment in history.\nAPPROACH: Starting broad to avoid revealing too much too soon.",
            "PARAGRAPH: The scale of the engagement was immense, involving vast numbers of participants from different regions.\nAPPROACH: Adding geographic and numerical scope without names.",
            "PARAGRAPH: A decisive turning point came when one side gained a critical advantage, forever altering the balance of power.\nAPPROACH: Hinting at the pivotal outcome.",
            "PARAGRAPH: The aftermath reshaped entire societies and redrew boundaries that would persist for generations.\nAPPROACH: Focusing on long-term consequences.",
            "PARAGRAPH: This episode is considered one of the most significant of its era and remains widely studied to this day.\nAPPROACH: Final broad characterisation.",
        ]

    def _custom_response(self, messages):
        if self._custom_responses:
            return self._custom_responses.pop(0)
        return "PARAGRAPH: Further context follows.\nAPPROACH: Elaborating on remaining details."


class ChronicleDetective(Player):
    def __init__(self, model):
        super().__init__(model)
        self._custom_responses = [
            "ANALYSIS: The description is very vague. Could be many historical conflicts.\nGUESS: The Battle of Waterloo",
            "ANALYSIS: Large-scale, multi-regional — possibly a famous European battle.\nGUESS: The Battle of Waterloo",
            "ANALYSIS: A decisive turning point with lasting political impact.\nGUESS: The Battle of Waterloo",
            "ANALYSIS: Reshaped boundaries for generations — this sounds like a landmark event.\nGUESS: The Battle of Waterloo",
            "ANALYSIS: Widely studied, considered a defining moment of its era.\nGUESS: The Battle of Waterloo",
        ]

    def _custom_response(self, messages):
        if self._custom_responses:
            return self._custom_responses.pop(0)
        return "ANALYSIS: Still deducing from the clues.\nGUESS: Unknown historical event"


# ---------------------------------------------------------------------------
# Game master
# ---------------------------------------------------------------------------

class ChronicleGameMaster(DialogueGameMaster):
    """Chronicle game master: orchestrates the Narrator–Detective exchange."""

    def __init__(self, game_spec: GameSpec, experiment: Dict, player_models: List):
        super().__init__(game_spec, experiment, player_models)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _on_setup(self, **game_instance):
        self.state = ChronicleState(
            target_event=game_instance["target_event"],
            event_category=game_instance.get("event_category", ""),
            difficulty=game_instance.get("difficulty", ""),
            event_summary=game_instance.get("event_summary", ""),
            forbidden_keywords=game_instance.get("forbidden_keywords", []),
            acceptable_answers=game_instance.get("acceptable_answers", []),
            max_detective_words=self.experiment.get("max_detective_words", 100),
            max_turns=self.experiment.get("max_turns", 5),
        )

        forbidden_str = (
            ", ".join(self.state.forbidden_keywords)
            if self.state.forbidden_keywords else "none"
        )
        narrator_prompt = (
            self.experiment["narrator_initial_prompt"]
            .replace("$FORBIDDEN_KEYWORDS$", forbidden_str)
            .replace("$TARGET_EVENT$", self.state.target_event)
        )
        detective_prompt = (
            self.experiment["detective_initial_prompt"]
            .replace("$MAX_DETECTIVE_WORDS$", str(self.state.max_detective_words))
        )

        self.narrator = ChronicleNarrator(self.player_models[0])
        self.detective = ChronicleDetective(self.player_models[1])
        self.add_player(self.narrator, initial_context=narrator_prompt)
        self.add_player(self.detective, initial_prompt=detective_prompt)

    # ------------------------------------------------------------------
    # Flow control
    # ------------------------------------------------------------------

    def _does_game_proceed(self) -> bool:
        return self.state.outcome == Outcome.RUNNING

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_player_response(self, player: Player, utterance: str) -> bool:
        if player == self.narrator:
            return self._validate_narrator(utterance)
        elif player == self.detective:
            return self._validate_detective(utterance)
        return True

    def _validate_narrator(self, utterance: str) -> bool:
        para_m = re.search(r'PARAGRAPH:\s*', utterance, re.IGNORECASE)
        if not para_m:
            self.log_to_self("invalid format", "narrator: missing PARAGRAPH section")
            self.state.abort()
            return False

        paragraph = utterance[para_m.end():].strip()

        if not paragraph:
            self.log_to_self("invalid format", "narrator: PARAGRAPH section is empty")
            self.state.abort()
            return False

        self.state._current_paragraph = paragraph

        # Check forbidden keywords — recorded but does NOT abort
        violated = self._check_forbidden(paragraph)
        self.state.violations_per_turn.append(1 if violated else 0)
        if violated:
            self.log_to_self("forbidden keyword violation", violated)

        return True

    def _validate_detective(self, utterance: str) -> bool:
        m = re.search(r'GUESS:\s*(.+)', utterance, re.IGNORECASE)
        if not m or not m.group(1).strip():
            self.log_to_self("invalid format", "detective: missing GUESS section")
            self.state.abort()
            return False

        self.state._current_guess = m.group(1).strip()
        return True

    # ------------------------------------------------------------------
    # Response handlers
    # ------------------------------------------------------------------

    def _on_valid_player_response(self, player: Player, parsed_response: str):
        if player == self.narrator:
            self.state.paragraphs_accumulated.append(self.state._current_paragraph)
            paragraph_num = len(self.state.paragraphs_accumulated)
            self.set_context_for(self.detective, f"[Sentence {paragraph_num}]\n{self.state._current_paragraph}")

        elif player == self.detective:
            guess = self.state._current_guess
            self.state.guesses.append(guess)

            if self._matches_answer(guess):
                self.state.success_turn = self.current_round + 1
                self.log_to_self("correct guess", f"round {self.state.success_turn}")
                self.state.succeed()
            else:
                rnd = self.current_round + 1
                self.set_context_for(
                    self.narrator,
                    f"Round {rnd} complete. The Detective's current guess: \"{guess}\". "
                    f"Please write a new paragraph with fresh information."
                )

    def _on_after_round(self):
        if (self.current_round + 1 >= self.state.max_turns
                and self.state.outcome == Outcome.RUNNING):
            self.log_to_self("max turns reached", str(self.state.max_turns))
            self.state.failed()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_forbidden(self, text: str) -> Optional[str]:
        tl = text.lower()
        for kw in self.state.forbidden_keywords:
            if re.search(r'\b' + re.escape(kw.lower()) + r'\b', tl):
                return kw
        return None

    def _significant_tokens(self, text: str) -> set:
        STOP = {"the", "of", "in", "a", "an", "and", "at", "on", "by", "to", "for", "its"}
        return {w for w in re.sub(r'[^\w\s]', '', text.lower()).split()
                if len(w) >= 3 and w not in STOP}

    def _matches_answer(self, guess: str) -> bool:
        gc = re.sub(r'[^\w\s]', '', guess.lower()).strip()
        gc_tokens = self._significant_tokens(guess)
        for ans in self.state.acceptable_answers:
            ac = re.sub(r'[^\w\s]', '', ans.lower()).strip()
            if ac and gc and (ac in gc or gc in ac):
                return True
            ac_tokens = self._significant_tokens(ans)
            if not ac_tokens:
                continue
            short, long_ = (
                (ac_tokens, gc_tokens) if len(ac_tokens) <= len(gc_tokens)
                else (gc_tokens, ac_tokens)
            )
            if short and short <= long_:
                return True
        return False


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class ChronicleScorer(GameScorer):

    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)

    def compute_scores(self, episode_interactions: Dict) -> None:
        turns = episode_interactions.get("turns", [])
        aborted = False
        success = False
        success_turn = None
        turn_violations: Dict[int, int] = {}

        for turn_idx, turn in enumerate(turns):
            turn_violations[turn_idx] = 0
            for event in turn:
                action = event.get("action", {})
                t = action.get("type", "")
                if t == "invalid format":
                    aborted = True
                elif t == "forbidden keyword violation":
                    turn_violations[turn_idx] = 1
                elif t == "correct guess":
                    success = True
                    success_turn = turn_idx + 1

        for t_idx in range(len(turns)):
            self.log_turn_score(t_idx, "narrator_violation", turn_violations.get(t_idx, 0))

        n_turns = len(turns) if turns else 1
        mean_viol = sum(turn_violations.values()) / n_turns

        self.log_episode_score(METRIC_ABORTED, 1 if aborted else 0)
        self.log_episode_score(METRIC_SUCCESS, 1 if success else 0)
        self.log_episode_score(METRIC_LOSE, 1 if (not success and not aborted) else 0)
        self.log_episode_score("narrator_mean_violation_rate", mean_viol)
        self.log_episode_score(
            "detective_turns_to_success",
            float(success_turn) if success_turn else np.nan,
        )

        if aborted:
            bench = np.nan
        elif not success:
            bench = 0.0
        else:
            T = success_turn
            # T=1→100, T=2→90, ..., T=10→10
            speed = max(10, 110 - T * 10)
            bench = float(speed) * (1.0 - mean_viol)

        self.log_episode_score(BENCH_SCORE, bench)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

class ChronicleGameBenchmark(GameBenchmark):

    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(
        self, experiment: Dict, player_models: List
    ) -> ChronicleGameMaster:
        return ChronicleGameMaster(self.game_spec, experiment, player_models)

    def create_game_scorer(
        self, experiment: Dict, game_instance: Dict
    ) -> ChronicleScorer:
        return ChronicleScorer(self.game_name, experiment, game_instance)
