"""SAT-Menu game implementation."""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from clemcore.backends import Model
from clemcore.clemgame import GameBenchmark, GameScorer, GameSpec, ParseError, Player
from clemcore.clemgame.master import DialogueGameMaster, GameState, Outcome
from clemcore.clemgame.metrics import BENCH_SCORE, METRIC_ABORTED, METRIC_LOSE, METRIC_SUCCESS

from utils import (evaluate_formula, find_nearest_satisfying_assignment,
                   parse_assignment_update, parse_menu_update)


GAME_NAME = "sat_menu"


@dataclass
class SATGameState(GameState):
    """Store SAT-Menu state for one episode."""

    framing: Optional[str] = None # whether the direct CNF or MENU representation is used
    formula: Optional[List[List[int]]] = None # complete 3-CNF formula
    food_items: Optional[List[str]] = None # menu items corresponding to formula variables
    preferences: Optional[List[Dict]] = None # participant preferences corresponding to formula clauses
    initial_true_variables: Optional[List[int]] = None # variables initially assigned true
    current_true_variables: Optional[List[int]] = None # variables currently assigned true
    initial_menu: Optional[List[str]] = None # menu provided at the start of the episode
    current_menu: Optional[List[str]] = None # menu after the most recent accepted update
    optimal_edit_distance: Optional[int] = None # fewest edits needed to repair the initial menu
    num_attempts: Optional[int] = None # maximum number of responses allowed
    max_changes_per_turn: Optional[int] = None # maximum additions and removals in one update
    attempts_used: int = 0 # number of accepted responses so far
    changes_used: int = 0 # number of menu edits accepted across the episode
    changes_to_true: Optional[List[int]] = None # variables made true by the current response
    changes_to_false: Optional[List[int]] = None # variables made false by the current response
    explanation: Optional[str] = None # explanation supplied with the current response
    satisfaction: Optional[List[bool]] = None # status of every participant
    invalid_response: bool = False # whether a response violates the required format

    def __post_init__(self):
        """Initialise the shared Clembench game state."""

        super().__init__()


class SATPlayer(Player):
    """Single player responsible for solving one SAT-Menu instance."""

    def __init__(self,
                 model: Model,
                 framing: str,
                 formula: List[List[int]],
                 food_items: List[str],
                 initial_true_variables: List[int],
                 max_changes_per_turn: int,
                 response_format: Dict[str, str]):
        """Create the solver player.

        Args:
            model: Model that produces the solver's responses.
            framing: Direct CNF or menu representation used in the episode.
            formula: Complete 3-CNF formula represented by this instance.
            food_items: Menu items mapped to the formula variables.
            initial_true_variables: Variables assigned true at the start.
            max_changes_per_turn: Maximum number of item edits in one update.
            response_format: Language-specific labels and no-change value.
        """

        super().__init__(model, game_role="Solver")
        self.framing = framing
        self.formula = formula
        self.food_items = food_items
        self.current_true_variables = set(initial_true_variables)
        self.max_changes_per_turn = max_changes_per_turn
        self.response_format = response_format
        self.mock_attempts = 0
        initial_assignment = {
            variable: variable in self.current_true_variables
            for variable in range(1, len(food_items) + 1)
        }
        assignment, _ = find_nearest_satisfying_assignment(
            formula,
            len(food_items),
            initial_assignment,
        )
        if assignment is None:
            raise ValueError('Repair instances must have a satisfying assignment.')
        self.target_true_variables = {variable for variable, value in assignment.items() if value}

    def _custom_response(self,
                         context: Dict) -> str:
        """Submit one repair update toward a deterministic solution.

        Args:
            context: Most recent game message supplied to the player.

        Returns:
            A valid SAT-Menu response for a programmatic test model.
        """

        # submit a no-op update first so tests observe correction feedback
        if self.mock_attempts == 0:
            changes_to_true = []
            changes_to_false = []
            explanation = 'I will first inspect the current state.'
        else:
            changes_to_true = [variable for variable in range(1, len(self.food_items) + 1)
                               if variable in self.target_true_variables
                               and variable not in self.current_true_variables]
            changes_to_false = [variable for variable in range(1, len(self.food_items) + 1)
                                if variable in self.current_true_variables
                                and variable not in self.target_true_variables]
            changes = changes_to_true + changes_to_false
            changes = changes[:self.max_changes_per_turn]
            changes_to_true = [variable for variable in changes if variable in changes_to_true]
            changes_to_false = [variable for variable in changes if variable in changes_to_false]
            self.current_true_variables.update(changes_to_true)
            self.current_true_variables.difference_update(changes_to_false)
            explanation = 'I am applying the next repair step.'

        self.mock_attempts += 1
        no_change = self.response_format['no_change']

        if self.framing == 'menu':
            positive_values = [self.food_items[variable - 1] for variable in changes_to_true]
            negative_values = [self.food_items[variable - 1] for variable in changes_to_false]
            positive_label = self.response_format['add_label']
            negative_label = self.response_format['remove_label']
        else:
            positive_values = [str(variable) for variable in changes_to_true]
            negative_values = [str(variable) for variable in changes_to_false]
            positive_label = self.response_format['set_true_label']
            negative_label = self.response_format['set_false_label']

        positive_text = ', '.join(positive_values) if positive_values else no_change
        negative_text = ', '.join(negative_values) if negative_values else no_change
        return (
            f"{positive_label}: {positive_text}\n"
            f"{negative_label}: {negative_text}\n"
            f"{self.response_format['explanation_label']}: {explanation}"
        )


class SATGameMaster(DialogueGameMaster):
    """Coordinate one SAT-Menu episode."""

    def __init__(self,
                 game_spec: GameSpec,
                 experiment: Dict,
                 player_models: List[Model]):
        """Create a SAT-Menu game master.

        Args:
            game_spec: Registered specification for the SAT-Menu game.
            experiment: Experiment configuration loaded from instances.json.
            player_models: Models assigned to the game's player slots.
        """

        super().__init__(game_spec, experiment, player_models)

    def _on_setup(self,
                  **game_instance):
        """Load one generated SAT-Menu instance and initialise the solver."""

        # load game, representation, and language-specific configuration
        self.game_instance = game_instance
        config = self.load_json('resources/experiment_config')
        lang = self.experiment['lang']
        framing = self.experiment['framing'].lower()
        framing_config = config['representations'][framing]
        lang_config = self.load_json('resources/langconfig')[lang]
        response_format = lang_config['response_formats'][framing]
        feedback_config = lang_config['feedback'][framing]

        # render the initial prompt for the selected problem representation
        initial_prompt = self.load_template(framing_config['initial_prompt'].format(lang=lang))
        initial_prompt = initial_prompt.replace('$NUM_ATTEMPTS$', str(config['num_attempts']))
        initial_prompt = initial_prompt.replace(
            '$MAX_CHANGES_PER_TURN$',
            str(config['max_changes_per_turn'])
        )
        initial_prompt = initial_prompt.replace(
            '$EXPLANATION_LABEL$',
            response_format['explanation_label']
        )
        initial_prompt = initial_prompt.replace('$NO_CHANGE$', response_format['no_change'])

        # render representation-specific problem data and response labels
        if framing == 'menu':
            preference_labels = lang_config['preference_labels']
            formatted_preferences = []
            for preference in game_instance['preferences']:
                likes = ', '.join(preference['likes']) or preference_labels['none']
                dislikes = ', '.join(preference['dislikes']) or preference_labels['none']
                formatted_preferences.append(f"{preference['name']}: {preference_labels['likes']}: {likes}. "
                                             f"{preference_labels['dislikes']}: {dislikes}.")
            initial_prompt = initial_prompt.replace('$MENU_ITEMS$', ', '.join(game_instance['food_items']))
            initial_prompt = initial_prompt.replace('$PREFERENCES$', '\n'.join(formatted_preferences))
            initial_menu = ', '.join(game_instance['initial_menu']) or response_format['no_change']
            initial_prompt = initial_prompt.replace('$INITIAL_MENU$', initial_menu)
            initial_prompt = initial_prompt.replace('$ADD_LABEL$', response_format['add_label'])
            initial_prompt = initial_prompt.replace('$REMOVE_LABEL$', response_format['remove_label'])
        else:
            formula = '\n'.join(str(clause) for clause in game_instance['formula'])
            initial_assignment = ', '.join(
                str(variable) for variable in game_instance['initial_true_variables']
            ) or response_format['no_change']
            initial_prompt = initial_prompt.replace('$NUM_VARIABLES$', str(self.experiment['num_variables']))
            initial_prompt = initial_prompt.replace('$FORMULA$', formula)
            initial_prompt = initial_prompt.replace('$INITIAL_ASSIGNMENT$', initial_assignment)
            initial_prompt = initial_prompt.replace('$SET_TRUE_LABEL$', response_format['set_true_label'])
            initial_prompt = initial_prompt.replace('$SET_FALSE_LABEL$', response_format['set_false_label'])

        # retain the instance data and attempt state for the current episode
        self.state = SATGameState(
            framing=framing,
            formula=game_instance['formula'],
            food_items=game_instance['food_items'],
            preferences=game_instance['preferences'],
            initial_true_variables=game_instance['initial_true_variables'],
            current_true_variables=list(game_instance['initial_true_variables']),
            initial_menu=game_instance['initial_menu'],
            current_menu=list(game_instance['initial_menu']),
            optimal_edit_distance=game_instance['optimal_edit_distance'],
            num_attempts=config['num_attempts'],
            max_changes_per_turn=config['max_changes_per_turn']
        )

        # retain response resources needed by subsequent validation and feedback steps
        self.response_format = response_format
        self.feedback_config = feedback_config
        self.feedback_template = self.load_template(framing_config['feedback_prompt'].format(lang=lang))
        self.feedback_template = self.feedback_template.replace(
            '$EXPLANATION_LABEL$',
            response_format['explanation_label']
        )
        self.feedback_template = self.feedback_template.replace('$NO_CHANGE$', response_format['no_change'])

        if framing == 'menu':
            self.feedback_template = self.feedback_template.replace('$ADD_LABEL$', response_format['add_label'])
            self.feedback_template = self.feedback_template.replace('$REMOVE_LABEL$', response_format['remove_label'])
        else:
            self.feedback_template = self.feedback_template.replace(
                '$SET_TRUE_LABEL$',
                response_format['set_true_label']
            )
            self.feedback_template = self.feedback_template.replace(
                '$SET_FALSE_LABEL$',
                response_format['set_false_label']
            )

        # register the single solver with the rendered game prompt
        self.solver = SATPlayer(
            self.player_models[0],
            self.state.framing,
            self.state.formula,
            self.state.food_items,
            self.state.initial_true_variables,
            self.state.max_changes_per_turn,
            self.response_format
        )
        self.add_player(self.solver, initial_context=initial_prompt)

    def _parse_response(self,
                        player: Player,
                        response: str) -> str:
        """Parse one solver response before it is accepted."""

        # parse one structured update without applying it to the current state
        try:
            if self.state.framing == 'menu':
                update = parse_menu_update(
                    response,
                    self.response_format,
                    self.state.food_items,
                )
                changes_to_true = [self.state.food_items.index(item) + 1
                                   for item in update['additions']]
                changes_to_false = [self.state.food_items.index(item) + 1
                                    for item in update['removals']]
                positive_changes = update['additions']
                negative_changes = update['removals']
            else:
                update = parse_assignment_update(
                    response,
                    self.response_format,
                    len(self.state.food_items),
                )
                changes_to_true = update['set_true']
                changes_to_false = update['set_false']
                positive_changes = changes_to_true
                negative_changes = changes_to_false
        except ValueError as error:
            raise ParseError(str(error), response=response, key='invalid_response') from error

        current_true_variables = set(self.state.current_true_variables)
        if len(changes_to_true) + len(changes_to_false) > self.state.max_changes_per_turn:
            raise ParseError('Response exceeds the maximum number of changes.',
                             response=response,
                             key='too_many_changes')
        if set(changes_to_true).intersection(current_true_variables):
            raise ParseError('Response attempts to make an already true value true.',
                             response=response,
                             key='redundant_positive_change')
        if not set(changes_to_false).issubset(current_true_variables):
            raise ParseError('Response attempts to make an already false value false.',
                             response=response,
                             key='redundant_negative_change')

        # retain each valid update for deterministic evaluation
        self.state.changes_to_true = changes_to_true
        self.state.changes_to_false = changes_to_false
        self.state.explanation = update['explanation']
        self.state.attempts_used += 1
        self.state.changes_used += len(changes_to_true) + len(changes_to_false)
        self.log_to_self('submitted repair update', {
            'framing': self.state.framing,
            'response': response,
            'explanation': self.state.explanation,
            'positive_changes': positive_changes,
            'negative_changes': negative_changes,
            'true_variables_before_update': self.state.current_true_variables,
            'menu_before_update': self.state.current_menu if self.state.framing == 'menu' else None
        })
        return response

    def _advance_game(self,
                      player: Player,
                      parsed_response: str):
        """Handle one valid solver response."""

        # apply the valid update to the persistent assignment before evaluation
        current_true_variables = set(self.state.current_true_variables)
        current_true_variables.update(self.state.changes_to_true)
        current_true_variables.difference_update(self.state.changes_to_false)
        self.state.current_true_variables = sorted(current_true_variables)
        self.state.current_menu = [
            self.state.food_items[variable - 1]
            for variable in self.state.current_true_variables
        ]
        self.state.satisfaction = evaluate_formula(self.state.formula, self.state.current_true_variables)
        is_correct = all(self.state.satisfaction)

        if self.state.framing == 'menu':
            satisfaction_summary = '\n'.join(
                f"{preference['name']}: "
                f"{self.feedback_config['satisfied'] if satisfied else self.feedback_config['unsatisfied']}"
                for preference, satisfied in zip(self.state.preferences, self.state.satisfaction)
            )
        else:
            satisfaction_summary = '\n'.join(
                f"{clause}: "
                f"{self.feedback_config['satisfied'] if satisfied else self.feedback_config['unsatisfied']}"
                for clause, satisfied in zip(self.state.formula, self.state.satisfaction)
            )

        self.log_to_self('repair update evaluation', {
            'framing': self.state.framing,
            'current_true_variables': self.state.current_true_variables,
            'current_menu': self.state.current_menu if self.state.framing == 'menu' else None,
            'satisfaction': self.state.satisfaction,
            'correct': is_correct,
            'changes_used': self.state.changes_used
        })

        # end episodes as soon as the player repairs the menu
        if is_correct:
            self.log_to_self('correct repair', {
                'current_true_variables': self.state.current_true_variables,
                'current_menu': self.state.current_menu if self.state.framing == 'menu' else None
            })
            self.state.succeed()
            return

        # end episodes after the final valid attempt
        if self.state.attempts_used >= self.state.num_attempts:
            self.log_to_self('attempt limit reached', str(self.state.num_attempts))
            self.state.failed()
            return

        # provide deterministic feedback before the next attempt
        if self.state.framing == 'menu':
            feedback = self.feedback_template.replace('$PARTICIPANT_SATISFACTION$', satisfaction_summary)
            current_menu = ', '.join(self.state.current_menu) or self.response_format['no_change']
            feedback = feedback.replace('$CURRENT_MENU$', current_menu)
        else:
            feedback = self.feedback_template.replace('$CLAUSE_SATISFACTION$', satisfaction_summary)
            current_assignment = ', '.join(
                str(variable) for variable in self.state.current_true_variables
            ) or self.response_format['no_change']
            feedback = feedback.replace('$CURRENT_ASSIGNMENT$', current_assignment)
        feedback = feedback.replace('$REMAINING_ATTEMPTS$', str(
            self.state.num_attempts - self.state.attempts_used
        ))
        feedback = feedback.replace('$MAX_CHANGES_PER_TURN$', str(self.state.max_changes_per_turn))
        self.set_context_for(player, feedback)

    def _on_parse_error(self,
                        error: ParseError):
        """Abort episodes whose response cannot be parsed."""

        # record the invalid response before terminating the episode
        self.state.invalid_response = True
        self.log_to_self('invalid response', {'response': error.response, 'error': error.reason})
        self.state.abort()

    def _does_game_proceed(self) -> bool:
        """Determine whether another game round should be played."""

        return super()._does_game_proceed()

    def _on_after_game(self):
        """Store the final state required for deterministic scoring."""

        # retain outcome and proposal data for the episode scorer
        self.log_key(METRIC_ABORTED, int(self.state.outcome == Outcome.ABORTED))
        self.log_key(METRIC_LOSE, int(self.state.outcome == Outcome.FAILURE))
        self.log_key(METRIC_SUCCESS, int(self.state.outcome == Outcome.SUCCESS))
        self.log_key('episode_result', {
            'outcome': self.state.outcome.value,
            'framing': self.state.framing,
            'attempts_used': self.state.attempts_used,
            'changes_used': self.state.changes_used,
            'optimal_edit_distance': self.state.optimal_edit_distance,
            'max_changes_per_turn': self.state.max_changes_per_turn,
            'initial_true_variables': self.state.initial_true_variables,
            'current_true_variables': self.state.current_true_variables,
            'initial_menu': self.state.initial_menu,
            'current_menu': self.state.current_menu,
            'satisfaction': self.state.satisfaction,
            'invalid_response': self.state.invalid_response
        })


class SATGameScorer(GameScorer):
    """Compute SAT-Menu scores from recorded interactions."""

    def compute_round_score(self,
                            round_idx: int,
                            round_events: List[Dict]) -> None:
        """Compute diagnostics for one game round."""

        evaluation = None
        for event in round_events:
            if event['action']['type'] == 'repair update evaluation':
                evaluation = event['action']['content']

        if evaluation is None:
            return

        satisfaction = evaluation['satisfaction']
        satisfaction_rate = np.nan if satisfaction is None else sum(satisfaction) / len(satisfaction)
        self.log_round_score(round_idx, 'Satisfaction Rate', satisfaction_rate)
        self.log_round_score(round_idx, 'Correct Menu', int(evaluation['correct']))
        self.log_round_score(round_idx, 'Changes Used', evaluation['changes_used'])

    def compute_episode_scores(self,
                               interactions: Dict) -> None:
        """Compute episode-level SAT-Menu scores."""

        episode = interactions['episode_result']
        satisfaction = episode['satisfaction']
        satisfaction_rate = np.nan if satisfaction is None else sum(satisfaction) / len(satisfaction)

        if interactions[METRIC_ABORTED]:
            main_score = np.nan
            edit_efficiency = np.nan
            turn_efficiency = np.nan
        elif interactions[METRIC_SUCCESS]:
            optimal_edit_distance = episode['optimal_edit_distance']
            minimum_possible_turns = (
                optimal_edit_distance + episode['max_changes_per_turn'] - 1
            ) // episode['max_changes_per_turn']
            edit_efficiency = optimal_edit_distance / episode['changes_used']
            turn_efficiency = minimum_possible_turns / episode['attempts_used']
            main_score = 100 * (edit_efficiency + turn_efficiency) / 2
        else:
            main_score = 0
            edit_efficiency = 0
            turn_efficiency = 0

        self.log_episode_score(BENCH_SCORE, main_score)
        self.log_episode_score('Edit Efficiency', 100 * edit_efficiency)
        self.log_episode_score('Turn Efficiency', 100 * turn_efficiency)
        self.log_episode_score('Turns Used', episode['attempts_used'])
        self.log_episode_score('Changes Used', episode['changes_used'])
        self.log_episode_score('Optimal Edit Distance', episode['optimal_edit_distance'])
        self.log_episode_score('Final Satisfaction Rate', satisfaction_rate)


class SATGameBenchmark(GameBenchmark):
    """Register SAT-Menu with Clembench's benchmark runner."""

    def create_game_master(self,
                           experiment: Dict,
                           player_models: List[Model]) -> DialogueGameMaster:
        """Create the game master for one SAT-Menu episode."""

        return SATGameMaster(self.game_spec, experiment, player_models)

    def create_game_scorer(self,
                           experiment: Dict,
                           game_instance: Dict) -> GameScorer:
        """Create the scorer for one SAT-Menu episode."""

        return SATGameScorer(GAME_NAME, experiment, game_instance)
