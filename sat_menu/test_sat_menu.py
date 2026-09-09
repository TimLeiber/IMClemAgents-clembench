"""Tests for SAT-Menu. Run: python test_sat_menu.py."""

import unittest
from pathlib import Path

from clemcore.backends import CustomResponseModel
from clemcore.clemgame import GameSpec
from clemcore.clemgame.master import Outcome
from clemcore.clemgame.metrics import BENCH_SCORE

from instancegenerator import SATGameInstanceGenerator
from master import SATGameBenchmark, SATGameMaster, SATGameScorer, SATGameState


GAME_PATH = Path(__file__).resolve().parent


def create_master(experiment: dict,
                  game_instance: dict) -> SATGameMaster:
    """Create an initialised SAT-Menu master backed by a programmatic solver.

    Args:
        experiment: Generated experiment configuration.
        game_instance: One generated SAT-Menu instance.

    Returns:
        Initialised SAT-Menu master backed by a programmatic solver.
    """

    game_spec = GameSpec.from_dict({
        'game_name': 'sat_menu',
        'description': 'Single-player 3-SAT game',
        'main_game': 'sat_menu',
        'players': 1,
        'languages': ['en'],
        'benchmark': ['3.0'],
        'game_path': str(GAME_PATH)
    })
    master = SATGameMaster(game_spec, experiment, [CustomResponseModel()])
    master.setup(**game_instance)
    master.before_game()
    return master


def create_test_master(initial_menu: list[str] | None = None,
                       framing: str = 'MENU') -> SATGameMaster:
    """Create a deterministic SAT-Menu repair master for one correction test."""

    food_items = ['apple pie', 'cherry soup', 'pear juice']
    initial_menu = initial_menu or []
    initial_true_variables = [index + 1 for index, item in enumerate(food_items)
                              if item in initial_menu]
    experiment = {
        'name': 'menu_repair_test',
        'framing': framing,
        'lang': 'en',
        'num_variables': 3
    }
    formula = [[1, 2, 3], [-1, -2, -3]]
    game_instance = {
        'game_id': 1,
        'formula': formula,
        'food_items': food_items,
        'preferences': [
            {'name': 'Ada', 'likes': ['apple pie', 'cherry soup', 'pear juice'], 'dislikes': []},
            {'name': 'Bert', 'likes': [], 'dislikes': ['apple pie', 'cherry soup', 'pear juice']}
        ],
        'initial_true_variables': initial_true_variables,
        'initial_menu': initial_menu,
        'optimal_edit_distance': 1
    }
    return create_master(experiment, game_instance)


class SATGameTestCase(unittest.TestCase):
    """Check SAT-Menu setup, correction, and deterministic mock behaviour."""

    def test_required_classes_are_defined(self):
        """Import the generator, benchmark, master, scorer, and state classes."""

        self.assertIsNotNone(SATGameBenchmark)
        self.assertIsNotNone(SATGameInstanceGenerator)
        self.assertIsNotNone(SATGameMaster)
        self.assertIsNotNone(SATGameScorer)
        self.assertIsNotNone(SATGameState)

    def test_menu_mock_repairs_the_persistent_menu(self):
        """Check that the mock receives feedback before repairing the menu."""

        master = create_test_master()
        initial_prompt = master.get_context_for(master.solver)['content']
        self.assertIn('Current menu:', initial_prompt)
        self.assertIn('at most\n2 menu items', initial_prompt)
        self.assertIn('EXPLANATION:', initial_prompt)
        self.assertIn('ADD:', initial_prompt)
        self.assertIn('REMOVE:', initial_prompt)
        self.assertNotIn('$', initial_prompt)
        first_response = master.solver(master.get_context_for(master.solver))
        done, _ = master.step(first_response)

        self.assertFalse(done)
        self.assertTrue(first_response.startswith('ADD:'))
        self.assertTrue(first_response.endswith('EXPLANATION: I will first inspect the current state.'))
        self.assertEqual(master.state.current_menu, [])
        self.assertEqual(master.state.changes_used, 0)
        self.assertIn('Submit another update', master.get_context_for(master.solver)['content'])
        self.assertIn('Ada: unhappy', master.get_context_for(master.solver)['content'])
        self.assertIn('Bert: happy', master.get_context_for(master.solver)['content'])
        self.assertIn('14 turns remaining', master.get_context_for(master.solver)['content'])
        self.assertNotIn('$', master.get_context_for(master.solver)['content'])

        second_response = master.solver(master.get_context_for(master.solver))
        done, _ = master.step(second_response)

        self.assertTrue(done)
        self.assertEqual(master.state.outcome, Outcome.SUCCESS)
        self.assertEqual(master.state.attempts_used, 2)

    def test_update_cannot_repeat_an_existing_menu_item(self):
        """Check that additions are interpreted relative to the current menu."""

        master = create_test_master(['apple pie'])
        response = '\n'.join([
            'ADD: apple pie',
            'REMOVE: NONE',
            'EXPLANATION: This response is intentionally invalid.'
        ])
        master.step(response)
        self.assertTrue(master.state.invalid_response)
        self.assertEqual(master.state.outcome, Outcome.ABORTED)

    def test_cnf_mock_receives_literal_clause_feedback(self):
        """Check the CNF update format and literal clause feedback."""

        master = create_test_master(framing='CNF')
        initial_prompt = master.get_context_for(master.solver)['content']
        self.assertIn('SET TRUE:', initial_prompt)
        self.assertIn('SET FALSE:', initial_prompt)
        self.assertIn('[1, 2, 3]', initial_prompt)
        self.assertNotIn('$', initial_prompt)

        first_response = master.solver(master.get_context_for(master.solver))
        done, _ = master.step(first_response)
        feedback = master.get_context_for(master.solver)['content']

        self.assertFalse(done)
        self.assertIn('[1, 2, 3]: not satisfied', feedback)
        self.assertIn('[-1, -2, -3]: satisfied', feedback)
        self.assertIn('14 turns remaining', feedback)
        self.assertNotIn('Clause 1', feedback)

        second_response = master.solver(master.get_context_for(master.solver))
        done, _ = master.step(second_response)

        self.assertTrue(done)
        self.assertEqual(master.state.outcome, Outcome.SUCCESS)
        self.assertTrue(second_response.startswith('SET TRUE:'))
        self.assertIn('\nSET FALSE:', second_response)
        self.assertIn('\nEXPLANATION:', second_response)

    def test_success_score_combines_edit_and_turn_efficiency(self):
        """Check the normalized arithmetic-mean score for a successful repair."""

        scorer = SATGameScorer('sat_menu', {}, {})
        scorer.compute_episode_scores({
            'Aborted': 0,
            'Success': 1,
            'episode_result': {
                'satisfaction': [True, True],
                'attempts_used': 4,
                'changes_used': 7,
                'optimal_edit_distance': 5,
                'max_changes_per_turn': 2
            }
        })
        episode_scores = scorer.scores['episode scores']
        edit_efficiency = 5 / 7
        turn_efficiency = 3 / 4
        expected_main_score = 100 * (edit_efficiency + turn_efficiency) / 2

        self.assertAlmostEqual(episode_scores[BENCH_SCORE], expected_main_score)
        self.assertAlmostEqual(episode_scores['Edit Efficiency'], 100 * edit_efficiency)
        self.assertAlmostEqual(episode_scores['Turn Efficiency'], 100 * turn_efficiency)


if __name__ == '__main__':
    unittest.main()
