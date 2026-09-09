"""Offline tests: scripted outputs only, no inference or external requests."""
import copy
import json
from pathlib import Path

import tempfile
import unittest

from geolocate.scripts import run_convergence_replay as replay


CONFIG = json.loads((Path(__file__).parent / 'resources/experiment_config.json').read_text())
TEMPLATE = (Path(__file__).parent / 'resources/feedback_prompts/en/geolocate.template').read_text()


def answer(latitude, country='Germany'):
    return f'EXPLANATION: Candidate from evidence.\nLATITUDE: {latitude}\nLONGITUDE: 10\nCOUNTRY: {country}'


class ScriptedModel:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.messages = []

    def generate_response(self, messages):
        self.messages.append(copy.deepcopy(messages))
        output = next(self.outputs)
        if isinstance(output, Exception):
            raise output
        return {}, {'choices': [{'finish_reason': 'stop',
                                'message': {'reasoning_content': 'PRIVATE_REASONING'}}]}, output


def item(used):
    interactions = {'episode_result': {'max_rounds': 3, 'predictions':
                                      [{'distance_km': 900}] * used}}
    allowance = replay.remaining_guess_count(interactions, CONFIG)
    return {'request_fingerprint': 'fixture', 'interactions': interactions,
            'instance': {'target_latitude': 50, 'target_longitude': 10,
                         'target_country': 'Germany', 'target_country_code': 'DE'},
            'request': {'remaining_guesses': allowance, 'feedback_template': TEMPLATE,
                        'messages': [{'role': 'user', 'content': replay.build_closure_prompt(
                            'Original rules', [], 'Visible evidence', allowance),
                            'image': ['a', 'b', 'c', 'd']}]}}


def test_exact_remaining_allowance_and_final_guess_scoring(tmp_path, used):
    data = item(used)
    remaining = 3 - used
    model = ScriptedModel([answer(51 + i) for i in range(remaining)])
    result = replay.run_remaining_guesses(data, tmp_path, lambda: model, CONFIG)
    assert len(model.messages) == remaining
    assert result['termination'] == 'guesses_exhausted'
    assert result['prediction']['latitude'] == 51 + remaining - 1
    assert result['counterfactual_clemscore'] == result['recovery_rounds'][-1]['score']['inverse_distance_quality']
    if remaining > 1:
        assert result['counterfactual_clemscore'] < result['recovery_rounds'][0]['score']['inverse_distance_quality']
        assert 'DISTANCE_FEEDBACK:' in model.messages[1][-1]['content']
        assert 'PREDICTIONS_REMAINING: ' + str(remaining - 1) in model.messages[1][-1]['content']
        assert 'PRIVATE_REASONING' not in json.dumps(model.messages)
        feedback = model.messages[1][-1]['content']
        assert '900' not in feedback
        assert str(result['recovery_rounds'][0]['score']['distance_km']) not in feedback
    assert result['recovery_rounds'][0]['game_round'] == used + 1


def test_success_stops_early_and_wrong_country_does_not(tmp_path):
    model = ScriptedModel([answer(50, 'France'), answer(50)])
    result = replay.run_remaining_guesses(item(0), tmp_path, lambda: model, CONFIG)
    assert len(model.messages) == 2
    assert result['termination'] == 'target_reached'
    assert result['counterfactual_clemscore'] == 100
    assert 'COUNTRY_FEEDBACK: incorrect' in model.messages[1][-1]['content']


def test_invalid_later_guess_aborts_without_salvaging_previous_guess(tmp_path):
    model = ScriptedModel([answer(51), '51, 10'])
    result = replay.run_remaining_guesses(item(0), tmp_path, lambda: model, CONFIG)
    assert len(model.messages) == 2
    assert result['status'] == 'invalid_response'
    assert result['valid_response'] is False
    assert result['counterfactual_clemscore'] == 0
    assert 'score' not in result


def test_resume_keeps_completed_rounds_and_retries_only_failed_request(tmp_path):
    data = item(1)
    first = ScriptedModel([answer(51), ConnectionError('offline fixture')])
    result = replay.run_remaining_guesses(data, tmp_path, lambda: first, CONFIG)
    assert result['status'] == 'request_error'
    second = ScriptedModel([answer(52)])
    resumed = replay.run_remaining_guesses(data, tmp_path, lambda: second, CONFIG)
    assert len(second.messages) == 1
    assert resumed['prediction']['latitude'] == 52
    assert len(resumed['recovery_rounds']) == 2
    def forbidden():
        raise AssertionError('No request should be repeated')
    assert replay.run_remaining_guesses(data, tmp_path, forbidden, CONFIG) == resumed


def test_feedback_uses_previous_pre_timeout_prediction():
    score = {'distance_km': 100.123, 'country_correct': True}
    text = replay.recovery_feedback(score, 900, 1, CONFIG, TEMPLATE)
    assert 'closer than the previous prediction' in text
    assert '100-500 km' in text
    assert '100.123' not in text
    assert '$' not in text


def test_reject_exhausted_allowance():
    with unittest.TestCase().assertRaisesRegex(ValueError, 'No legal'):
        item(3)


class TestRemainingGuessReplay(unittest.TestCase):
    def test_allowances(self):
        for used in (0, 1, 2):
            with self.subTest(used=used), tempfile.TemporaryDirectory() as directory:
                test_exact_remaining_allowance_and_final_guess_scoring(Path(directory), used)

    def test_success_country_and_invalid_response(self):
        for test in (test_success_stops_early_and_wrong_country_does_not,
                     test_invalid_later_guess_aborts_without_salvaging_previous_guess):
            with self.subTest(test=test.__name__), tempfile.TemporaryDirectory() as directory:
                test(Path(directory))

    def test_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            test_resume_keeps_completed_rounds_and_retries_only_failed_request(Path(directory))

    def test_feedback_and_exhaustion(self):
        test_feedback_uses_previous_pre_timeout_prediction()
        test_reject_exhausted_allowance()
