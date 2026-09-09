"""Generate SAT-Menu game instances."""

import os
import random

from clemcore.clemgame import GameInstanceGenerator

from utils import (find_greedy_trap,
                   find_satisfying_assignment_distance_counts,
                   sample_3sat_formula)

DEFAULT_SEED = 73
FRAMINGS = ['CNF', 'MENU'] # direct formula and menu repair representations
DIFFICULTY = ['easy', 'hard', 'trap'] # repair landscape around the initial assignment
EASY_NUM_VARIABLES = [8, 9, 10] # small formulae with many nearby solutions
HARD_NUM_VARIABLES = [12] # larger formulae that remain feasible to enumerate
TRAP_NUM_VARIABLES = [7] # compact deceptive landscapes for non-reasoning models
EXPERIMENTS = [(framing, 'easy', num_variables)
               for framing in FRAMINGS
               for num_variables in EASY_NUM_VARIABLES]
EXPERIMENTS += [(framing, 'hard', num_variables)
                for framing in FRAMINGS
                for num_variables in HARD_NUM_VARIABLES]
EXPERIMENTS += [('MENU', 'trap', num_variables)
                for num_variables in TRAP_NUM_VARIABLES]

# Easy instances use a sparse formula. Hard instances remain below the phase
# transition. The compact trap set uses a denser formula so a seven-variable
# landscape can still contain globally deceptive first updates.
FORMULA_ALPHAS = {'easy': 2.5, 'hard': 3.5, 'trap': 4.0}
INSTANCES_PER_EXPERIMENT = {'easy': 10, 'hard': 10, 'trap': 15}

# easy instances have many satisfying endpoints close to the initial assignment
EASY_MAX_OPTIMAL_EDIT_DISTANCE = 2
EASY_MIN_NEAR_SOLUTIONS = 6
EASY_MIN_NEAR_PROPORTION = 0.50

# hard instances require a longer repair route through a deceptive landscape
HARD_MIN_OPTIMAL_EDIT_DISTANCE = 6
HARD_MAX_OPTIMAL_EDIT_DISTANCE = 8
HARD_MIN_INITIAL_UNSATISFIED_CLAUSES = 8
HARD_MAX_INITIAL_UNSATISFIED_CLAUSES = 12
HARD_MIN_GREEDY_REMAINING_DISTANCE = 4
HARD_MIN_GREEDY_EXCESS_EDITS = 2

# Trap instances are deliberately small enough for non-reasoning models to
# engage with, but every maximally satisfying first update is globally costly.
# A valid optimal path must begin with a less immediately rewarding update.
TRAP_MIN_OPTIMAL_EDIT_DISTANCE = 4
TRAP_MAX_OPTIMAL_EDIT_DISTANCE = 5
TRAP_MIN_INITIAL_UNSATISFIED_CLAUSES = 4
TRAP_MAX_INITIAL_UNSATISFIED_CLAUSES = 7
TRAP_MIN_GREEDY_REMAINING_DISTANCE = 4
TRAP_MIN_GREEDY_EXCESS_EDITS = 2

# stop when the constraints make valid instances impractical to sample
MAX_GENERATION_ATTEMPTS = 10000


class SATGameInstanceGenerator(GameInstanceGenerator):
    """Generate SAT-Menu experiments and game instances."""

    def __init__(self):
        """Create an instance generator rooted at the SAT-Menu game directory."""

        super().__init__(os.path.dirname(__file__))

    def on_generate(self,
                    seed: int,
                    **kwargs):
        """Create SAT-Menu experiments and their serialized instances.

        Args:
            seed: Random seed used for reproducible instance generation.
            **kwargs: Generation options, including the target language.
        """

        rng = random.Random(seed)
        lang = kwargs.get('lang', 'en')
        lexicon = self.load_json(f'resources/lexicons/{lang}')
        config = self.load_json('resources/experiment_config')
        max_total_changes = config['num_attempts'] * config['max_changes_per_turn']
        if HARD_MAX_OPTIMAL_EDIT_DISTANCE > max_total_changes:
            raise ValueError('The configured repair distance exceeds the available edit budget.')

        # load language-specific food items and participant names.
        food_items = lexicon['food_items']
        people = lexicon['people']
        logical_instances = {}

        # generate logical instances once so both framings receive the same formulas.
        for _, difficulty, num_variables in EXPERIMENTS:
            instance_key = (difficulty, num_variables)
            if instance_key in logical_instances:
                continue

            num_clauses = round(FORMULA_ALPHAS[difficulty] * num_variables)
            if len(food_items) < num_variables or len(people) < num_clauses:
                raise ValueError(f'Lexicon for {lang} is too small for {instance_key}.')
            if (difficulty == 'hard'
                    and HARD_MAX_OPTIMAL_EDIT_DISTANCE > num_variables):
                raise ValueError(
                    f'The configured repair distance exceeds {num_variables} variables.'
                )
            if (difficulty == 'trap'
                    and TRAP_MAX_OPTIMAL_EDIT_DISTANCE > num_variables):
                raise ValueError(
                    f'The configured trap distance exceeds {num_variables} variables.'
                )

            selected_food_items = food_items[:num_variables]
            logical_instances[instance_key] = []
            for game_id in range(1, INSTANCES_PER_EXPERIMENT[difficulty] + 1):
                for _ in range(MAX_GENERATION_ATTEMPTS):
                    greedy_trap = None
                    formula = sample_3sat_formula(num_variables, num_clauses, rng)
                    initial_assignment = {
                        variable: rng.choice([False, True])
                        for variable in range(1, num_variables + 1)
                    }
                    satisfying_distance_counts = find_satisfying_assignment_distance_counts(
                        formula,
                        num_variables,
                        initial_assignment,
                    )
                    if not satisfying_distance_counts:
                        continue

                    optimal_edit_distance = min(satisfying_distance_counts)
                    near_solution_count = (
                        satisfying_distance_counts.get(optimal_edit_distance, 0)
                        + satisfying_distance_counts.get(optimal_edit_distance + 1, 0)
                    )
                    total_solution_count = sum(satisfying_distance_counts.values())
                    near_solution_proportion = near_solution_count / total_solution_count
                    initial_satisfied_clauses = sum(
                        any(initial_assignment[abs(literal)] == (literal > 0)
                            for literal in clause)
                        for clause in formula
                    )
                    initial_unsatisfied_clauses = num_clauses - initial_satisfied_clauses

                    if difficulty == 'easy' and (
                            optimal_edit_distance <= EASY_MAX_OPTIMAL_EDIT_DISTANCE
                            and near_solution_count >= EASY_MIN_NEAR_SOLUTIONS
                            and near_solution_proportion >= EASY_MIN_NEAR_PROPORTION):
                        break
                    if difficulty == 'hard' and (
                            HARD_MIN_OPTIMAL_EDIT_DISTANCE <= optimal_edit_distance
                            <= HARD_MAX_OPTIMAL_EDIT_DISTANCE
                            and HARD_MIN_INITIAL_UNSATISFIED_CLAUSES
                            <= initial_unsatisfied_clauses
                            <= HARD_MAX_INITIAL_UNSATISFIED_CLAUSES):
                        greedy_trap = find_greedy_trap(
                            formula,
                            num_variables,
                            initial_assignment,
                            config['max_changes_per_turn'],
                            HARD_MIN_GREEDY_REMAINING_DISTANCE,
                            HARD_MIN_GREEDY_EXCESS_EDITS
                        )
                        if greedy_trap is not None:
                            break
                    if difficulty == 'trap' and (
                            TRAP_MIN_OPTIMAL_EDIT_DISTANCE <= optimal_edit_distance
                            <= TRAP_MAX_OPTIMAL_EDIT_DISTANCE
                            and TRAP_MIN_INITIAL_UNSATISFIED_CLAUSES
                            <= initial_unsatisfied_clauses
                            <= TRAP_MAX_INITIAL_UNSATISFIED_CLAUSES):
                        greedy_trap = find_greedy_trap(
                            formula,
                            num_variables,
                            initial_assignment,
                            config['max_changes_per_turn'],
                            TRAP_MIN_GREEDY_REMAINING_DISTANCE,
                            TRAP_MIN_GREEDY_EXCESS_EDITS
                        )
                        if greedy_trap is not None:
                            break
                else:
                    raise RuntimeError(
                        f'Could not generate a {difficulty} repair landscape after '
                        f'{MAX_GENERATION_ATTEMPTS} attempts.'
                    )

                initial_menu = [
                    selected_food_items[variable - 1]
                    for variable in range(1, num_variables + 1)
                    if initial_assignment[variable]
                ]
                initial_true_variables = [variable for variable in range(1, num_variables + 1)
                                          if initial_assignment[variable]]

                # map literals to foods and clauses to people by their lexicon indices.
                preferences = []
                for clause_index, clause in enumerate(formula):
                    likes = [selected_food_items[literal - 1] for literal in clause if literal > 0]
                    dislikes = [selected_food_items[abs(literal) - 1] for literal in clause if literal < 0]
                    preferences.append({'name': people[clause_index], 'likes': likes, 'dislikes': dislikes})

                instance_data = {'formula': formula,
                                 'food_items': selected_food_items,
                                 'preferences': preferences,
                                 'initial_true_variables': initial_true_variables,
                                 'initial_menu': initial_menu,
                                 'optimal_edit_distance': optimal_edit_distance,
                                 'satisfying_distance_counts': satisfying_distance_counts}
                if difficulty in {'hard', 'trap'}:
                    instance_data['greedy_trap'] = greedy_trap
                logical_instances[instance_key].append(instance_data)

        # create paired CNF and MENU experiments from the same logical instances.
        for framing, difficulty, num_variables in EXPERIMENTS:
            experiment = self.add_experiment(f'{framing}_{difficulty}_{num_variables}')
            experiment['framing'] = framing
            experiment['difficulty'] = difficulty
            experiment['num_variables'] = num_variables
            experiment['lang'] = lang
            for game_id, instance_data in enumerate(logical_instances[(difficulty, num_variables)], start=1):
                game_instance = self.add_game_instance(experiment, game_id)
                game_instance.update(instance_data)


if __name__ == "__main__":
    SATGameInstanceGenerator().generate(seed=DEFAULT_SEED, lang="en")
