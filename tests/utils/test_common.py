import functools
import operator
import unittest

from kgm.utils.common import from_dot, generate_experiments, to_dot


class ExperimentGenerationTests(unittest.TestCase):
    """unittests for experiment generation utilities."""
    num_workers: int = 3

    def setUp(self) -> None:
        self.parameter_grid = {
            'a': [1, 2],
            'b': [None, 'A', 'B'],
        }
        self.explicit = [
            {'a': 4, 'b': 'C'},
        ]

    def test_generate_experiments(self):
        experiments = generate_experiments(
            grid_params=self.parameter_grid,
            explicit=self.explicit,
        )

        num_experiments = (len(self.explicit) + functools.reduce(operator.mul, map(len, self.parameter_grid.values())))
        assert len(experiments) == num_experiments


class DotConversionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.nested_dictionary = {
            'a': {
                'b': 1,
                'c': {
                    'd': 2,
                    'e': 3,
                }
            }
        }
        self.flat_dictionary = {
            'a.b': 1,
            'a.c.d': 2,
            'a.c.e': 3,
        }

    def test_to_dot(self):
        assert self.flat_dictionary == to_dot(self.nested_dictionary, separator='.')

    def test_from_dot(self):
        assert from_dot(self.flat_dictionary, separator='.') == self.nested_dictionary
