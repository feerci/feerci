from feerci import feer
from unittest import TestCase

import numpy as np


class TestFeer(TestCase):

    def test_feer_happy(self):
        genuines = np.linspace(0, 1, 100)
        impostors = np.linspace(-.5, .5, 100)
        eer = feer(impostors, genuines, is_sorted=True)
        assert eer == 0.2525252401828766

    def test_feer_wrongly_unsorted(self):
        np.random.seed(0)
        genuines = np.random.rand(100)
        impostors = np.random.rand(100)
        """
        There is no way to test for sorted-ness in constant time. If provided with an unsorted list while claiming it's sorted, just accept whatever is being given. It is likely this will result in a failed overlap check, giving an EER of 0 or 1.
        """
        eer = feer(impostors, genuines, is_sorted=True)
        # assert eer == 0.2525252401828766
