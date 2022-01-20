import glob
import json

from feerci import feerci
from unittest import TestCase

import numpy as np


class TestFeerci(TestCase):
    def test_feerci_happy(self):
        genuines = np.linspace(0, 1, 10000)
        impostors = np.linspace(-0.5, 0.5, 10000)
        np.random.seed(0)
        eer, lower, upper, eers = feerci(impostors, genuines, is_sorted=True)
        assert len(eers) == 10000
        assert eers[0] <= eers[-1]
        assert lower <= upper
        assert lower <= eer <= upper
        assert lower in eers
        assert upper in eers

    def test_feerci_otherm(self):
        genuines = np.linspace(0, 1, 10000)
        impostors = np.linspace(-0.5, 0.5, 10000)
        np.random.seed(0)
        m = 999
        eer, lower, upper, eers = feerci(impostors, genuines, is_sorted=True, m=m)
        assert len(eers) == m
        assert eers[0] <= eers[-1]
        assert lower <= upper
        assert lower <= eer <= upper
        assert lower in eers
        assert upper in eers

    def test_feerci_no_ci(self):
        genuines = np.linspace(0, 1, 100)
        impostors = np.linspace(-0.5, 0.5, 100)
        eer, lower, upper, eers = feerci(impostors, genuines, is_sorted=True, m=0)
        assert len(eers) == 0
        assert eer == 0.2525252401828766
        assert lower is None
        assert upper is None

    def test_feerci_ci_negative(self):
        genuines = np.linspace(0, 1, 100)
        impostors = np.linspace(-0.5, 0.5, 100)
        eer, lower, upper, eers = feerci(impostors, genuines, is_sorted=True, m=-1)
        assert len(eers) == 0
        assert eer == 0.2525252401828766
        assert lower is None
        assert upper is None

    def test_feerci_ci_unsorted(self):
        np.random.seed(0)
        genuines = np.random.rand(100) + 0.5
        impostors = np.random.rand(100)
        eer, lower, upper, eers = feerci(impostors, genuines)
        assert eer == 0.27272728085517883
        assert eer != 0 and eer != 1
        assert lower <= eer <= upper
        assert lower in eers
        assert upper in eers

    def test_feerci_return_threshold(self):
        genuines = np.linspace(0, 1, 100)
        impostors = np.linspace(-0.5, 0.5, 100)

        # default value of return threshold should be False
        out = feerci(impostors, genuines, is_sorted=True)

        assert len(out) == 4

        out = feerci(impostors, genuines, is_sorted=True, return_threshold=False)
        assert len(out) == 4

        out = feerci(impostors, genuines, is_sorted=True, return_threshold=True)
        assert len(out) == 5

    def test_feerci_complex_testvector(self):
        for f in glob.glob("tests/scores/*.json"):
            with open(f) as fh:
                d = json.load(fh)

            eer = d["eer"]
            impostors = np.array(d["impostors"])
            genuines = np.array(d["genuines"])

            calced_eer, eer_cil, eer_ciu, eers, threshold = feerci(impostors, genuines, is_sorted=False, m=1000, return_threshold=True)
            assert calced_eer == eer
