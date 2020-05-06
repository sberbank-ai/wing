# -*- coding: utf-8 -*-

from .functions import make_edges, generate_combs, check_variant, add_infinity, split_by_edges, check_mono
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.tree import DecisionTreeClassifier, _tree

LIST_OF_ALGOS = ['full-search', 'tree-binning']


class WingOptimizer:
    def __init__(self, x: np.ndarray, y: np.ndarray, total_good: int, total_bad: int, n_initial: int, n_target: int,
                 bin_minimal_size: float, bin_size_increase: float, is_monotone: bool, optimizer="tree-binning",
                 tree_random_state=None, verbose=False):
        """
        :param x
            array to search optimal edges
        :param y
            target array
        :param n_initial
            initial groups to search optimal edges
        :param n_target
            max amout of target groups
        :param optimizer
            string optimizer name
        """
        self.X = x
        self.y = y
        self.n_initial = n_initial
        self.n_target = n_target
        self.optimizer = optimizer
        self.bin_minimal_size = bin_minimal_size
        self.bin_size_increase = bin_size_increase
        self.is_monotone = is_monotone
        self.total_good = total_good
        self.total_bad = total_bad
        self.tree_random_state = tree_random_state
        self.verbose = verbose
        self.init_edges = None
        self._check_inputs()

    def _print(self, string):
        """
        Sub function of print to print only in self.verbose set to True
        :param string
            string to be printed to stdout
        """
        if self.verbose:
            print(string)
        else:
            pass

    def _check_inputs(self):
        """
        Makes assertion of initialization parameters
        :return: None
        """
        assert len(self.X) == len(self.y)
        assert len(self.X) > self.n_initial
        assert self.n_initial > 0
        assert self.n_target <= self.n_initial
        assert self.n_target > 1
        assert self.total_bad > 0
        assert self.total_good > 0
        if self.tree_random_state is not None:
            assert self.tree_random_state > 0
        assert self.optimizer in LIST_OF_ALGOS
        assert isinstance(self.verbose, bool)

    def optimize(self) -> Tuple[np.ndarray, np.float]:
        """
        Initializes main logic
        :return (optimal_edges,
        """
        return self._search_optimals()

    def _initial_split(self) -> np.ndarray:
        """
        Calculates initial group split
        :return initial edges
        :rtype np.ndarray
        """
        return make_edges(self.X, self.n_initial, print_func=self._print, is_add_infinity=False)

    def _search_optimals(self) -> Tuple[np.ndarray, np.float]:
        """
        Searches optimal edges of given X.
        :return optimal_edges,gini
        :rtype Tuple(np.ndarray,np.float)
        """
        if self.optimizer == "full-search":
            self._print("Doing full-search with init: %s" % self.init_edges)
            self.init_edges = self._initial_split()
            all_edge_variants = generate_combs(self.init_edges[1:-1], self.n_target)
            self._print("FS variants total %i" % len(all_edge_variants))
            mono_variants = []
            for edge_variant in all_edge_variants:
                edge_variant = add_infinity(edge_variant)
                bins = split_by_edges(self.X, edge_variant)
                is_mono, gini = check_variant(bins, self.y, t_good=self.total_good, t_bad=self.total_bad)
                if is_mono:
                    mono_variants.append((edge_variant, gini))
            self._print("Total mono variants: %i" % len(mono_variants))
            optimization_result = sorted(mono_variants, key=lambda x: x[1])[-1]
        elif self.optimizer == "tree-binning":
            self._print("Doing tree-binning")
            monotonicity_flag = False
            bin_count = 1000
            curr_min_bin = self.bin_minimal_size
            while (monotonicity_flag is False) & (bin_count >= 2):
                binner = DecisionTreeClassifier(min_samples_leaf=curr_min_bin, random_state=self.tree_random_state)
                binner.fit(X=self.X.reshape(-1, 1), y=self.y)
                bins = binner.tree_.threshold[binner.tree_.threshold != _tree.TREE_UNDEFINED]
                bin_count = len(bins)
                if (self.is_monotone is False) | (bin_count < 2):
                    if bin_count == 0:
                        binner = DecisionTreeClassifier(max_depth=1, random_state=None)
                        binner.fit(X=self.X.reshape(-1, 1), y=self.y)
                        bins = binner.tree_.threshold[binner.tree_.threshold != _tree.TREE_UNDEFINED]
                    break
                else:
                    bins_probs = pd.DataFrame({'predicted_proba': binner.predict_proba(self.X.reshape(-1, 1))[:, 1],
                                               'feature_value': self.X}).groupby('predicted_proba').max().reset_index()\
                        .sort_values('feature_value')['predicted_proba'].values
                    monotonicity_flag = check_mono(bins_probs)
                    if not monotonicity_flag:
                        curr_min_bin = curr_min_bin + self.bin_size_increase
            bins = add_infinity(sorted(bins))
            optimization_result = (bins, 0)
            self._print(optimization_result)

        else:
            raise NotImplementedError("""Optimization algo %s is not implemented. 
            Current implemented algos are: %s""" % (self.optimizer, LIST_OF_ALGOS))
        self.X = None
        self.y = None

        return optimization_result
