# -*- coding: utf-8 -*-
# main module logic.

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from .functions import calc_descriptive_from_vector, split_by_edges, gini_index, calculate_loc_woe
from .optimizer import WingOptimizer, LIST_OF_ALGOS
from typing import Tuple, Dict
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier


class WingOfEvidence(BaseEstimator, TransformerMixin):
    """
    This class creates realization of WoE for one variable
    """
    def __init__(self, variable_name=None, vector_type="c", bin_minimal_size=.05, is_monotone=False,
                 bin_size_increase=.05, optimizer="ilya-binning", spec_values=None, n_initial=10, n_target=5,
                 tree_random_state=None, verbose=False):
        """
        One feature wing.
        :param variable_name
            Optional, name of variable to analyze
        :param vector_type
            Required, str, ('c' or 'd') - type of vector to analyze
        :param n_initial
            Required only for full-search, number of initial groups to split X-vector.
        :param n_target
            Required only for full-search, number of maximal amount of target groups
        :param spec_values
            Optional, dict in form {value:"GROUP_NAME"} to generate special groups to analyze.
        :param optimizer
            Optinal, string with name of optimizer.
            List of availiable optimizers can be called via from wing.core.optimizer.LIST_OF_ALGOS
        :param bin_minimal_size
            Optional, minimal bin size (float) to be selected via initial splitting and optimal edges selection.
        :param verbose
            Optional, turns on verbose output
        """
        self.variable_name = variable_name
        self.vector_type = vector_type
        self.n_initial = n_initial
        self.n_target = n_target
        self.spec_values = spec_values
        self.optimizer = optimizer
        self.bin_minimal_size = bin_minimal_size
        self.bin_size_increase = bin_size_increase
        self.is_monotone = is_monotone
        self.tree_random_state = tree_random_state
        self.verbose = verbose
        self.__TOTAL_GOOD = None
        self.__TOTAL_BAD = None
        self._check_params()

    def _print(self, string: str):
        """
        Handler function for producing output,based on True/False logic of value self.verbose
        :param string:
            string to print
        :return: None
        """
        if self.verbose:
            print(string)
        else:
            pass

    def __missing_handler(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Hanldes missing values in input data frame
        :param df
            Data frame with columns ["X","y"]
        :return clear_df,miss_woe
        :rtype Tuple[pd.DataFrame,dict]
        """
        loc_df = df.copy()
        missing_mask = loc_df["X"].isnull()
        miss_df = loc_df[missing_mask]
        non_miss_df = loc_df[~missing_mask]
        dict_skeleton = {
            "good": None,
            "bad": None,
            "total": None,
            "woe": None,
            "local_event_rate": None}
        if len(miss_df) == 0:
            pass
        else:
            vect = pd.Series([
                miss_df["y"].sum(),
                len(miss_df) - miss_df["y"].sum()],
                index=["good", "bad"])
            miss_woe = calculate_loc_woe(vect, self.__TOTAL_GOOD, self.__TOTAL_BAD)
            dict_skeleton['good'] = vect['good']
            dict_skeleton['bad'] = vect['bad']
            dict_skeleton['total'] = vect["good"] + vect["bad"]
            dict_skeleton['woe'] = miss_woe
            dict_skeleton['local_event_rate'] = vect["good"] / (vect["good"] + vect["bad"])
        return non_miss_df, dict_skeleton

    def __special_handler(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Handles special values in data frame
        :param df
            pd.DataFrame with ['X','y'] values
        :return clear_df,miss_woe
        :rtype Tuple[pd.DataFrame,dict]
        """
        loc_df = df.copy()
        special_mask = loc_df["X"].isin(list(self.spec_values.keys()))
        special_df = loc_df[special_mask]
        non_special_df = loc_df[~special_mask]
        spec_values_woe = {}
        for key in self.spec_values:
            dict_skeleton = {
                "good": None,
                "bad": None,
                "total": None,
                "woe": None,
                "local_event_rate": None}
            key_df = special_df[special_df["X"] == key]
            if len(key_df) == 0:
                spec_values_woe[key] = dict_skeleton
            else:
                vect = pd.Series([
                    key_df["y"].sum(),
                    len(key_df) - key_df["y"].sum()],
                    index=["good", "bad"])
                key_woe = calculate_loc_woe(vect, self.__TOTAL_GOOD, self.__TOTAL_BAD)
                dict_skeleton['good'] = vect['good']
                dict_skeleton['bad'] = vect['bad']
                dict_skeleton['total'] = vect["good"] + vect["bad"]
                dict_skeleton['woe'] = key_woe
                dict_skeleton['local_event_rate'] = vect["good"] / (vect["good"] + vect["bad"])
                spec_values_woe[key] = dict_skeleton
        return non_special_df, spec_values_woe

    def fit(self, X, y):
        """
        Fits the model
        :param X
            Types of input:
                pd.DataFrame with 1 column
                pd.Series
                np.ndarray with shape (X,1) or (X,)
                list
                tuple
        :param y
            Types of input:
                pd.DataFrame with 1 column
                pd.Series
                np.ndarray with shape (X,1) or (X,)
                list
                tuple
        """
        self._print("Started fit")
        # Проверяем входной вектор на содержимое
        df = self._check_data(X, y)
        self._print("Data check success,df size: %s" % df.shape[0])
        # рассчитываем Total good/bad
        # Эти переменные глобальные потому, что они нужны для
        # WoE функции
        self.__TOTAL_GOOD = df["y"].sum()
        self.__TOTAL_BAD = len(df["y"]) - self.__TOTAL_GOOD
        self._print("Got G: %i B: %i" % (self.__TOTAL_GOOD, self.__TOTAL_BAD))
        # Теперь выделяем пропущенные значения.
        df, miss_woe = self.__missing_handler(df)
        self.miss_woe = miss_woe
        self._print("Missing values done, miss woe is %s,df size: %s" % (self.miss_woe, df.shape[0]))
        if len(df) == 0:
            # Весь оставшийся набор данных пустой
            # Значит в датасете были только миссинги, fit закончен
            return self
        # Теперь проверяем спецзначения.
        df, spec_values_woe = self.__special_handler(df)
        self.spec_values_woe = spec_values_woe
        self._print("Spec values done, spec woe is %s,df size: %s" % (self.spec_values_woe, df.shape[0]))
        if len(df) == 0:
            # после отбора миссингов и спецзначений датасет пуст
            return self
        #  Проверяем количество уникальных записей в чистом DF
        if df["X"].dtype == np.dtype("O"):
            self._print("""X-vector dtype is object, vector_type will be converted to 'd'""")
            self.vector_type = "d"
        else:
            self._print("X-vector type is numeric")
        d_cnt = len(df["X"].unique())
        self._print("D-values in  clear X: %i" % d_cnt)
        if (self.vector_type == 'c') and (d_cnt <= self.n_initial):
            self._print("Converting data type to discrete because of low uniques")
            self.vector_type = "d"
        else:
            pass
        self._print("Current vector type: %s" % self.vector_type)
        if self.vector_type == "c":
            #######################################################
            #  тут рассчитываем для непрерывной переменной
            #######################################################
            X, y = df["X"].values, df["y"].values
            self._print("Starting optimizer search")
            self.optimizer_obj = WingOptimizer(x=X, y=y,
                                           total_good=self.__TOTAL_GOOD, total_bad=self.__TOTAL_BAD,
                                           n_initial=self.n_initial, n_target=self.n_target,
                                           optimizer=self.optimizer, verbose=self.verbose,
                                           bin_size_increase=self.bin_size_increase,
                                           bin_minimal_size=self.bin_minimal_size, is_monotone=self.is_monotone,
                                           tree_random_state=self.tree_random_state)
            self.optimal_edges, best_gini = self.optimizer_obj.optimize()
            self._print("Optimal edges found: %s" % self.optimal_edges)
            self._print("With gini: %0.4f" % best_gini)
            bins = split_by_edges(X,self.optimal_edges)
            self.cont_df_woe = calc_descriptive_from_vector(bins, y, self.__TOTAL_GOOD, self.__TOTAL_BAD)
            self.optimal_edges_dict = self._generate_edge_dict(self.optimal_edges)
            self.wing_id_dict = self.cont_df_woe["woe"].to_dict()
        elif self.vector_type == "d":
            #######################################################
            #  тут рассчитываем для дискретной переменной
            #######################################################
            self._print("Inside vector type 'd'")

            self.discrete_label_enc = LabelEncoder()
            self.discrete_onehot_enc = OneHotEncoder(sparse=False, handle_unknown='error')
            transformed_X = self.discrete_onehot_enc.fit_transform(
                self.discrete_label_enc.fit_transform(X).reshape(-1, 1))
            self.discrete_binner = DecisionTreeClassifier(min_samples_leaf=self.bin_minimal_size,
                                                          random_state=self.tree_random_state)
            self.discrete_binner.fit(X=transformed_X, y=y)
            bins = self.discrete_binner.predict_proba(X=transformed_X)[:, 1]

            self.discrete_df_woe = calc_descriptive_from_vector(bins, y, self.__TOTAL_GOOD, self.__TOTAL_BAD)
            self.wing_id_dict = self.discrete_df_woe["woe"].to_dict()
            self._print("Discrete woe df created")
            # self.categories = self.discrete_df_woe["woe"].to_dict()
        return self

    def transform(self, X, y=None):
        if y is None:
            # bugfix for compatability
            y = pd.Series([1 for i in range(len(X))])
        df = self._check_data(X, y)
        # fill miss
        miss_df = df[pd.isnull(df["X"])].copy()
        miss_df["woe_group"] = "AUTO_MISS"
        miss_df["woe"] = self.miss_woe["woe"]
        #######################################################
        # TODO: Расписать что тут происходит
        #######################################################
        spec_df = df[df["X"].isin(self.spec_values)].copy()
        spec_df["woe_group"] = spec_df["X"].apply(lambda x: self.spec_values.get(x))
        spec_df["woe"] = spec_df["X"].apply(lambda x: self.spec_values_woe.get(x).get("woe"))

        # fill dat
        flt_conc = (~pd.isnull(df["X"]) & (~df["X"].isin(self.spec_values)))
        clear_df = df[flt_conc].copy()
        if self.vector_type == "c":
            #######################################################
            # быстрый фикс ошибки в том случае, когда opt
            # не рассчитан
            #######################################################
            if hasattr(self,"optimal_edges"):
                clear_df["woe_group"] = split_by_edges(clear_df["X"], self.optimal_edges)
                clear_df["woe"] = clear_df["woe_group"].apply(lambda x: self.wing_id_dict[x])
            else:
                clear_df["woe_group"] = "NO_GROUP"
                clear_df["woe"] = None
        else:
            if hasattr(self, "discrete_df_woe"):
                transformed_X = self.discrete_onehot_enc.transform(
                    self.discrete_label_enc.transform(clear_df["X"]).reshape(-1, 1))
                clear_df["woe_group"] = self.discrete_binner.predict_proba(X=transformed_X)[:, 1]
                clear_df["woe"] = clear_df["woe_group"].apply(lambda x: self.wing_id_dict[x])
            else:
                clear_df["woe_group"] = "NO_GROUP"
                clear_df["woe"] = None
        miss_df["woe_group"] = miss_df["woe_group"].astype(str)
        spec_df["woe_group"] = spec_df["woe_group"].astype(str)
        clear_df["woe_group"] = clear_df["woe_group"].astype(str)
        full_transform = pd.concat([miss_df, spec_df, clear_df], axis=0)  # ["woe"]
        #######################################################
        # TODO: Расписать что тут происходит + алго выбора
        #######################################################
        miss_wing_selector = [self.miss_woe["woe"]]
        spec_wing_selector = [sub_d.get("woe") for sub_d in self.spec_values_woe.values()]
        if self.vector_type == "c":
            if hasattr(self, "wing_id_dict"):
                grpd_wing_selector = list(self.wing_id_dict.values())
            else:
                grpd_wing_selector = [None]
        else:
            grpd_wing_selector = list(self.discrete_df_woe["woe"].values)
        allv_wing_selector = miss_wing_selector+spec_wing_selector+grpd_wing_selector
        allv_wing_selector_flt = [v for v in allv_wing_selector if v is not None]
        max_woe_replacer = np.min(allv_wing_selector_flt)
        full_transform["woe"] = full_transform["woe"].fillna(max_woe_replacer)
        full_transform = full_transform.sort_index()
        return full_transform

    def get_wing_agg(self, only_clear=True):
        """
        Shows result of WoE fitting as table bins,woe,iv
        Returns:
            woe_df (pd.DataFrame): data frame with WoE fitter parameters
        """
        if only_clear:
            if self.vector_type == "c":
                cont_df_woe_loc = self.cont_df_woe.copy()
                cont_df_woe_loc.index = [self.optimal_edges_dict[v] for v in cont_df_woe_loc.index]
                return cont_df_woe_loc
            else:
                return self.discrete_df_woe
        if self.miss_woe:
            miss_wect = pd.DataFrame(self.miss_woe).T
        else:
            miss_wect = pd.DataFrame(columns=["good", "bad", "woe", "total", "local_event_rate"])
        if self.spec_values_woe:
            spec_v_df = pd.DataFrame(self.spec_values_woe).T
        else:
            spec_v_df = pd.DataFrame(columns=["good", "bad", "woe", "total", "local_event_rate"])
        if self.vector_type == "c":
            miss_wect = miss_wect[['good', 'bad', 'total', 'woe', 'local_event_rate']]
            spec_v_df = spec_v_df[['good', 'bad', 'total', 'woe', 'local_event_rate']]
            cont_df_woe_loc = self.cont_df_woe.copy()
            cont_df_woe_loc.index = [self.optimal_edges_dict[v] for v in cont_df_woe_loc.index]
            full_agg = pd.concat([miss_wect, spec_v_df, cont_df_woe_loc], axis=0)
        else:
            miss_wect = miss_wect[['good', 'bad', 'total', 'woe', 'local_event_rate']]
            spec_v_df = spec_v_df[['good', 'bad', 'total', 'woe', 'local_event_rate']]
            full_agg = pd.concat([miss_wect, spec_v_df, self.discrete_df_woe], axis=0)
        return full_agg

    def get_global_gini(self):
        woe_df = self.get_wing_agg()
        woe_df = woe_df.sort_values(by="local_event_rate", ascending=False)
        gini_index_value = gini_index(woe_df["good"].values, woe_df["bad"].values)
        return gini_index_value

    def _check_params(self):
        """
        This method checks parameters in __init__, raises error in case of errors
        """
        if self.vector_type not in ("c", "d"):
            raise ValueError("Bad vector_type, should be one of ('c','d')")
        if self.n_initial < self.n_target:
            raise ValueError("Number of target groups higher than pre-binning groups")
        if self.n_initial < 2:
            raise ValueError("Number of initial groups too low to create accurate bins")
        if self.n_target < 1:
            raise ValueError("Set more target groups to search optimal parameters")
        if self.optimizer not in LIST_OF_ALGOS:
            raise ValueError('Selected optimizer %s not in list of optimizers %s' % (self.optimizer, LIST_OF_ALGOS))

    def _check_data(self, X, y):
        """
        Should raise some error if any test is not OK, else do nothing
        Args:
            X (numpy.ndarray): numpy array of X
            y (numpy.ndarray): numpy array of y
        Returns:
            None if everything is ok, else raises error
        """
        if (X.size != y.size):
            raise ValueError("y-size ( %i ) doesn't match X size ( %i )" % (y.size, X.size))
        try:
            X = np.array(X).ravel()
            y = np.array(y).ravel()
        except:
            raise ValueError("X or Y vector cannot by transformed to np.array")
        common_df = pd.DataFrame(np.array([X, y]).T, columns=["X", "y"])
        return common_df

    def _generate_edge_dict(self, edges):
        edges_dict = {}
        for idx, (low, high) in enumerate(zip(edges, edges[1:])):
            edges_dict[idx + 1] = (low, high)
        return edges_dict

    def plot_woe(self):
        """
        Creates woe plot for data in woe_df
        """
        import matplotlib.pyplot as plt
        woe_df = self.get_woe()
        f, ax = plt.subplots(1, 1)
        p1 = woe_df.plot(kind="bar", x=woe_df.index, y="good", ax=ax, sharex=True, figsize=(20, 10), edgecolor="k")
        p2 = woe_df.plot(kind="line", x=woe_df.index, y="woe", ax=ax, secondary_y=True, style="o-", lw=3, c="r", ms=10)
        return (p1, p2)


class WingsOfEvidence(BaseEstimator, TransformerMixin):
    def __init__(self, bin_minimal_size=.05, is_monotone=False,
                 bin_size_increase=.05, columns_to_apply="all",
                 n_initial=10, n_target=5, mass_spec_values={},
                 optimizer="ilya-binning", only_values=True,
                 tree_random_state=None, verbose=False):
        """
        Этот класс реализует расчет WoE по многим переменным
        :Args:
            columns_to_apply (list or "all"): список колонок, к которым нужно применить преобразование.
                Если задано all, рассчитает WoE по всем.
            n_initial (int): Число групп для инициирующего разбиения
            n_target (int): Число групп для целевого разбиения
            mass_spec_values (dict): Словарь спецзначений для колонок, пример: {"col1":{0:"ZERO"}}
            optimizer (str): Вид используемого оптимизатора. Доступны:
                "full-search" - полный перебор всех вариантов
                "adaptive" - адаптивный поиск
            only_values (bool): Возвращать только числовые значения?
                Если True, то вернет только WOE_x (где x - колонки из columns_to_apply)

        """
        self.columns_to_apply = columns_to_apply
        self.mass_spec_values = mass_spec_values
        self.n_initial = n_initial
        self.n_target = n_target
        self.optimizer = optimizer
        self.only_values = only_values
        self.verbose = verbose
        self.bin_minimal_size = bin_minimal_size
        self.is_monotone = is_monotone
        self.bin_size_increase = bin_size_increase
        self.tree_random_state = tree_random_state

    def _print(self, string: str):
        """
        Handler function for producing output,based on True/False logic of value self.verbose
        :param string:
            string to print
        :return: None
        """
        if self.verbose:
            print(string)
        else:
            pass

    def fit(self, X, y):
        """
        This class fits onefeature woe for each column in columns_to_apply
        Args:
            X (pd.DataFrame): pandas dataframe with X values
            y (pd.Series): pandas series with target value
        Returns:
            self
        """
        if self.columns_to_apply == "all":
            self.columns_to_apply = X.columns
        self.fitted_wing = {}
        self.gini_dict = {}
        self.error_columns = []
        for column in self.columns_to_apply:
            self._print("==="*20)
            self._print("Working with variable: %s"%column)
            X_vec = X[column]
            column_dict = self.mass_spec_values.get(column)
            if not column_dict:
                spec_values = {}
            else:
                spec_values = column_dict
            wing = WingOfEvidence(variable_name=column, vector_type="c", n_initial=self.n_initial,
                                  n_target=self.n_target, spec_values=spec_values,
                                  optimizer=self.optimizer, verbose=self.verbose,
                                  bin_minimal_size=self.bin_minimal_size, is_monotone=self.is_monotone,
                                  bin_size_increase=self.bin_size_increase, tree_random_state=self.tree_random_state)
            try:
                wing.fit(X_vec, y)
                self.fitted_wing[column] = wing
                self.gini_dict[column] = wing.get_global_gini()
            except Exception as e:
                self.error_columns.append(column)
                self._print("Got error: %s" % e)

        return self

    def transform(self, X, y=None):
        result_frame = pd.DataFrame()
        for column in self.columns_to_apply:
            if column not in self.error_columns:
                woe_transformer = self.fitted_wing[column]
                woe_values = woe_transformer.transform(X[column])["woe"]
                result_frame["WOE_%s" % column] = woe_values
                if not self.only_values:
                    result_frame["WOE_g_%s" % column] = woe_transformer.transform(X[column])["woe_group"]
        return result_frame

    def get_gini_vector(self):
        gini_series = pd.Series(self.gini_dict)
        return gini_series


if __name__ == "__main__":
    pass
