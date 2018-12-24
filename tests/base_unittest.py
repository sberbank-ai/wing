# -*- coding: utf-8 -*-
import unittest
import pandas as pd
from wing.core import WingsOfEvidence


class BaseTest(unittest.TestCase):
    """
    Класс для тестов 
    """
    def test_columns(self):
        """
        Test to filter column list
        """
        cols = list("ABCD")
        wings = WingsOfEvidence(columns_to_apply=cols)
        self.assertEqual(cols, wings.columns_to_apply)

    def testTitanicCSpec(self):
        """
        Test if spec value not in train
        """
        train_df = pd.read_csv("../datasets/titanic/train.csv", sep=",")
        colnames = ["Age"]
        X_train = train_df[colnames]
        y_train = train_df["Survived"]
        wings = WingsOfEvidence(n_initial=12, n_target=5,
                                columns_to_apply=colnames,
                                optimizer="full-search",
                                mass_spec_values={"Age": {0: "ZERO"}}
                                )
        wings.fit(X_train, y_train)
        loc_w = wings.fitted_wing["Age"]
        result = ["%0.5f" % el for el in loc_w.transform(pd.Series([0, 0, 0]))["woe"].values]
        check_w = loc_w.get_wing_agg()["woe"].min()
        tester = ["%0.5f" % check_w for i in range(3)]
        self.assertEqual(result, tester)

    def checkMono(self):
        train_df = pd.read_csv("../datasets/titanic/train.csv", sep=",")
        colnames = ["Age"]
        X_train = train_df[colnames]
        y_train = train_df["Survived"]
        wings = WingsOfEvidence(n_initial=10, n_target=5,
                                columns_to_apply=colnames,
                                optimizer="full-search",
                                mass_spec_values={"Age": {0: "ZERO"}}
                                )
        wings.fit(X_train, y_train)
        loc_w = wings.fitted_wing["Age"]
        does_local = loc_w._check_mono(loc_w.get_wing_agg()["woe"])
        self.assertEqual(does_local, True)

    def checkLowGroup(self):
        train_df = pd.read_csv("../datasets/titanic/train.csv",sep=",")
        colnames = ["Age"]
        X_train = train_df[colnames]
        y_train = train_df["Survived"]
        wings = WingsOfEvidence(n_initial=3, n_target=2,
                                columns_to_apply=colnames,
                                optimizer="full-search",
                                mass_spec_values={"Age": {0: "ZERO"}}
                                )
        wings.fit(X_train, y_train)
        loc_w = wings.fitted_wing["Age"]
        does_local = loc_w._check_mono(loc_w.get_wing_agg()["woe"])
        self.assertEqual(does_local, True)


if __name__ == "__main__":
    unittest.main()
