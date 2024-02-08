import os
import re
import pandas as pd

def calulate_productivity(path_joined_derivatives, mode):
    """
    Calculates the productivity for each affix
    :param path_joined_derivatives: path to excel file including all derivatives
    :param mode: prefix or suffix
    :return: dict including all affixes and their variety, frequency and hapax count
    :rtype: dict
    """
    df = pd.read_excel(path_joined_derivatives)
    df = df.loc[df["mode"] == mode]

    affix_l = df["affix"].unique()
    affix_prod = dict()  # affix: 0.03

    for affix in affix_l:
        d1 = df[df["affix"] == affix]
        d2 = df[df["affix"] == affix]
        hapaxes = len(d1[d1["count"] == 1].index)
        v_types = len(d2.index)
        n_numtok = d2["count"].sum()
        affix_prod[affix] = (v_types, n_numtok, hapaxes)

    with open(f"../data/affix_productivity_{mode}.txt", "a+", encoding="utf-8") as f:
        f.write(f"affix,v,n,abshapax\n")
        for affix, values in affix_prod.items():
            v, n, h = values
            f.write(f"{affix},{v},{n},{h}\n")

    return affix_prod


if __name__ == "__main__":
    path_to_all_joined_derivatives = "../derivative_lists/cleaned_joined.xlsx"  # TODO: change to your path
    print(calulate_productivity(path_to_all_joined_derivatives, "prefix"))
    print(calulate_productivity(path_to_all_joined_derivatives, "suffix"))