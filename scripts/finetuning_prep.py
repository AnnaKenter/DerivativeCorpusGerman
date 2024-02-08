import os
import pandas as pd
import numpy as np
import math

"""
This file splits our data as required for finetuning DagGER-BERT into frequency bins and test/train/dev
In the SHARED condition the context sentences are used to split the data into test/train/dev set
In the SPLIT condition the derivatives are used to split the data into test/train/dev set
"""


def create_shared_split(df, out_name, size_train=0.8, size_test=0.15):
    """
    Return tuple (df_shared, df_split)
    with df_xxx being a dict with {train:, test:, dev: }
    """
    # create shared corpus
    shared_train = df.sample(frac=size_train, random_state=200)
    test_dev = df.drop(shared_train.index)
    size_test2 = size_test/(1-size_train)
    shared_test = test_dev.sample(frac=size_test2, random_state=200)
    shared_dev = test_dev.drop(shared_test.index)
    shared = {"train": shared_train, "test": shared_test, "dev": shared_dev}
    
    # create outpaths for shared corpus
    shared_train_p = os.path.join("..\corpus", f"{out_name}_shared_train.xlsx")
    shared_test_p = os.path.join("..\corpus", f"{out_name}_shared_test.xlsx")
    shared_dev_p = os.path.join("..\corpus", f"{out_name}_shared_dev.xlsx")
    shared_train.to_excel(shared_train_p)
    shared_test.to_excel(shared_test_p)
    shared_dev.to_excel(shared_dev_p)
    

    # create split corpus
    create_split(df, "base", size_train, size_test, out_name)
    create_split(df, "stem", size_train, size_test, out_name)

    return shared, None


def create_split(df, row_name, size_train, size_test, out_name):
    stems = df[row_name].unique()
    np.random.shuffle(stems)
    df = df[df[row_name].notna()]  # dont use entries without stem
    train_sample_size = math.floor(size_train*len(stems))
    test_sample_size = math.floor(size_test*len(stems))
    stems_train = stems[:train_sample_size]
    stems_test = stems[train_sample_size:train_sample_size+test_sample_size]
    stems_dev = stems[train_sample_size+test_sample_size:]
    split_train = df.loc[df[row_name].isin(stems_train)]
    split_test = df.loc[df[row_name].isin(stems_test)]
    split_dev = df.loc[df[row_name].isin(stems_dev)]

    # write split corpus
    split_train_p = os.path.join("..\corpus", f"{out_name}_split_{row_name}_train.xlsx")
    split_test_p = os.path.join("..\corpus", f"{out_name}_split_{row_name}_test.xlsx")
    split_dev_p = os.path.join("..\corpus", f"{out_name}_split_{row_name}_dev.xlsx")
    split_train.to_excel(split_train_p)
    split_test.to_excel(split_test_p)
    split_dev.to_excel(split_dev_p)


def create_bins(df):
    """
    Return list of bins 1-8
    """
    bin_sizes = {1: (1,1), 2: (2,3), 3: (4,7), 4: (8,15), 5: (16,31), 6: (32,64), 7: (64, 127), 8: (128, np.inf)}
    bins = dict()
    for bin_id in range(1, 9):
        start, end = bin_sizes[bin_id]
        bin_df = df.loc[(df["count"] >= start) & (df["count"] <= end)]
        bins[bin_id] = bin_df
    return bins



def split_pfx_sfx_b(df):
    """
    Return tuple with pfx, sfx, b
    """
    pfx = df.loc[df["mode"] == "prefix"]
    sfx = df.loc[df["mode"] == "suffix"]
    b = df.loc[df["mode"] == "both"]
    return pfx, sfx, b


def create_train_test_dev(df, out_name):
    bins = create_bins(df)
    for bin_id, bin in bins.items():
        bin2 = transform_sentence_level(bin, f"{out_name}_{bin_id}")
        create_shared_split(bin2, f"{out_name}_{bin_id}")

def transform_sentence_level(df, outname):
    new_data = list()
    for i, row in df.iterrows():
        context_sents = row["context"][2:-2].replace("', '", "']['").split("']['")
        for context in context_sents:
            sml_row = row.copy()
            sml_row.loc["context"] = context
            new_data.append(sml_row)
    newdf = pd.DataFrame(new_data).reset_index()
    return newdf

def orchestrater(file_path):
    df = pd.read_excel(file_path)
    pfx, sfx, b = split_pfx_sfx_b(df)
    create_train_test_dev(pfx, "pfx")
    create_train_test_dev(sfx, "sfx")
    create_train_test_dev(b, "both")


if __name__ == "__main__":
    orchestrater("../derivative_lists/cleaned_joined.xlsx")  # TODO: change path if required