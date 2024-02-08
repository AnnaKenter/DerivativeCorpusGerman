import os
import pandas as pd

def join_excel(dir_path, out_path):
    list_of_df = []
    for file in os.listdir(dir_path):
        df = pd.read_excel(os.path.join(dir_path, file), header=0, engine="openpyxl")
        list_of_df.append(df)

    big_df = pd.concat(list_of_df, axis=0, ignore_index=True)
    big_df = big_df.rename(columns={"Unnamed: 0": "token"})

    grouped_df = big_df.groupby(by=["token", "affix", "base", "in_lexica", "stem", "mode"], as_index=False, dropna=False).agg({"count": "sum", "context": "sum"})

    tokens = grouped_df["token"]
    duplicates = grouped_df[tokens.isin(tokens[tokens.duplicated()])].sort_values("token")

    grouped_df.to_excel(out_path)

    sample = grouped_df.sample(100)
    print(sample)



if __name__ == "__main__":
    print("Starting the join")
    join_excel("../derivative_lists/excel", "../derivative_lists/all_derivates.xlsx")
    print("Ended the join")
