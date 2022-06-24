import os

import json
import glob
import argparse

import pandas as pd

from pathlib import Path
from tqdm import tqdm


def read_notebook(glob_path):
    return (
        pd.read_json(glob_path, dtype={"cell_type": "category", "source": "str"})
        .assign(id=glob_path.stem)
        .rename_axis("cell_id")
    )


def merge_notebooks(notebooks_list):
    return (
        pd.concat(notebooks_list)
        .set_index("id", append=True)
        .swaplevel()
        .sort_index(level="id", sort_remaining=False)
    )


def load_example(data_path, id, is_train=True):
    """
    Helper for loading json file of a training example
    """
    with open(os.path.join(data_path, f"{id}.json")) as f:
        example = json.load(f)
    return example


def get_example_df(train_path, example_id, train, ancestors):
    """
    Creates a pandas dataframe of the json cells and correct order.
    """
    cell_order = train.query("id == @example_id")["cell_order"].values[0]
    example_df = pd.DataFrame(load_example(train_path, example_id))
    example_df["id"] = example_id
    my_orders = {}
    for idx, c in enumerate(cell_order.split(" ")):
        my_orders[c] = idx
    example_df["order"] = example_df.index.map(my_orders)
    example_df.reset_index().rename(columns={"index": "cell"})

    example_df["ancestor_id"] = ancestors.query("id == @example_id")[
        "ancestor_id"
    ].values[0]
    example_df["parent_id"] = ancestors.query("id == @example_id")["parent_id"].values[
        0
    ]
    example_df = example_df.reset_index().rename(columns={"index": "cell"})
    example_df = example_df.sort_values("order").reset_index(drop=True)
    example_df["id"] = example_id
    col_order = [
        "id",
        "cell",
        "cell_type",
        "source",
        "order",
        "ancestor_id",
        "parent_id",
    ]
    example_df = example_df[col_order]
    return example_df


def merge_train(json_dir, orders_path, ancestors_path):
    train = pd.read_csv(orders_path)
    ancestors = pd.read_csv(ancestors_path)

    all_ids = train["id"].unique()
    data_gen = ((ids, train, ancestors) for ids in all_ids)

    results = []
    for arg in tqdm(
        data_gen, desc="Merge data from .csv and .json", total=len(all_ids)
    ):
        results.append(get_example_df(json_dir, *arg).reset_index(drop=True))
    data = pd.concat(results).reset_index(drop=True)
    return data


def merge_test(json_dir):
    paths = glob.glob(json_dir + "/" + "*.json")
    notebooks = [read_notebook(path) for path in tqdm(paths, desc="read notebooks")]
    return merge_notebooks(notebooks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_orders", type=str)
    parser.add_argument("--train_ancestors", type=str)
    parser.add_argument("--mode", type=str)
    parser.add_argument("--data", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    # Get the list of json files
    notebooks_jsons = os.listdir(args.data)
    print(f"There are {len(notebooks_jsons)} notebooks json files")

    if args.mode == "train":
        all_results = merge_train(args.data, args.train_orders, args.train_ancestors)
    elif args.mode == "test":
        all_results = merge_test(args.data)

    all_results.to_parquet(args.output)


if __name__ == "__main__":
    main()
