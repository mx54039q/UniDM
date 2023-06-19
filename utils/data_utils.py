#!/usr/bin/env python
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
import logging, os
from functools import partial
from pathlib import Path
from typing import Dict, List
import pandas as pd

from . import constants

logger = logging.getLogger(__name__)


def sample_train_data(train: pd.DataFrame, n_rows: int):
    res = train.sample(n_rows)
    return res

def serialize_row(
    row: pd.core.series.Series,
    column_map: Dict[str, str],
    sep_tok: str = ".",
    nan_tok: str = "nan",
) -> str:
    """Turn structured row into string."""
    res = []
    for c_og, c_map in column_map.items():
        if str(row[c_og]) == "nan":
            row[c_og] = nan_tok
        else:
            row[c_og] = f"{str(row[c_og]).strip()}"
        res.append(f"{c_map}: {row[c_og]}".lstrip())
    if len(sep_tok) > 0 and sep_tok != ".":
        sep_tok = f" {sep_tok}"
    return f"{sep_tok} ".join(res)

def serialize_match_pair(
    row: pd.core.series.Series,
    column_mapA: Dict[str, str],
    column_mapB: Dict[str, str],
    sep_tok: str,
    nan_tok: str,
    prod_name='Item',
) -> str:
    """Turn structured pair of entities into string for matching."""
    res = (
        f"{prod_name} A is {serialize_row(row, column_mapA, sep_tok, nan_tok)}."
        f" {prod_name} B is {serialize_row(row, column_mapB, sep_tok, nan_tok)}."
    )
    return res

def serialize_imputation(
    row: pd.core.series.Series,
    column_map: Dict[str, str],
    impute_col: str,
    sep_tok: str = ".",
    nan_tok: str = "nan",
) -> str:
    """Turn single entity into string for imputation."""
    assert impute_col not in column_map, f"{impute_col} cannot be in column map"
    # Rename to avoid passing white spaced sep token to serialize_row
    sep_tok_ws = sep_tok
    if len(sep_tok) > 0 and sep_tok != ".":
        sep_tok_ws = f" {sep_tok}"
    res = f"{serialize_row(row, column_map, sep_tok, nan_tok)}{sep_tok_ws} "
    return res

def serialize_error_detection_spelling(
    row: pd.core.series.Series,
    sep_tok: str,
    nan_tok: str,
) -> str:
    """Turn single cell into string for error detection."""
    column_map = {row["col_name"]: row["col_name"]}
    res = f"Is there a x spelling error in {serialize_row(row, column_map, sep_tok, nan_tok)} "
    return res

def serialize_error_detection(
    row: pd.core.series.Series,
    sep_tok: str,
    nan_tok: str,
) -> str:
    """Turn single cell into string for error detection."""
    column_map = {
        c: c
        for c in row.index
        if str(c) not in ["Unnamed: 0", "text", "col_name", "label_str", "is_clean"]
    }
    entire_row = serialize_row(row, column_map, sep_tok, nan_tok)
    column_map = {row["col_name"]: row["col_name"]}
    res = f"{entire_row}\n\nIs there an error in {serialize_row(row, column_map, sep_tok, nan_tok)}"
    return res


def read_blocked_pairs(
    split_path: str,
    tableA: pd.DataFrame,
    tableB: pd.DataFrame,
    cols_to_drop: List[str],
    col_renaming: Dict[str, str],
    sep_tok: str,
    nan_tok: str,
) -> pd.DataFrame:
    """Read in pre-blocked pairs with T/F match labels."""
    for c in cols_to_drop:
        tableA = tableA.drop(c, axis=1, inplace=False)
        tableB = tableB.drop(c, axis=1, inplace=False)
    if len(col_renaming) > 0:
        tableA = tableA.rename(columns=col_renaming, inplace=False)
        tableB = tableB.rename(columns=col_renaming, inplace=False)

    column_mapA = {f"{c}_A": c for c in tableA.columns if c != "id"}
    column_mapB = {f"{c}_B": c for c in tableB.columns if c != "id"}

    labels = pd.read_csv(split_path)

    mergedA = pd.merge(labels, tableA, right_on="id", left_on="ltable_id")
    merged = pd.merge(
        mergedA,
        tableB,
        right_on="id",
        left_on="rtable_id",
        suffixes=("_A", "_B"),
    )

    merged["serialized_A"] = merged.apply(
        lambda row: serialize_row(
            row,
            column_mapA,
            sep_tok,
            nan_tok,
        ),
        axis=1,
    )
    merged["serialized_B"] = merged.apply(
        lambda row: serialize_row(
            row,
            column_mapB,
            sep_tok,
            nan_tok,
        ),
        axis=1,
    )
    merged["label_str"] = merged.apply(
        lambda row: "Yes\n" if row["label"] == 1 else "No\n", axis=1
    )
    return merged


def read_imputation_single(
    split_path: str,
    impute_col: str,
    cols_to_drop: List[str],
    col_renaming: Dict[str, str],
    sep_tok: str,
    nan_tok: str,
) -> pd.DataFrame:
    """Read in table and create label impute col."""
    table = pd.read_csv(split_path)
    for c in cols_to_drop:
        table = table.drop(c, axis=1, inplace=False)
    if len(col_renaming) > 0:
        table = table.rename(columns=col_renaming, inplace=False)

    table["label_str"] = table[impute_col].apply(lambda x: f"{x}\n")
    return table


def read_transformation(
    split_path: str,
    dataset_name: str,
    # cols_to_drop: List[str],
    # col_renaming: Dict[str, str],
    sep_tok: str,
    nan_tok: str,
) -> pd.DataFrame:
    # table = pd.read_csv(split_path)
    # for c in cols_to_drop:
    #     table = table.drop(c, axis=1, inplace=False)
    # if len(col_renaming) > 0:
    #     table = table.rename(columns=col_renaming, inplace=False)
    
    table = []

    for data in split_path:
        if (dataset_name == "benchmark-stackoverflow" and data.endswith('.txt')) or \
           (dataset_name == "benchmark-bing-query-logs" and data.endswith('.txt') and "semantic" in data):
            file = pd.read_csv(data, sep="\t\t",  encoding='cp1252', 
                names=["data before tansformation", "data after tansformation"], skiprows=1, engine='python')
            with open(data, 'r') as f:
                instruction = f.readlines()[0].strip("\n")
        else:
            continue

        column_map = {c: c for c in file.columns}
        train, test = file[:3], file[3:]

        context = train.apply(
            lambda row: serialize_row(
                row,
                column_map,
                sep_tok,
            ),
            axis=1,
        )
        context = "\n\n".join(list(context))
        
        for _,row in test.iterrows():
            table.append([instruction,context,str(row[0]),str(row[1])])

    table = pd.DataFrame(table, columns=['instruction','context','input','label_str'])
    return table


def read_raw_data(
    task: str,
    data_dir: str,
    sep_tok: str = ".",
    nan_tok: str = "nan",
):
    """Read in data where each directory is unique for a task."""
    dataset_name = data_dir.split('/')[-1]
    data_files_sep = {"test": {}, "train": {}, "validation": {}}
    logger.info(f"Processing {dataset_name}")

    cols_to_drop = constants.DATA2DROPCOLS[dataset_name]
    col_renaming = constants.DATA2COLREMAP[dataset_name]
    data_dir_p = Path(data_dir)
    if task == "entity_resolution":
        train_file = data_dir_p / "train.csv"
        valid_file = data_dir_p / "valid.csv"
        test_file = data_dir_p / "test.csv"
        tableA_file = data_dir_p / "tableA.csv"
        tableB_file = data_dir_p / "tableB.csv"

        tableA = pd.read_csv(tableA_file)
        tableB = pd.read_csv(tableB_file)

        label_col = "label"
        read_data_func = partial(
            read_blocked_pairs,
            tableA=tableA,
            tableB=tableB,
            cols_to_drop=cols_to_drop,
            col_renaming=col_renaming,            
            sep_tok=sep_tok,
            nan_tok=nan_tok,
        )
    elif task == "data_imputation":
        train_file = data_dir_p / "train.csv"
        valid_file = data_dir_p / "valid.csv"
        test_file = data_dir_p / "test.csv"
        label_col = constants.IMPUTE_COLS[dataset_name]
        read_data_func = partial(
            read_imputation_single,
            impute_col=label_col,
            cols_to_drop=cols_to_drop,
            col_renaming=col_renaming,
            sep_tok=sep_tok,
            nan_tok=nan_tok,
        )
    elif task == "data_transformation":
        files = os.listdir(data_dir_p)
        train_file = test_file = [os.path.join(data_dir_p, f) for f in files]
        valid_file = data_dir_p / "valid.csv"
        label_col = 'label_str'
        read_data_func = partial(
            read_transformation,
            dataset_name=dataset_name,
            sep_tok='\n', # sep_tok,
            nan_tok=nan_tok,
        )
    elif task == "error_detection":
        train_file = data_dir_p / "train.csv"
        valid_file = data_dir_p / "valid.csv"
        test_file = data_dir_p / "test.csv"
        table_file = data_dir_p / "table.csv"

        table = pd.read_csv(table_file)
        label_col = "is_clean"
        read_data_func = partial(
            read_error_detection_single,
            table=table,
            cols_to_drop=cols_to_drop,
            col_renaming=col_renaming,
            sep_tok=sep_tok,
            nan_tok=nan_tok,
            spelling=False,
        )
    else:
        raise ValueError(f"Task {task} not recognized.")

    data_files_sep["train"] = read_data_func(train_file)
    data_files_sep["test"] = read_data_func(test_file)
    # Read validation
    if valid_file.exists():
        data_files_sep["validation"] = read_data_func(valid_file)
    return data_files_sep, label_col


def read_data(
    task: str,
    data_dir: str,
    class_balanced: bool = False,
    sep_tok: str = ".",
    nan_tok: str = "nan",
):
    """Read in data where each directory is unique for a task."""
    data_files_sep, label_col = read_raw_data(
        task=task,
        data_dir=data_dir,
        sep_tok=sep_tok,
        nan_tok=nan_tok,
    )

    # Shuffle train data
    data_files_sep["train"] = (
        data_files_sep["train"].sample(frac=1, random_state=42).reset_index(drop=True)
    )

    return data_files_sep
