#!/usr/bin/env python
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import os
import argparse
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from utils.utils import compute_metrics, setup_logger
from utils.data_utils import read_data
from model import builder


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Generate args."""
    parser = argparse.ArgumentParser(description="You should add those parameter")
    parser.add_argument(
        "--task",
        type=str,
        help="Which task to run.",
        default="data_imputation",
        choices=["data_imputation", "data_transformation", "entity_resolution"]
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="The path of dataset.",
        required=True
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        help="Output path.", 
        default="outputs"
    )
    parser.add_argument(
        "--instance_wise",
        help="Set instance-wise component.",
        action="store_true"
    )
    parser.add_argument(
        "--context_num", 
        type=int, 
        help="The number of sample examples for instance-wise retrieval.", 
        default=20,
    )
    parser.add_argument(
        "--instance_num",
        type=int,
        help="The number of instances to retrieve.",
        default=3
    )    
    parser.add_argument(
        "--metadata_wise",
        help="Set metadata-wise component.",
        action="store_true"
    )
    parser.add_argument(
        "--data_parsing",
        help="Set adaptive data parsing module.",
        action="store_true"
    )
    parser.add_argument(
        "--prompt_engineering",
        help="Set prompt engineering module.",
        action="store_true"
    )
    # Model args
    parser.add_argument(
        "--api_key", 
        type=str, 
        help="The OpenAI API uses API keys for authentication.", 
        required=True
    )
    parser.add_argument("--temperature", type=float, help="Temperature.", default=0.0)
    parser.add_argument("--max_tokens", type=int, help="Max tokens to generate.", default=100)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # Set api args
    os.environ["OPENAI_API_KEY"] = args.api_key
    dataset_name = args.data_dir.split('/')[-1]
    setup_logger(os.path.join(args.output_dir, dataset_name))
    logger.info(json.dumps(vars(args), indent=4))

    # set seed
    np.random.seed(args.seed)

    dataset = read_data(
        task=args.task,
        data_dir=args.data_dir,
    )
    train_data = dataset["train"]
    test_data = dataset["test"]
    logger.info(f"Test shape is {test_data.shape[0]}")

    # UniDM
    model = builder.build_model(args, logger)
        
    # Run 
    preds = model.run(train_data, test_data)

    # Metric
    trial_metrics = {"prec": [], "rec": [], "f1": [], "acc": []}
    gt = test_data["label_str"]
    prec, rec, acc, f1 = compute_metrics(preds, gt, args.task)

    logger.info(
        f"Metrics\n"
        f"Prec: {prec:.3f} Recall: {rec:.3f} Acc: {acc:.3f} F1: {f1:.3f}"
    )
    trial_metrics["rec"].append(rec)
    trial_metrics["prec"].append(prec)
    trial_metrics["acc"].append(acc)
    trial_metrics["f1"].append(f1)

    # save result
    output_file = (
            Path(args.output_dir)
            / f"{Path(args.data_dir).stem}"
            / f"k{args.instance_num}"
            / f"trial.feather"
        )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saved to {output_file}")
    save_data = test_data.copy(deep=True).reset_index()
    save_data["preds"] = preds
    save_data["p_as"] = model.p_as
    save_data.to_feather(output_file)

    output_metrics = output_file.parent / "metrics.json"
    json.dump(trial_metrics, open(output_metrics, "w"))
    logger.info(f"Final Metrics {json.dumps(trial_metrics, indent=4)}")
    logger.info(f"Metrics dumped to {output_metrics}")


if __name__ == "__main__":
    main()