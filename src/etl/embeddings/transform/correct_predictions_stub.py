import os
import sys
import json
import mlflow
import random
import argparse
import numpy as np

from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from mcqa_utils import Dataset as McqaDataset
from mcqa_utils.utils import label_to_id, id_to_label
from mc_transformers.mc_transformers import softmax

base_path = os.path.dirname(os.path.dirname(__file__))
src_root = os.path.dirname(os.path.dirname(base_path))

sys.path.append(os.path.join(src_root, "processing"))
sys.path.append(os.path.join(src_root, "etl"))
sys.path.append(os.path.join(base_path, "classify"))
sys.path.append(os.path.join(base_path, "extract"))

from utils import load_classifier  # noqa: E402
from hyperp_utils import load_params  # noqa: E402
from dataset_class import Dataset  # noqa: E402
from classification import get_features_from_object  # noqa: E402
from classification import get_data_path_from_features  # noqa: E402
from choices.reformat_predictions import get_index_matching_text  # noqa: E402


class Args():
    def __init__(self, features, data_path=None):
        self.features = features
        self.data_path = data_path


def save_predictions(output_dir, **kwargs):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for fname, content in kwargs.items():
        fpath = str(output_path.joinpath(f"{fname}.json"))
        with open(fpath, "w") as fout:
            fout.write(json.dumps(content) + "\n")


def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--classifier_path", required=True, default=None, type=str,
        help="Path to the classifier"
    )
    parser.add_argument(
        "-e", "--embeddings_path", required=True, default=None, type=str,
        help="Path to embeddings dataset directory (not the dataset itself)"
    )
    parser.add_argument(
        "-o", "--output_dir", required=False, type=str,
        help="Dir to store corrected predictions"
    )
    parser.add_argument(
        "--strategy", required=False, default="no_answer", type=str,
        choices=["no_answer", "random", "longest", "empty_answer"],
        help="Strategy to apply over answers where the model is wrong "
        "(by the classifier criteria)"
    )
    parser.add_argument(
        "--no_answer_text", required=False, default=None, type=str,
        help="Text of an unaswerable question answer (only necessary "
        "for no_answer strategy)"
    )
    parser.add_argument(
        "-d", "--data_path", required=True, default=None, type=str,
        help="Path to original dataset to extract gold answers"
    )
    parser.add_argument(
        "-s", "--split", required=False, default="dev", type=str,
        choices=["train", "dev", "test"],
        help="The split of the dataset to extract embeddings from"
    )
    parser.add_argument(
        "-p", "--params_file", required=False, default="params.yaml", type=str,
        help="Path to params file to get features from (default to root"
        "params.yaml)"
    )
    parser.add_argument(
        "-t", "--task", default="generic", required=False,
        help="Task to evaluate (default = generic). This "
        "is needed for the dataset processor (see geblanco/mc-transformers)"
    )

    args = parser.parse_args()
    if args.strategy == "no_answer" and args.no_answer_text is None:
        raise ValueError(
            "When `no_answer` strategy is selected, you must provide "
            "no_answer_text in order to find the correct answer!"
        )
    return args


def get_path_from_features(classifier_path, data_path, params_file):
    params = load_params(params_file)
    features = get_features_from_object(params, allow_all_feats=True)
    embeddings_path = get_data_path_from_features(
        args=Args(features=features, data_path=data_path)
    )
    # no oversampling in dev set
    if "oversample_embeddings/" in embeddings_path:
        embeddings_path = embeddings_path.replace(
            "oversample_embeddings/", ""
        )
    elif "oversample_" in embeddings_path:
        embeddings_path = embeddings_path.replace("oversample_", "")

    return (
        embeddings_path,
        get_features_from_object(params, allow_all_feats=False)
    )


def main(
    classifier_path,
    embeddings_path,
    output_dir,
    strategy,
    no_answer_text,
    data_path,
    split,
    params_file,
    task,
):
    print(f"Load gold answers from {data_path}")
    print(f"Load classifier from {classifier_path}")
    embeddings_path, features = get_path_from_features(
        classifier_path, embeddings_path, params_file
    )
    print(f"Load embeddings from {embeddings_path}")
    print(f"Features {features}")

    prefix = "eval" if split == "dev" else split
    save_dict = {
        f"{prefix}_nbest_predictions": data_path,
    }

    save_predictions(output_dir, **save_dict)
    mlflow.log_artifact(output_dir)


if __name__ == "__main__":
    args = parse_flags()
    main(**vars(args))
