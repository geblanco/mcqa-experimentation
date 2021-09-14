# import os
# os.chdir(os.path.abspath("../.."))

import yaml
import argparse
import tempfile
from hyperp_utils import combination_to_params

import mlflow.projects
from mlflow.tracking.client import MlflowClient


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier_dir", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--params_file", type=str)
    parser.add_argument("--classification_train_embeddings_path", type=str)
    parser.add_argument("--classification_eval_embeddings_path", type=str)
    parser.add_argument("--classification_metrics_dir", type=str)
    parser.add_argument("--corrections_embeddings_path", type=str)
    parser.add_argument("--corrections_output_dir", type=str)
    parser.add_argument("--corrections_split", type=str)
    parser.add_argument("--corrections_strategy", type=str)
    parser.add_argument("--corrections_no_answer_text", type=str)
    parser.add_argument("--evaluation_nbest_predictions", type=str)
    parser.add_argument("--evaluation_output", type=str)
    parser.add_argument("--evaluation_task", type=str)
    args, unk = parser.parse_known_args()
    return args


def run_classification_step(tracking_client, experiment_id, params):
    # ToDo := Correctly log metrics
    with mlflow.start_run(nested=True) as child_run:
        # run the classification Step and wait it finishes
        p = mlflow.projects.run(
            uri=".",
            entry_point="classification",
            run_id=child_run.info.run_id,
            parameters={
                "train_embeddings_path": params["classification_train_embeddings_path"],
                "eval_embeddings_path": params["classification_eval_embeddings_path"],
                "classifier_dir": params["classifier_dir"],
                "metrics_dir": params["classification_metrics_dir"],
                "params_file": params["params_file"]
            },
            experiment_id=experiment_id,
            use_conda=False,
            synchronous=False
        )
        succeeded = p.wait()
        return tracking_client.get_run(p.run_id) if succeeded else None


def run_correction_step(tracking_client, experiment_id, params):
    step_params = {
        "classifier_dir": params["classifier_dir"],
        "embeddings_path": params["corrections_embeddings_path"],
        "output_dir": params["corrections_output_dir"],
        "dataset": params["dataset"],
        "params_file": params["params_file"],
        "split": params["corrections_split"],
        "strategy": params["corrections_strategy"],
    }
    if params["corrections_strategy"] == "no_answer":
        step_params.update(no_answer_text=params["corrections_no_answer_text"])

    # ToDo := Correctly log metrics
    with mlflow.start_run(nested=True) as child_run:
        # run the correction Step and wait it finishes
        p = mlflow.projects.run(
            uri=".",
            entry_point="classifier_corrects_model",
            run_id=child_run.info.run_id,
            parameters=step_params,
            experiment_id=experiment_id,
            use_conda=False,
            synchronous=False
        )
        succeeded = p.wait()
        return tracking_client.get_run(p.run_id) if succeeded else None


def run_class_evaluation_step(tracking_client, experiment_id, params):
    # ToDo := Correctly log metrics
    with mlflow.start_run(nested=True) as child_run:
        # run the evaluation Step and wait it finishes
        p = mlflow.projects.run(
            uri=".",
            entry_point="evaluate_corrections",
            run_id=child_run.info.run_id,
            parameters={
                "dataset": params["dataset"],
                "nbest_predictions": params["evaluation_nbest_predictions"],
                "output": params["evaluation_output"],
                "task": params["evaluation_task"],
            },
            experiment_id=experiment_id,
            use_conda=False,
            synchronous=False
        )
        succeeded = p.wait()
        return tracking_client.get_run(p.run_id) if succeeded else None


def run_model_evaluation_step(tracking_client, experiment_id, params):
    # ToDo := Correctly log metrics
    with mlflow.start_run(nested=True) as child_run:
        # run the evaluation Step and wait it finishes
        p = mlflow.projects.run(
            uri=".",
            entry_point="evaluate_model",
            run_id=child_run.info.run_id,
            parameters={
                "dataset": params["dataset"],
                "nbest_predictions": params["evaluation_model_nbest_predictions"],  # noqa: E501
                "output": params["evaluation_model_output"],
                "task": params["evaluation_task"],
            },
            experiment_id=experiment_id,
            use_conda=False,
            synchronous=False
        )
        succeeded = p.wait()
        return tracking_client.get_run(p.run_id) if succeeded else None


def grid(params):
    if "hyper-search" in params and "grid" in params["hyper-search"]:
        for combination in params["hyper-search"]["grid"]:
            yield combination_to_params(combination)


def main(args):
    steps_params = vars(args)
    params = yaml.safe_load(open(args.params_file, "r"))
    param_set_file = tempfile.mktemp()
    steps_params.update(params_file=param_set_file)
    steps = [
        run_classification_step,
        run_correction_step,
        run_class_evaluation_step,
        run_model_evaluation_step,
    ]

    tracking_client = mlflow.tracking.MlflowClient()
    with mlflow.start_run() as run:
        experiment_id = run.info.experiment_id
        for param_set in grid(params):
            yaml.safe_dump(param_set, open(param_set_file, "w"))
            for step in steps:
                step_result = step(tracking_client, experiment_id, steps_params)
                if step_result is not None:
                    mlflow.log_metrics(step_result.data.metrics)
                else:
                    print(f"Failed to reproduce step {step}!")
                    break
    # for combination in feature sweep:
    # - mlflow classification step
    # - mlflow correction step
    # - mlflow evaluation step
    # - log metrics, params and artifacts on each step


if __name__ == "__main__":
    main(parse_args())
