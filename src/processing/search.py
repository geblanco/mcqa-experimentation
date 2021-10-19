import os
import yaml
import json
import argparse
import tempfile
from hyperp_utils import combination_to_params

import mlflow
import mlflow.projects


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
    parser.add_argument("--evaluation_split", type=str)
    parser.add_argument("--evaluation_output", type=str)
    parser.add_argument("--evaluation_task", type=str)
    parser.add_argument("--evaluation_model_nbest_predictions", type=str)
    parser.add_argument("--evaluation_model_output", type=str)
    parser.add_argument("--evaluation_utility_function", type=str)
    args, unk = parser.parse_known_args()
    return args


def update_classifier_metrics(params):
    model_metrics = json.load(open(params["evaluation_model_output"]))
    classifier_metrics = json.load(open(params["evaluation_output"]))
    model_correct = model_metrics["avg_correct"]
    model_incorrect = model_metrics["avg_incorrect"]
    class_correct = classifier_metrics["C_at_1_correct"]
    class_incorrect = classifier_metrics["C_at_1_incorrect"]
    classifier_metrics.update({
        "C_at_1_unanswered_correct": model_correct - class_correct,
        "C_at_1_unanswered_incorrect": model_incorrect - class_incorrect
    })
    return classifier_metrics


def run_classification_step(tracking_client, experiment_id, params, run_name):
    train_embeddings_path = params["classification_train_embeddings_path"]
    eval_embeddings_path = params["classification_eval_embeddings_path"]
    # ToDo := Correctly log metrics
    with mlflow.start_run(nested=True, run_name=run_name) as child_run:
        # run the classification Step and wait it finishes
        p = mlflow.projects.run(
            uri=".",
            entry_point="classification",
            run_id=child_run.info.run_id,
            parameters={
                "train_embeddings_path": train_embeddings_path,
                "eval_embeddings_path": eval_embeddings_path,
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


def run_correction_step(tracking_client, experiment_id, params, run_name):
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
    else:
        step_params.update(no_answer_text=None)

    # ToDo := Correctly log metrics
    with mlflow.start_run(nested=True, run_name=run_name) as child_run:
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


def run_model_eval_step(tracking_client, experiment_id, params, run_name):
    run_name = f"model_{run_name}"
    # ToDo := Correctly log metrics
    with mlflow.start_run(nested=True, run_name=run_name) as child_run:
        # run the evaluation Step and wait it finishes
        p = mlflow.projects.run(
            uri=".",
            entry_point="evaluate_model",
            run_id=child_run.info.run_id,
            parameters={
                "dataset": params["dataset"],
                "nbest_predictions": params["evaluation_model_nbest_predictions"],  # noqa: E501
                "split": params["evaluation_split"],
                "output": params["evaluation_model_output"],
                "utility_function": params["evaluation_utility_function"],
                "task": params["evaluation_task"],
            },
            experiment_id=experiment_id,
            use_conda=False,
            synchronous=False
        )
        succeeded = p.wait()
        return tracking_client.get_run(p.run_id) if succeeded else None


# ToDo := merge evaluations to get unans_{correct,incorrect}
def run_class_eval_step(tracking_client, experiment_id, params, run_name):
    run_name = f"classifier_{run_name}"
    # ToDo := Correctly log metrics
    with mlflow.start_run(nested=True, run_name=run_name) as child_run:
        # run the evaluation Step and wait it finishes
        p = mlflow.projects.run(
            uri=".",
            entry_point="evaluate_corrections",
            run_id=child_run.info.run_id,
            parameters={
                "dataset": params["dataset"],
                "nbest_predictions": params["evaluation_nbest_predictions"],
                "split": params["evaluation_split"],
                "output": params["evaluation_output"],
                "utility_function": params["evaluation_utility_function"],
                "task": params["evaluation_task"],
            },
            experiment_id=experiment_id,
            use_conda=False,
            synchronous=False
        )
        succeeded = p.wait()
        # update model eval with classifier eval
        metrics = update_classifier_metrics(params):
        mlflow.log_metrics(metrics)
        return tracking_client.get_run(p.run_id) if succeeded else None


def grid(params):
    if "hyper-search" in params and "grid" in params["hyper-search"]:
        for combination in params["hyper-search"]["grid"]:
            yield combination_to_params(combination)


def name_from_param_set(param_set):
    name = param_set["classification"]["pipeline"]
    name += "_" + "_".join(param_set["features"])
    return name


def main(args):
    steps_params = vars(args)
    params = yaml.safe_load(open(args.params_file, "r"))
    param_set_file = tempfile.mktemp()
    steps_params.update(params_file=param_set_file)
    steps = [
        run_classification_step,
        run_correction_step,
        run_model_eval_step,
        run_class_eval_step,
    ]

    experiment_name = os.path.basename(args.dataset)
    client = mlflow.tracking.MlflowClient()
    with mlflow.start_run(run_name=experiment_name) as run:
        for param_set in grid(params):
            run_name = experiment_name + "_" + name_from_param_set(param_set)
            exp_id = run.info.experiment_id
            yaml.safe_dump(param_set, open(param_set_file, "w"))
            for step in steps:
                step_result = step(client, exp_id, steps_params, run_name)
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
