import argparse
import json

from tabulate_dict import print_dict

from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType


def parse_args():
   parser = argparse.ArgumentParser()
   parser.add_argument(
      "--translations_dict", default=None, type=str,
      help="Path to a dictionary translating name queries to run ids"
   )
   parser.add_argument(
      "--run_id", default=None, type=str,
      help="Work only with experiments child of the given run"
   )
   parser.add_argument(
      "--metric", default="C_at_1", type=str,
      help="Default metric to filter out runs and sort them"
   )
   parser.add_argument(
      "--topk", default=10000, type=int,
      help="Only print `topk` run results"
   )
   parser.add_argument(
      "-d", "--digits", type=int, required=False, default=4,
      help="Number of digits to round floats in report"
   )
   return parser.parse_args()


def filter_without_metric(runs, metric=None):
   if metric is None:
      metric = "C_at_1"
   return [rr for rr in runs if metric in rr.data.metrics]


def bucket_by_eval_type(runs):
   classifier_bucket = []
   model_bucket = []
   for rr in runs:
      run_name = rr.data.tags["mlflow.runName"]
      parts = run_name.split("_")
      if parts[0] == "classifier":
         classifier_bucket.append(rr)
      else:
         model_bucket.append(rr)

   return classifier_bucket, model_bucket


def remove_model_name(run_name):
   for model in [
      "quail_race_fmt_",
      "quail_no_empty_answers_",
      "race_",
      "race_with_empty_answers_"
   ]:
      run_name = run_name.replace(model, "")

   return run_name


def run_to_dict(run):
   # ret = run.data.metrics.copy()
   run_name = run.data.tags["mlflow.runName"]
   run_name = remove_model_name(run_name)
   parts = run_name.split("_")
   eval_type = parts[0]

   ret = {
      "pipeline": parts[1],
      "features": ",".join(parts[2:]),
   }
   ret.update({key: value for key, value in run.data.metrics.items()})
   return ret


def main(args):   
   client = MlflowClient()
   query = ""
   translations_dict = {}

   if args.translations_dict is not None:
      translations_dict = json.load(open(args.translations_dict, "r"))

   if args.run_id is not None:
      args.run_id = translations_dict.get(args.run_id, args.run_id)
      query = f"tags.`mlflow.parentRunId` = '{args.run_id}'"


   runs = client.search_runs(
      experiment_ids=[exp.experiment_id for exp in client.list_experiments()],
      filter_string=query,
      run_view_type=ViewType.ACTIVE_ONLY,
      max_results=100,
      order_by=[f"metrics.{args.metric} DESC"]
   )
   if len(runs) <= 1:
      print(runs)
      return

   runs = filter_without_metric(runs, metric=args.metric)
   class_runs, model_runs = bucket_by_eval_type(runs)
   class_runs = class_runs[:min(args.topk - 1, len(class_runs))]
   class_dicts = [run_to_dict(run) for run in class_runs]
   for run_dict in class_dicts:
      print(print_dict(run_dict, digits=args.digits))

   print(print_dict(run_to_dict(model_runs[0]), digits=args.digits))


if __name__ == "__main__":
   main(parse_args())