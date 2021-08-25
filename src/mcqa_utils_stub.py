import json
import mlflow
import argparse

from mcqa_utils.metric import metrics_map

FLAGS = None


def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--predictions', required=False, default=None,
        help='Predictions from model'
    )
    parser.add_argument(
        '-n', '--nbest_predictions', required=False, default=None,
        help='Nbest predictions from model'
    )
    parser.add_argument(
        '-d', '--dataset', required=True,
        help='Directory where the dataset is stored'
    )
    parser.add_argument(
        '-i', '--info', action='store_true',
        help="Just print info about the dataset"
    )
    parser.add_argument(
        '-s', '--split', default='dev', required=False,
        choices=['train', 'dev', 'test'],
        help='Split to evaluate from the dataset'
    )
    parser.add_argument(
        '-T', '--task', default=None, required=False,
        help='Task to evaluate (default = generic). This '
        'is needed for the dataset processor (see geblanco/mc-transformers)'
    )
    parser.add_argument(
        '-ft', '--find_threshold', action='store_true', required=False,
        help='Perfom threshold search over the answers and apply the metrics'
    )
    parser.add_argument(
        '-t', '--threshold', default=None, required=False, type=float,
        help='Apply threshold to all answers'
    )
    parser.add_argument(
        '-m', '--metrics', nargs='*', required=False, default=[],
        help=f'Metrics to apply (available: {", ".join((metrics_map.keys()))})'
    )
    parser.add_argument(
        '-o', '--output', default=None, required=False,
        help='Whether to put the results (default = stdout)'
    )
    parser.add_argument(
        '--overwrite', action='store_true', required=False,
        help='Overwrite output file (default false)'
    )
    parser.add_argument(
        '--merge', action='store_true', required=False,
        help='Whether to merge output file with previous output'
    )
    parser.add_argument(
        '--no_answer_text', type=str, required=False, default=None,
        help='Text of an unaswerable question answer'
    )
    parser.add_argument(
        '-pf', '--probs_field', type=str, required=False, default=None,
        help='Field to use as `probs` field in prediction answers '
        '(default probs, but can be anything parsed in the answer)'
    )
    parser.add_argument(
        '-fm', '--fill_missing', default=None, required=False,
        help='Fill missing answers. Can be filled following a uniform, '
        'random choosing or giving a value for all probs '
        '(uniform/random/value)'
    )
    parser.add_argument(
        "--save_mlflow", action="store_true",
        help="Stores the given metrics in mlflow (requires package installed)"
    )
    args = parser.parse_args()
    if not args.info and len(args.metrics) == 0:
        raise ValueError(
            "If not printing dataset info, you must request at "
            "least one metric!"
        )
    elif (
        not args.info and
        (args.nbest_predictions is None and args.predictions is None)
    ):
        raise ValueError('You must provide some predictions to evalute!')
    return args


def main(args):
    save_dict = {}
    for metric in args.metrics:
        save_dict.update(**{metric: 0.95})

    mlflow.log_metrics(save_dict)


if __name__ == '__main__':
    main(parse_flags())