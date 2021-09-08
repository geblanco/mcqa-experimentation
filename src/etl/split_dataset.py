import os
import json
import random
import argparse
import shutil

from pathlib import Path
from functools import reduce
from mcqa_utils import Dataset, get_mask_matching_text
from mcqa_utils.utils import label_to_id, update_example


def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_path", required=True, type=str,
        help="Directory containing the dataset"
    )
    parser.add_argument(
        "-o", "--output_dir", required=False, default=None, type=str,
        help="Directory to write the modified dataset split"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Whether to overwrite final files in output directory"
    )
    parser.add_argument(
        "--outputs", required=True, type=str, nargs=2,
        help="Output file names for splitted dataset (under output_dir)"
    )
    parser.add_argument(
        "--move", required=False, default=None, type=str, nargs=2,
        help="Move a file from data_path to output_dir before writing"
        " anything. Useful to move dev split to other name before writing"
        " new splits"
    )
    parser.add_argument(
        "-t", "--task", required=False, default="generic", type=str,
        help="The task pertaining the dataset (RACE/generic)"
    )
    parser.add_argument(
        "-s", "--split", required=True, type=str,
        choices=["train", "dev", "test"],
        help="The split of the dataset to modify"
    )
    parser.add_argument(
        "--no_answer_text", default=None, type=str, required=False,
        help="Text of an unaswerable question answer"
    )
    parser.add_argument(
        "--proportion", type=float, required=False, default=0.75,
        help="Proportion to of extra optadd extra option."
    )
    return parser.parse_args()


def get_examples_split_by_text(dataset, no_answer_text, split):
    ans_mask = get_mask_matching_text(no_answer_text, match=False)
    unans_mask = get_mask_matching_text(no_answer_text, match=True)
    examples = dataset.get_split(split)
    ans_examples = dataset.reduce_by_mask(examples, ans_mask)
    unans_examples = dataset.reduce_by_mask(examples, unans_mask)
    return ans_examples, unans_examples


def random_indices(examples, num_samples):
    total_examples = len(examples)
    if num_samples > total_examples:
        raise ValueError(
            "Requested more examples than available"
        )
    return random.sample(range(total_examples), k=num_samples)


def gather_indices(examples, indices):
    split_1, split_2 = [], []
    for idx, sample in enumerate(examples):
        if idx in indices:
            split_1.append(sample)
        else:
            split_2.append(sample)
    return split_1, split_2


def calculate_amounts(*args, proportion=0.5):
    ans_examples = args[0]
    unans_examples = None
    if len(args) > 1:
        unans_examples = args[1]

    total_examples = len(ans_examples)
    if unans_examples is not None:
        total_examples += len(unans_examples)

    train_amount = round(total_examples * proportion)

    if unans_examples is None:
        return train_amount

    unans_proportion = len(unans_examples) / total_examples
    unans_train_amount = round(train_amount * unans_proportion)
    ans_train_amount = train_amount - unans_train_amount

    return ans_train_amount, unans_train_amount


def train_test_split(examples, amount):
    indices = random_indices(examples, amount)
    train, dev = gather_indices(examples, indices)
    return train, dev


def randomize_split(split_data):
    indices = random_indices(split_data, len(split_data))
    split, _ = gather_indices(split_data, indices)

    return split


def check_args(args):
    if args.proportion > 1.0 or args.proportion < 0.0:
        raise ValueError(
            f"Invalid proportion of samples!"
        )

    if args.output_dir is None and (not args.overwrite and args.move is None):
        raise ValueError(
            "Data path and output dir are the same, but overwrite = False!"
        )


def setup_dirs(args):
    if args.output_dir is None:
        output_dir = Path(args.data_path)
    else:
        output_dir = Path(args.output_dir)

    outputs = [output_dir.joinpath(out) for out in args.outputs]

    moves, move_from, move_to = None, None, None
    if args.move is not None:
        move_from = output_dir.joinpath(args.move[0])
        move_to = output_dir.joinpath(args.move[1])
        moves = [move_from, move_to]

    for out in outputs:
        if out != move_from and out.exists() and not args.overwrite:
            raise ValueError(
                f"Output {out} file already exists, but overwrite = False!"
            )

    if move_to is not None and move_to.exists() and not args.overwrite:
        raise ValueError(
            f"Move destination {move_to} already exists, but"
            " overwrite = False!"
        )

    return outputs, moves


def save_data(json_data, output_path):
    output_path.parent.mkdir(exist_ok=True, parents=True)
    json_str = json.dumps(json_data, ensure_ascii=False) + "\n"
    with open(output_path, "w") as fout:
        fout.write(json_str + "\n")


def main(args):
    check_args(args)
    outputs, moves = setup_dirs(args)

    dataset = Dataset(data_path=args.data_path, task=args.task)
    if args.no_answer_text is not None:
        # works
        ans_examples, unans_examples = get_examples_split_by_text(
            dataset, args.no_answer_text, args.split
        )
        # works
        ans_amount, unans_amount = calculate_amounts(
            ans_examples, unans_examples, proportion=args.proportion
        )
        # not working
        ans_train, ans_dev = train_test_split(ans_examples, ans_amount)
        unans_train, unans_dev = train_test_split(unans_examples, unans_amount)
        train_split = ans_train + unans_train
        dev_split = ans_dev + unans_dev
    else:
        examples = dataset.get_split(args.split)
        train_amount = calculate_amounts(examples, proportion=args.proportion)
        train_split, dev_split = train_test_split(examples, train_amount)

    train_split = randomize_split(train_split)
    dev_split = randomize_split(dev_split)

    if moves is not None:
        shutil.move(moves[0], moves[1])

    save_data(dataset.to_json(train_split), outputs[0])
    save_data(dataset.to_json(dev_split), outputs[1])


if __name__ == "__main__":
    args = parse_flags()
    main(args)
