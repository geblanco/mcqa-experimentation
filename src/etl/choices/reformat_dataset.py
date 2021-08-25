import os
import json
import random
import argparse

from pathlib import Path
from mcqa_utils import Dataset, get_mask_matching_text
from mcqa_utils.utils import label_to_id, update_example


def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_path", required=True, type=str,
        help="Directory containing the dataset"
    )
    parser.add_argument(
        "-o", "--output_dir", required=True, type=str,
        help="Directory to write the modified dataset split"
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
        "--no_answer_text", type=str, required=True,
        help="Text of an unaswerable question answer"
    )
    parser.add_argument(
        "--index_list_path", type=str, required=False,
        help="Where to save the index list of removed options"
    )
    parser.add_argument(
        "--keep_matching_text", action="store_true",
        help="Whether to keep the examples with the answer matching the given"
        " text (default is to remove those examples)"
    )
    parser.add_argument(
        "--mask", action="store_true",
        help="Mask any --proportion (default 0.25) of samples with the text"
        " given in --no_answer_text, optionally --mask_correct"
        " (default is random masking)"
    )
    parser.add_argument(
        "--mask_correct", action="store_true",
        help="Whether to mask the correct option on each sample, or just"
        " a random option."
    )
    parser.add_argument(
        "--proportion", type=float, required=False, default=0.25,
        help="Proportion to of extra optadd extra option."
    )
    return parser.parse_args()


def get_index_matching_text(sample, answer_text):
    match_text = answer_text.lower()
    matching_idx = 0
    while (
        matching_idx < len(sample.endings) and  # noqa: W504
        sample.endings[matching_idx].lower().find(match_text) == -1
    ):
        matching_idx += 1
    return matching_idx


def remove_extra_option(examples, answer_text, fix_label=True):
    ret_examples = []
    remove_list = {}
    # -1 for removal, -1 for 0'ed index
    max_label_id = len(examples[0].endings) - 2
    for sample in examples:
        matching_idx = get_index_matching_text(sample, answer_text)
        remove_list[sample.example_id] = matching_idx
        if not fix_label:
            del sample.endings[matching_idx]
            ex = sample
            if sample.label is not None:
                label = min(label_to_id(sample.label), max_label_id)
                ex = update_example(sample, label=label)
            ret_examples.append(ex)
        else:
            ans_index = label_to_id(sample.label)
            if matching_idx < len(sample.endings):
                if ans_index == matching_idx:
                    print(sample)
                    raise ValueError(
                        "Something went really wrong, the answer to delete "
                        "is the correct one, did you masked the wrong way?"
                    )
                del sample.endings[matching_idx]

            if ans_index > matching_idx:
                ans_index -= 1
                ex = update_example(sample, label=ans_index)
                ret_examples.append(ex)
            else:
                ret_examples.append(sample)

    return ret_examples, remove_list


def dump_examples(dataset, examples, output_dir, split):
    json_data = dataset.to_json(examples)
    json_str = json.dumps(json_data, ensure_ascii=False) + "\n"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_dir, f"{split}.json"), "w") as fout:
        fout.write(json_str)


def dump_index_list(index_list_path, index_list):
    json_index_list = json.dumps(index_list) + "\n"
    Path(index_list_path).parent.mkdir(parents=True, exist_ok=True)
    with open(index_list_path, "w") as fout:
        fout.write(json_index_list)


def random_indices(examples, num_samples):
    total_examples = len(examples)
    if num_samples > total_examples:
        raise ValueError(
            "Requested more examples than available"
        )
    return random.choices(range(total_examples), k=num_samples + 1)


def apply_mask(examples, mask_indices, mask_correct, mask_text):
    for idx in mask_indices:
        sample = examples[idx]
        if mask_correct:
            ending_idx = label_to_id(sample.label)
        else:
            ending_idx = random.choice(list(range(len(sample.endings))))
        endings = sample.todict()["endings"]
        endings[ending_idx] = mask_text
        examples[idx] = update_example(sample, endings=endings)
    return examples


def main(
    data_path,
    output_dir,
    task,
    split,
    no_answer_text,
    keep_matching_text,
    index_list_path,
    mask,
    mask_correct,
    proportion
):
    dataset = Dataset(data_path=data_path, task=task)
    answerable_mask = get_mask_matching_text(no_answer_text, match=False)
    examples = dataset.get_split(split)
    if not keep_matching_text:
        answerable_examples = dataset.reduce_by_mask(
            examples, answerable_mask
        )
        stats_str = f"({len(answerable_examples)}/{len(examples)}: "
        stats_str += "{:.2f}%)".format(
            (len(answerable_examples) / len(examples)) * 100
        )
        print(f"{stats_str} of examples are valid")
        examples = answerable_examples

    mod_examples, index_list = remove_extra_option(
        examples, no_answer_text, fix_label=(not keep_matching_text)
    )
    dump_examples(dataset, mod_examples, output_dir, split)
    if index_list_path is not None:
        # convert index list ids to standard ids
        index_list = {
            "-".join(dataset.processor._decode_id(id)): value
            for id, value in index_list.items()
        }
        dump_index_list(index_list_path, index_list)


def mask_dataset(
    data_path,
    output_dir,
    task,
    split,
    no_answer_text,
    keep_matching_text,
    index_list_path,
    mask,
    mask_correct,
    proportion
):
    if proportion > 1.0:
        raise ValueError(
            f"Unable to mask higher than {proportion} of samples!"
        )

    dataset = Dataset(data_path=data_path, task=task)
    examples = dataset.get_split(split, with_text_values=True)
    num_samples = round(len(examples) * proportion)
    mask_indices = random_indices(examples, num_samples)
    masked_dataset = apply_mask(
        examples, mask_indices,
        mask_correct=mask_correct, mask_text=no_answer_text
    )
    dump_examples(dataset, masked_dataset, output_dir, split)


if __name__ == "__main__":
    args = parse_flags()
    if args.mask:
        mask_dataset(**vars(args))
    else:
        main(**vars(args))
