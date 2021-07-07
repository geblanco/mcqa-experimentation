#!/usr/bin/env python

import json
import yaml
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file", type=str,
        help="File to extract arguments from (accepts yaml and json)",
    )
    parser.add_argument(
        "-x", "--exclude", help="Fields to exclude", nargs="*",
        required=False, type=str, default=None
    )
    parser.add_argument(
        "-j", "--json", help="Print as json", action="store_true",
    )
    return parser.parse_known_args()


def args_to_array_args(prog_args):
    array_args = []
    for arg_key, arg_value in prog_args.items():
        array_args.append("--" + arg_key)
        if type(arg_value) is bool:
            continue
        # should be stringified?
        array_args.append(str(arg_value))
    return array_args


def parse_args_line(line):
    extract = []
    path = line
    if ":" in line:
        full = line.split(":")
        path = full[0]
        extract = list(
            filter(bool, map(lambda ex: ex.strip(), full[1].split(",")))
        )

    return path, extract


def load_args_file(args_line):
    if args_line.endswith(".yaml") or args_line.endswith(".yml"):
        parser = yaml.safe_load
    else:
        parser = json.load

    return parser(Path(args_line).read_text())


def extract_keys(prog_args, extract):
    ret = {}
    for ex in extract:
        if "." not in ex:
            ret.update(**prog_args.get(ex))
        else:
            ret_key = prog_args.copy()
            for key in ex.split("."):
                ret_key = ret_key[key]
            ret.update(**ret_key)
    return ret


def main(args):
    args_path, extract = parse_args_line(args.file)
    prog_args = load_args_file(args_path)
    if extract is not None:
        prog_args = extract_keys(prog_args, extract)
    if args.exclude is not None:
        for exclude in args.exclude:
            del prog_args[exclude]
    if args.json:
        print(json.dumps(prog_args))
    else:
        array_args = args_to_array_args(prog_args)
        print(" ".join(array_args), end=" ")


if __name__ == "__main__":
    args, unknown = parse_args()
    main(args)
    print(" ".join(unknown))
