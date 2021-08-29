#!/usr/bin/env python

import json
import argparse

from params_utils import parse


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


def main(args):
    prog_args = parse(args.file)
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
