import yaml
import json

from pathlib import Path


def load_file(line):
    if line.endswith(".yaml") or line.endswith(".yml"):
        parser = yaml.safe_load
    else:
        parser = json.load

    return parser(Path(line).read_text())


def parse_file_line(line):
    extract = []
    path = line
    if ":" in line:
        full = line.split(":")
        path = full[0]
        extract = list(
            filter(bool, map(lambda ex: ex.strip(), full[1].split(",")))
        )

    return path, extract


def extract_keys(data, extract):
    ret = {}
    for ex in extract:
        if "." not in ex:
            ret.update(**data.get(ex))
        else:
            ret_key = data.copy()
            for key in ex.split("."):
                ret_key = ret_key[key]
            ret.update(**ret_key)
    return ret


def parse(line):
    path, extract = parse_file_line(line)
    data = load_file(path)
    parsed = extract_keys(data, extract)
    return parsed

