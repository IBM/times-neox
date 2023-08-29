#!/usr/bin/env python

from gluonts.dataset.repository.datasets import get_dataset
import argparse, json, os, re

parser = argparse.ArgumentParser("Preload GluonTS datasets")
parser.add_argument("filename", help="Config file path", type=str)
args = parser.parse_args()
filename = args.filename

if not os.path.exists(filename):
    raise FileNotFoundError(f"File {filename} does not exist.")

with open(filename, "r") as file:
    content = file.read()
    obj = json.loads(re.sub("^\s*\#.*", "", content, flags=re.MULTILINE))


datasets = obj["times_args"]["datasets"]

to_load = []
if "train" in datasets:
    to_load.extend(datasets["train"])
if "validation" in datasets:
    to_load.extend(datasets["validation"])
if "test" in datasets:
    to_load.extend(datasets["test"])

for i in to_load:
    get_dataset(i)