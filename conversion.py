import sys
import json
import yaml


def convert_data(file):
    """
    converts data from the original tacred files to a jsonl (dict each line) format
    """
    with open(file, "r") as rfile:
        data = json.load(rfile)
    with open(file, "w") as wfile:
        for line in data:
            wfile.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    config = yaml.safe_load(open(sys.argv[1]))
    for file in ["train.json", "dev.json", "test.json"]:
        convert_data(config["data_path"] + file)
        print(f"converted {file} to huggingface dataset format")
