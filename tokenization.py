from itertools import chain
from functools import reduce
import multiprocessing
import argparse
from typing import List
from datasets import concatenate_datasets, load_dataset, load_from_disk, Features, Sequence, Value
from transformers import AutoTokenizer

IGNORE_INDEX = -100

def main(args):
    if args.dataset is None or len(args.dataset[0]) == 0:
        raise RuntimeError("No datasets provided")
    datasets = args.dataset[0]

    splits = [x.split(",")[1] if len(x.split(",")) == 2 else "" for x in datasets]
    datasets = [x.split(",")[0] for x in datasets]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    if args.json:
        dataset = load_dataset("json", data_files=datasets)[args.split]
        if reduce(lambda x,y: x or len(y) > 0, splits, False):
            if len(datasets) > 1:
                raise RuntimeError("Can only use splitting on json datasets if there is exactly one input file")
            dataset = dataset.train_test_split(train_size=float(splits[0]), seed=args.seed)["train"]
    else:
        to_concatenate = []
        for i in range(0, len(datasets)):
            try:
                loaded = load_from_disk(datasets[i])
            except:
                loaded = load_dataset(datasets[i])[args.split]
            if len(splits[i]) > 0:
                loaded = loaded.train_test_split(train_size=float(splits[i]), seed=args.seed)["train"]
            to_concatenate.append(loaded)
        dataset = concatenate_datasets(to_concatenate)

    dataset = dataset.remove_columns([x for x in dataset.column_names if x not in [args.feature]])

    if args.limit:
        dataset = dataset.select(range(args.limit))

    def build(example):
        input_ids = []
        attention_mask = []
        labels = []

        for part in example[args.feature]:
            if part["from"] == "input" or part["from"] == "human":
                t = tokenizer("### Instruction:\n" + part["value"] + "\n", add_special_tokens=False)
                input_ids.extend(t.input_ids)
                attention_mask.extend(t.attention_mask)
                if args.train_on_input:
                    labels.extend(t.input_ids)
                else:
                    labels.extend([IGNORE_INDEX] * len(t.input_ids))
            elif part["from"] == "output" or part["from"] == "gpt":
                t = tokenizer("### Response:\n" + part["value"] + "\n", add_special_tokens=False)
                input_ids.extend(t.input_ids)
                attention_mask.extend(t.attention_mask)
                labels.extend(t.input_ids)

        input_ids.append(tokenizer.eos_token_id)
        attention_mask.append(1)
        labels.append(tokenizer.eos_token_id)

        if args.pad_to_length:
            to_pad = args.pad_to_length - len(input_ids)
            if to_pad < 0:
                if args.truncate:
                    input_ids = input_ids[0:args.pad_to_length]
                    attention_mask = attention_mask[0:args.pad_to_length]
                    labels = labels[0:args.pad_to_length]
                else:
                    raise RuntimeError(f"pad_to_length {args.pad_to_length} too short, sample is of length {len(input_ids)}")
            elif to_pad > 0:
                input_ids.extend([tokenizer.pad_token_id] * to_pad)
                attention_mask.extend([0] * to_pad)
                labels.extend([IGNORE_INDEX] * to_pad)

            assert len(input_ids) == len(attention_mask)
            assert len(attention_mask) == len(labels)
            assert len(labels) == args.pad_to_length

        ret = { "input_ids": input_ids, "attention_mask": attention_mask, "labels": labels }
        return ret

    tokenized_dataset = dataset.map(
        build,
        num_proc=args.num_proc,
        remove_columns=[args.feature],
        features=Features({"input_ids": Sequence(Value("int32")), "attention_mask": Sequence(Value("int8")), "labels": Sequence(Value("int32"))})
    )

    if args.pad_to_length and args.truncate and args.count_truncated:
        num_truncated = len(tokenized_dataset.filter(lambda x: x["input_ids"][-1] != tokenizer.pad_token_id))
        print(f"Truncated {num_truncated} samples")

    train_dataset = tokenized_dataset

    if args.output:
        train_dataset.save_to_disk(args.output)
    if args.push_to_hub:
        train_dataset.push_to_hub(args.push_to_hub, private=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", nargs="+")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--feature", type=str, default="conversations")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output", type=str)
    parser.add_argument("--push-to-hub", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pad-to-length", type=int)
    parser.add_argument("--train-on-input", action="store_true")
    parser.add_argument("--truncate", action="store_true")
    parser.add_argument("--count-truncated", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--num-proc", type=int,
                        default=multiprocessing.cpu_count())
    main(parser.parse_args())
