import copy
import os
import shutil
import sys
from datetime import datetime

import numpy as np
import yaml
import torch
from datasets import ClassLabel, load_dataset
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AdamW, AutoModel, AutoTokenizer, get_scheduler

from data import convert_token, get_labels


def tokenize_batch(batch, tokenizer):
    batch_tokens = []
    for batch_idx, x in enumerate(batch["token"]):
        tokens = []
        for token_idx, token in enumerate(x):
            if token_idx == batch["subj_start"][batch_idx]:
                tokens.append("<e1>")
            elif token_idx == batch["obj_start"][batch_idx]:
                tokens.append("<e2>")
            tokens.append(convert_token(token))
            if token_idx == batch["subj_end"][batch_idx]:
                tokens.append("</e1>")
            elif token_idx == batch["obj_end"][batch_idx]:
                tokens.append("</e2>")
        batch_tokens.append(tokens)
    return tokenizer(
        batch_tokens,
        is_split_into_words=True,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )


def get_untokenized_dataset(data_file):
    dataset = load_dataset("json", data_files=data_file)["train"]
    unused_columns = list(
        set(dataset.column_names)
        & {
            "id",
            "docid",
            "stanford_pos",
            "stanford_ner",
            "stanford_head",
            "stanford_deprel",
        }
    )
    return dataset.remove_columns(unused_columns)


def class2int(labels, config_label):
    return lambda x: {"label": labels.str2int(x[config_label])}


def collate_func(batch):
    return {key: [dic[key] for dic in batch] for key in batch[0]}


class FullModel(nn.Module):
    def __init__(self, language_model, out_features, dropout=0.1):
        super(FullModel, self).__init__()
        self.language_model = language_model
        self.in_features = language_model.config.hidden_size
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(2 * self.in_features, out_features)

    def forward(self, counters, **kwargs):
        x = self.language_model(**kwargs).last_hidden_state
        counters = counters.expand((counters.shape[0], 2, self.in_features))
        x = torch.gather(x, 1, counters)
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.dropout(x)
        return self.linear(x)


def calc_start_counters(input_ids, tokenizer_len, device):
    # tokenizer_len - 4 is index of the fourth last token, i.e. <e1>
    # tokenizer_len - 2 is index of the fourth last token, i.e. <e2>
    counters = torch.zeros((input_ids.shape[0], 2), dtype=torch.long, device=device)

    c0 = (input_ids == tokenizer_len - 4).nonzero(as_tuple=True)
    for row_idx, ent_idx in zip(*c0):
        counters[row_idx, 0] = ent_idx

    c1 = (input_ids == tokenizer_len - 2).nonzero(as_tuple=True)
    for row_idx, ent_idx in zip(*c1):
        counters[row_idx, 1] = ent_idx

    return counters.unsqueeze(2)


def baseline_run(config, logs_folder):
    batch_size = config["batch_size"]
    label_names = get_labels(config["labels_file"])
    class_labels = ClassLabel(names=label_names)

    train_dataset = get_untokenized_dataset(config["train_file"])
    dev_dataset = get_untokenized_dataset(config["dev_file"])
    eval_dataset = get_untokenized_dataset(config["eval_file"])

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_func
    )
    dev_dataloader = DataLoader(
        dev_dataset, batch_size=batch_size, collate_fn=collate_func
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=batch_size, collate_fn=collate_func
    )

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<e1>", "</e1>", "<e2>", "</e2>"]}
    )

    model = AutoModel.from_pretrained(config["model"])
    model.resize_token_embeddings(len(tokenizer))
    full_model = FullModel(model, len(label_names))
    optimizer = AdamW(full_model.parameters(), lr=config["lr"])
    loss_func = nn.CrossEntropyLoss()

    num_epochs = config["num_epochs"]
    train_steps = num_epochs * len(train_dataloader) / config["batch_accumulations"]
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=train_steps / 10,
        num_training_steps=train_steps,
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    full_model.to(device)
    best_f1 = -1.0
    best_epoch = 0

    for epoch in range(num_epochs):
        full_model.train()

        progress_bar = tqdm(range(len(train_dataloader)))
        loss_sum = 0
        for idx, batch in enumerate(train_dataloader):
            batch_encoding = tokenize_batch(batch, tokenizer)
            batch_encoding = {k: v.to(device) for k, v in batch_encoding.items()}

            batch_labels = torch.tensor(
                [class_labels.str2int(relation) for relation in batch["relation"]],
                device=device,
            )
            start_counters = calc_start_counters(
                batch_encoding["input_ids"], len(tokenizer), device
            )
            output = full_model(start_counters, **batch_encoding)

            loss = loss_func(output, batch_labels)
            loss.backward()
            loss_sum += loss.item()
            progress_bar.update(1)

            if (idx + 1) % config["batch_accumulations"] == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

        print(f"\nloss during epoch: {loss_sum / len(train_dataloader)}\n")

        label_list, pred_list = evaluate_model(
            full_model, dev_dataloader, class_labels, tokenizer, device
        )

        f1_float = f1_score(
            np.concatenate(label_list),
            np.concatenate(pred_list),
            average="micro",
            labels=np.arange(1, len(label_names)),
        )

        if f1_float > best_f1:
            best_f1 = f1_float
            best_epoch = epoch + 1
            best_model = copy.deepcopy(full_model)

    label_list, pred_list = evaluate_model(
        best_model, eval_dataloader, class_labels, tokenizer, device
    )

    concat_labels = np.concatenate(label_list)
    concat_preds = np.concatenate(pred_list)

    f1_float = f1_score(
        concat_labels,
        concat_preds,
        average="micro",
        labels=np.arange(1, len(label_names)),
    )

    creport = classification_report(
        concat_labels,
        concat_preds,
        labels=np.arange(0, len(label_names)),
        target_names=label_names,
    )

    with open(f"{logs_folder}/classification.txt", "w") as class_file:
        class_file.write(
            f"micro-f1 w/o no-rel after {best_epoch} epochs: "
            f"{f1_float:.3}\n\n{creport}\n\n"
        )

    with open(f"{logs_folder}/confusion.txt", "w") as confusion_file:
        confusion_file.write(f"confusion matrix after {best_epoch} epochs:\n")
        np.savetxt(
            confusion_file,
            confusion_matrix(
                concat_labels, concat_preds, labels=np.arange(0, len(label_names))
            ),
            fmt="%i",
            delimiter=",",
        )

    if "save_model" in config.keys() and config["save_model"]:
        torch.save(
            {"model_state_dict": full_model.state_dict()}, f"{logs_folder}/model.pt"
        )


def evaluate_model(model, dataloader, labels, tokenizer, device):
    with torch.no_grad():
        model.eval()

        label_list = []
        pred_list = []
        progress_bar = tqdm(range(len(dataloader)))
        for batch in dataloader:
            batch_encoding = tokenize_batch(batch, tokenizer)
            batch_encoding = {k: v.to(device) for k, v in batch_encoding.items()}

            batch_labels = [labels.str2int(relation) for relation in batch["relation"]]
            label_list.append(batch_labels)

            start_counters = calc_start_counters(
                batch_encoding["input_ids"], len(tokenizer), device
            )
            output = model(start_counters, **batch_encoding)

            pred = torch.argmax(output, dim=1).detach().cpu().numpy()
            pred_list.append(pred)
            progress_bar.update(1)

    return label_list, pred_list


if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    config = yaml.safe_load(open(sys.argv[1]))
    logs_folder = f"logs/{datetime.today().strftime('%Y%m%d-%H%M%S')}"
    os.mkdir(logs_folder)
    shutil.copy(sys.argv[1], logs_folder)

    baseline_run(config, logs_folder)