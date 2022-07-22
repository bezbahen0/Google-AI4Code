import sys
import logging
import argparse
import torch
import numpy as np

from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

from tqdm import tqdm
from .models import XGBrankerModel, TransformersModel
from .dataset import XGBrankerDataSet, TransformersDataset


#def to_cuda(data):
#    return [d.cuda() for d in data[0]], data[1].cuda()


def train_xgbranker(data_path, output_model_path):
    dataset = XGBrankerDataSet(data_path)
    X_train, y_train, groups = dataset.load_data()

    model = XGBrankerModel()
    model.model.fit(X_train, y_train[:, 1], group=groups)
    model.model.save_model(output_model_path)


def train_transformer(
    data_path,
    output_model_path,
    data_fts_path,
    model_name_or_path,
    max_len,
    accumulation_steps,
    batch_size,
    epochs,
    n_workers,
):
    train_data = TransformersDataset(
        data_path,
        model_name_or_path=model_name_or_path,
        max_len=max_len,
        data_fts_path=data_fts_path,
    )

    train_data.load_data()

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=False,
        drop_last=True,
    )

    model = TransformersModel(model_name_or_path)
    model = model.cuda()

    # Creating optimizer and lr schedulers
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_optimization_steps = int(epochs * len(train_loader) / accumulation_steps)
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=3e-5, correct_bias=False
    )  # To reproduce BertAdam specific behavior set correct_bias=False

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.05 * num_train_optimization_steps,
        num_training_steps=num_train_optimization_steps,
    )  # PyTorch scheduler

    criterion = torch.nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler()

    for e in range(epochs):
        model.train()
        tbar = tqdm(train_loader, file=sys.stdout)
        loss_list = []
        preds = []
        labels = []

        for idx, data in enumerate(tbar):
            ids, mask, fts, target = data

            ids = ids.cuda()
            mask = mask.cuda()
            fts = fts.cuda()
            target = target.cuda()

            with torch.cuda.amp.autocast():
                pred = model(ids=ids, mask=mask, fts=fts)
                loss = criterion(pred, target)

            scaler.scale(loss).backward()
            if idx % accumulation_steps == 0 or idx == len(tbar) - 1:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            loss_list.append(loss.detach().cpu().item())
           
            avg_loss = np.round(np.mean(loss_list), 4)

            tbar.set_description(
                f"Epoch {e + 1} Loss: {avg_loss} lr: {scheduler.get_last_lr()}"
            )

            if idx % 10000 == 0:
                torch.save(
                    model.state_dict(), output_model_path + f".epoch_{e}_idx_{idx}"
                )

        torch.save(model.state_dict(), output_model_path)


def main():
    """Runs data processing scripts add features to clean data (saved in data/clean),
    splits the featurized data into training and test datasets and saves them as new
    dataset (in data/featurized)
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--task", type=str)

    parser.add_argument("--features_data_path", type=str)
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--max_len", type=int)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_workers", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.info("--TRAIN MODEL--")

    if args.task == "xgbranker":
        train_xgbranker(args.data, args.output)
    if args.task == "transformer":
        train_transformer(
            args.data,
            args.output,
            args.features_data_path,
            args.model_name_or_path,
            args.max_len,
            args.accumulation_steps,
            args.batch_size,
            args.epochs,
            args.n_workers,
        )

    logger.info(f"Save trained {args.task} model to {args.output}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
