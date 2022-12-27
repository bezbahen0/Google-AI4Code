import sys
import logging
import argparse
import torch
import numpy as np

from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

from tqdm import tqdm
from .models import TransformersModel
from .dataset import TransformersDataset

from .config import Config

def train_transformer(
    data_path,
    output_model_path,
    data_fts_path,
    config,
):
    train_data = TransformersDataset(
        data_path,
        data_fts_path=data_fts_path,
        config=config
    )

    train_data.load_data()

    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.n_workers,
        pin_memory=False,
        drop_last=True,
    )

    model = TransformersModel(config.model_name_or_path)
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

    num_train_optimization_steps = int(config.epochs * len(train_loader) / config.accumulation_steps)
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

    for e in range(config.epochs):
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
            if idx % config.accumulation_steps == 0 or idx == len(tbar) - 1:
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

    parser.add_argument("--features_data_path", type=str)
    parser.add_argument("--config", type=str)
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.info("--TRAIN MODEL--")

    config = Config()
    config.load_config(args.config, logger)

    
    train_transformer(
        args.data,
        args.output,
        args.features_data_path,
        config
    )

    logger.info(f"Save trained {args.task} model to {args.output}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
