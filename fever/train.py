import os.path

from torch.utils.data import DataLoader
from transformers import BertTokenizer

from fever.feverDataset import FEVERDataset
from fever.model import RumorDetectionModel, train_model


def fever(args, config):
    tokenizer = BertTokenizer.from_pretrained(args.bert)
    model = RumorDetectionModel(args, config["num_labels"])
    save_path = "fever_model"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_dataset = FEVERDataset(config['train'], config['dataset'], tokenizer)
    val_dataset = FEVERDataset(config['dev'], config['dataset'], tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

    train_model(model, train_loader, val_loader)

    model.save_pretrained(save_path)

