import argparse
from email.policy import default

import yaml
from nltk.sem.chat80 import continent
from pyparsing import empty
from torch_geometric.graphgym import train

from create_dataset.predict import process_file
from fever.train import fever
from model import train_model
from create_dataset.evidence import get_evidence

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train rumor detection model")
    parser.add_argument('--config', default='config/model.yaml', type=str, help='Path to config file')
    parser.add_argument('--mode', default='model', type=str, help='Execution mode')
    parser.add_argument('--bert', default='roberta-base', type=str, help='Pretrained BERT model')
    parser.add_argument("--bart", default='facebook/bart-large', type=str, help='Pretrained BART model')
    parser.add_argument('--output_dir', default='result', type=str, help='Directory to save outputs')
    parser.add_argument('--log_dir', default='log', type=str, help='Directory to save logs')
    parser.add_argument('--checkpoint', default='rumor.pth', type=str, help='Path to checkpoint file')
    parser.add_argument('--dataset', default='pheme', type=str, help='Dataset name')
    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility')
    parser.add_argument('--epochs', default=40, type=int, help='Number of training epochs')
    parser.add_argument('--folds', default=2, type=int, help='Number of folds for cross-validation')
    parser.add_argument("--base_dir", default='predict', type=str, help="Base of dataset")

    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    print(args)
    print("****************************")
    print(config)
    print("****************************")

    if args.mode == 'model':
        train_model(args, config)
    if args.mode == 'evidence':
        get_evidence(args, config)
    if args.mode == 'fever':
        fever(args, config)
    if args.mode == 'predict':
        process_file(args, config)