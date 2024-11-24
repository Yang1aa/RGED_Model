import torch
import torch.nn as nn
from sympy import false
from torch_geometric.loader import DataLoader as GeoDataLoader
import random
from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW
from collections import Counter
from sklearn.model_selection import StratifiedKFold

from models.Rumor_detection import RumorDetectionModel
from models.bert import BERTEncoder
from models.connnect import FullyConnected
from models.explain import ExplanationLayer
from models.rgcn import RGCN
from train.rumor_train import rumor_train
from test.rumor_test import rumor_test
from utils import *


def train_model(args, config):
    seed = args.seed
    set_random_seed(seed)
    num_epochs = args.epochs
    num_folds = args.folds
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    gpt = config['gpt_dataset_file'][0]
    wiki = config['wiki_dataset_file'][0]

    germanwings_crash_files = get_files(args.dataset, gpt, wiki)

    if args.dataset == 't16' or args.dataset == 't15':
        label_mapping = {'unverified': 0, 'non-rumor': 1, 'true': 2, 'false': 3}
        edge_type_mapping = {'SUPPORTS': 0, 'REFUTES': 1}
    else:
        label_mapping = {'rumor': 0, 'non-rumor': 1}
        edge_type_mapping = {'SUPPORTS': 0, 'REFUTES': 1}

    train_data_list, train_texts, train_labels = create_graph_data_from_csv(
        germanwings_crash_files["train"], label_mapping, edge_type_mapping, augment=True)

    dev_data_list, _, _ = create_graph_data_from_csv(
        germanwings_crash_files["dev"], label_mapping, edge_type_mapping)
    test_data_list, _, _ = create_graph_data_from_csv(
        germanwings_crash_files["test"], label_mapping, edge_type_mapping)

    all_data_list = train_data_list + dev_data_list + test_data_list
    all_labels = [data.y.item() for data in all_data_list]

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_data_list, all_labels)):
        print(f"Fold {fold + 1}/{num_folds}")

        train_data_fold = [all_data_list[i] for i in train_idx]
        val_data_fold = [all_data_list[i] for i in val_idx]

        label_counts = Counter([data.y.item() for data in train_data_fold])
        max_count = max(label_counts.values())

        balanced_train_data_fold = []
        for label in label_counts:
            label_data = [data for data in train_data_fold if data.y.item() == label]
            num_to_add = max_count - label_counts[label]
            if num_to_add > 0:
                label_data_extended = label_data * (num_to_add // len(label_data)) + label_data[
                                                                                     :num_to_add % len(label_data)]
                balanced_train_data_fold.extend(label_data_extended)
            balanced_train_data_fold.extend(label_data)

        random.shuffle(balanced_train_data_fold)

        print("Checking class distribution for this fold...")
        check_class_distribution(balanced_train_data_fold, "Training")
        check_class_distribution(val_data_fold, "Validation")

        train_loader = GeoDataLoader(balanced_train_data_fold, batch_size=config['batch_size'], shuffle=True,
                                     collate_fn=custom_collate)
        val_loader = GeoDataLoader(val_data_fold, batch_size=config['batch_size'], shuffle=False,
                                   collate_fn=custom_collate)

        model_name = args.bert
        encoder = BERTEncoder(model_name).to(device)
        rgcn_model = RGCN(in_channels=config['in_channels'], out_channels=config['out_channels'], num_relations=2).to(
            device)
        explanation_model = ExplanationLayer(config['out_channels']).to(device)
        if args.dataset == 't16' or args.dataset == 't15':
            fc_model = FullyConnected(config['out_channels'], output_dim=4).to(device)
        else:
            fc_model = FullyConnected(config['out_channels'], output_dim=2).to(device)

        rumor_detection_model = RumorDetectionModel(encoder, rgcn_model, explanation_model, fc_model, args, decoder_model_name=args.bart).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer_config = config.get('optimizer',{})
        optimizer = create_optimizer(rumor_detection_model, optimizer_config)

        total_steps = len(train_loader) * num_epochs
        scheduler_config = config.get('scheduler', {})
        scheduler = create_lr_scheduler(optimizer, scheduler_config, total_steps)

        print(f"Training on fold {fold + 1}...")
        rumor_train(rumor_detection_model, train_loader, val_loader, optimizer, criterion, scheduler,
                    device, args, num_epochs=num_epochs, patience=config['patience'])

        print(f"Evaluating on fold {fold + 1} validation set...")
        accuracy = rumor_test(rumor_detection_model, val_loader, fold,
                              output_csv=f"evaluation_results_fold_{args.dataset}_{fold + 1}.csv",
                              device=device, args=args)
        fold_accuracies.append(accuracy)

        avg_accuracy = sum(fold_accuracies) / len(fold_accuracies)
        print(f"Average accuracy across folds: {avg_accuracy:.4f}")
        print("Cross-validation completed.")
