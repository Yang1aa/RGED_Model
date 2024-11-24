import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import torch

from utils import process_tweets


def rumor_test(model, test_loader, fold, output_csv, device, args):
    # 加载模型权重
    model_path = os.path.join(args.output_dir, args.dataset, "model", args.checkpoint)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []
    all_generated_texts = []
    mispredictions = []

    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            logits, generated_texts = model(data)
        predicted_labels = torch.argmax(logits, dim=1)
        all_predictions.extend(predicted_labels.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())
        all_generated_texts.extend(generated_texts)

        for i in range(len(predicted_labels)):
            if predicted_labels[i].item() != data.y[i].item():
                mispredictions.append({
                    'tweet_text': data.tweet_text[i],
                    'evidence_text': data.evidence_text[i],
                    'true_label': data.y[i].item(),
                    'predicted_label': predicted_labels[i].item(),
                    'generated_text': generated_texts[i]
                })

    print("Predicted Labels:", all_predictions)
    print("True Labels:", all_labels)

    cm = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:")
    print(cm)

    unique_labels = sorted(list(set(all_labels) | set(all_predictions)))

    if args.dataset in ['t15', 't16']:
        label_names = ['unverified', 'non-rumor', 'true', 'false']
    else:
        label_names = ['Rumor', 'Non-Rumor']

    target_names = [label_names[i] for i in unique_labels if i < len(label_names)]

    report = classification_report(all_labels, all_predictions, target_names=target_names,
                                   labels=unique_labels, zero_division=0)
    print("Classification Report:")
    print(report)

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average=None, labels=unique_labels, zero_division=0)
    recall = recall_score(all_labels, all_predictions, average=None, labels=unique_labels, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average=None, labels=unique_labels, zero_division=0)

    if args.dataset in ['t15', 't16']:
        results = {
            "Accuracy": [round(accuracy, 3)],
            "Unverified F1": [round(f1[0], 3)],
            "Non-Rumor F1": [round(f1[1], 3)],
            "True F1": [round(f1[2], 3)],
            "False F1": [round(f1[3], 3)]
        }
    else:
        results = {
            "Accuracy": [round(accuracy, 3)],
            "Non-Rumor Precision": [round(precision[unique_labels.index(1)] if 1 in unique_labels else 0, 3)],
            "Rumor Precision": [round(precision[unique_labels.index(0)] if 0 in unique_labels else 0, 3)],
            "Non-Rumor Recall": [round(recall[unique_labels.index(1)] if 1 in unique_labels else 0, 3)],
            "Rumor Recall": [round(recall[unique_labels.index(0)] if 0 in unique_labels else 0, 3)],
            "Non-Rumor F1": [round(f1[unique_labels.index(1)] if 1 in unique_labels else 0, 3)],
            "Rumor F1": [round(f1[unique_labels.index(0)] if 0 in unique_labels else 0, 3)]
        }

    results_df = pd.DataFrame(results)
    output_path = os.path.join(args.output_dir, args.dataset, "result")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    results_df.to_csv(os.path.join(output_path, output_csv), index=False, encoding='utf-8-sig')

    log_path = args.log_dir
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    with open(os.path.join(log_path, "confusion_matrix.txt"), "w", encoding='utf-8') as f:
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n\nClassification Report:\n")
        f.write(report)

    mispredictions_df = pd.DataFrame(mispredictions)
    mispredictions_df.to_csv(os.path.join(log_path, "mispredictions.csv"), index=False, encoding='utf-8-sig')

    tweet_texts = []
    generated_texts = []
    for i in range(len(all_predictions)):
        tweet_texts.append(test_loader.dataset[i].tweet_text)
        generated_texts.append(all_generated_texts[i])

    process_tweets(tweet_texts, generated_texts, os.path.join(output_path, f"all_generated_explanations_{fold}.csv"))

    print(f"Confusion matrix and classification report saved to '{os.path.join(log_path, 'confusion_matrix.txt')}'.")
    print(f"Mispredicted samples saved to '{os.path.join(log_path, 'mispredictions.csv')}'.")
    print(f"All generated explanations saved to '{os.path.join(output_path, 'all_generated_explanations.csv')}'.")

    return accuracy
