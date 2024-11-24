import logging
import os.path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

from utils import setup_logging


def rumor_train(model, train_loader, dev_loader, optimizer, criterion, scheduler, device, args, num_epochs=10,
                patience=3):
    setup_logging(args)
    best_dev_loss = np.inf
    patience_counter = 0
    save_path = args.output_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_preds, total_preds = 0, 0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", dynamic_ncols=True) as pbar:
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                logits, _ = model(data)
                loss = criterion(logits, data.y.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                running_loss += loss.item()
                predicted_labels = torch.argmax(logits, dim=1)
                correct_preds += (predicted_labels == data.y.view(-1)).sum().item()
                total_preds += data.y.size(0)

                pbar.set_postfix({"Loss": running_loss / (pbar.n + 1)})
                pbar.update(1)

        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = correct_preds / total_preds
        logging.info(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        model.eval()
        dev_loss = 0.0
        all_dev_predictions = []
        all_dev_labels = []
        target_names = ['unverified', 'non-rumor', 'true', 'false']
        with torch.no_grad():
            for data in dev_loader:
                data = data.to(device)
                logits, _ = model(data)
                loss = criterion(logits, data.y.view(-1))
                dev_loss += loss.item()

                predicted_labels = torch.argmax(logits, dim=1)
                all_dev_predictions.extend(predicted_labels.cpu().numpy())
                all_dev_labels.extend(data.y.cpu().numpy())

        avg_dev_loss = dev_loss / len(dev_loader)
        dev_accuracy = accuracy_score(all_dev_labels, all_dev_predictions)

        unique_labels = list(set(all_dev_labels) | set(all_dev_predictions))
        dev_report = classification_report(all_dev_labels, all_dev_predictions,
                                           target_names=[target_names[i] for i in unique_labels], labels=unique_labels,
                                           zero_division=0)
        logging.info(f"Epoch {epoch + 1}, Dev Loss: {avg_dev_loss:.4f}")
        logging.info(f"Validation Accuracy: {dev_accuracy:.4f}")
        logging.info("Validation Classification Report:")
        logging.info("\n" + dev_report)

        if avg_dev_loss < best_dev_loss:
            best_dev_loss = avg_dev_loss
            patience_counter = 0
            save_model = os.path.join(save_path, args.dataset, "model")
            if not os.path.exists(save_model):
                os.makedirs(save_model)
            save = os.path.join(save_model, args.checkpoint)
            torch.save(model.state_dict(), save)
            logging.info("Best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info("Early stopping triggered.")
                break
