import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertForSequenceClassification
from tqdm import tqdm


class RumorDetectionModel(nn.Module):
    def __init__(self, args, num_labels):
        super(RumorDetectionModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(args.bert, num_labels=num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return outputs.logits


def train_model(model, train_loader, val_loader, epochs=3, lr=2e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_loader_iter = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch in train_loader_iter:
            inputs, labels = batch
            inputs = {key: val.to(device) for key, val in inputs.items()}
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_loader_iter.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}, Training Loss: {avg_loss:.4f}')

        evaluate(model, val_loader, criterion, device)


def evaluate(model, val_loader, criterion, device):
    model.eval()
    total, correct = 0, 0
    total_loss = 0
    val_loader_iter = tqdm(val_loader, desc="Evaluating")

    with torch.no_grad():
        for batch in val_loader_iter:
            inputs, labels = batch
            inputs = {key: val.to(device) for key, val in inputs.items()}
            labels = labels.to(device)
            outputs = model(**inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    print(f'Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.4f}')
