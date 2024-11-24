import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from feverDataset import FEVERDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    print(f"Using device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA is not available. Using CPU.")


def train_model(model, train_loader, val_loader, epochs=10, lr=1e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_loader = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch in train_loader:
            inputs, labels = batch
            inputs = {key: val.to(device) for key, val in inputs.items()}
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(**inputs).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_loader.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')

        evaluate(model, val_loader, criterion, device)


def evaluate(model, val_loader, criterion, device):
    model.eval()
    total, correct = 0, 0
    total_loss = 0
    with torch.no_grad():
        val_loader = tqdm(val_loader, desc="Evaluating")
        for batch in val_loader:
            inputs, labels = batch
            inputs = {key: val.to(device) for key, val in inputs.items()}
            labels = labels.to(device)

            outputs = model(**inputs).logits
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    print(f'Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')


def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

    train_dataset = FEVERDataset('./train.jsonl', './wiki-pages', tokenizer)
    val_dataset = FEVERDataset('./shared_task_dev.jsonl', './wiki-pages', tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    train_model(model, train_loader, val_loader, epochs=15, lr=3e-5)

    model.save_pretrained('./models')

if __name__ == "__main__":
    main()