import json
import os
import torch
from torch.utils.data import Dataset


class FEVERDataset(Dataset):
    LABELS = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}

    def __init__(self, data_file, wiki_pages_dir, tokenizer, max_length=512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.wiki_data = self.load_wiki_pages(wiki_pages_dir)

        # Debug: Check if loadModel exists
        if not os.path.exists(data_file):
            print(f"File not found: {data_file}")
            raise FileNotFoundError(f"File not found: {data_file}")

        with open(data_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

    def load_wiki_pages(self, wiki_pages_dir):
        wiki_data = {}
        for filename in os.listdir(wiki_pages_dir):
            if filename.endswith('.jsonl'):
                with open(os.path.join(wiki_pages_dir, filename), 'r') as f:
                    for line in f:
                        page = json.loads(line)
                        wiki_data[page['id']] = page['lines']
        return wiki_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        claim = item['claim']
        evidence = self.get_evidence_text(item['evidence'])
        label = self.LABELS[item['label']]  # Convert label to integer
        inputs = self.tokenizer(claim, evidence, truncation=True, padding='max_length', max_length=self.max_length,
                                return_tensors='pt')
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        return inputs, torch.tensor(label)

    def get_evidence_text(self, evidence):
        evidence_texts = []
        for ev_set in evidence:
            for ev in ev_set:
                page_id = ev[2]
                line_num = ev[3]
                if page_id and line_num is not None:
                    lines = self.wiki_data.get(page_id, "")
                    line_text = lines.split('\n')[line_num].split('\t')[1] if lines else ""
                    evidence_texts.append(line_text)
        return " ".join(evidence_texts)
