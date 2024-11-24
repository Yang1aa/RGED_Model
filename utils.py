import csv
import logging
import os.path
import random
import re
import time
from collections import Counter

import nltk
import numpy as np
import pandas as pd
import requests
import torch
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer, util
from torch.optim import AdamW, SGD, Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch_geometric.data import Data, Batch
from transformers import get_cosine_schedule_with_warmup


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def synonym_replacement(sentence, n=1):
    words = sentence.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word.isalpha()]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = set()
        for syn in wordnet.synsets(random_word):
            for lemma in syn.lemmas():
                synonym = lemma.name()
                if synonym != random_word and synonym.isalpha():
                    synonyms.add(synonym.replace('_', ' '))
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    sentence = ' '.join(new_words)
    return sentence


def random_deletion(sentence, p=0.05):
    words = sentence.split()
    if len(words) == 1:
        return sentence
    new_words = [word for word in words if random.uniform(0, 1) > p]
    if len(new_words) == 0:
        new_words = [random.choice(words)]
    sentence = ' '.join(new_words)
    return sentence


def random_swap(sentence, n=1):
    words = sentence.split()
    length = len(words)
    for _ in range(n):
        if length < 2:
            break
        idx1, idx2 = random.sample(range(length), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    sentence = ' '.join(words)
    return sentence


def check_class_distribution(data_list, dataset_name):
    labels = [data.y.item() for data in data_list]
    counter = Counter(labels)
    print(f"{dataset_name} set class distribution: {counter}")


def create_graph_data_from_csv(csv_files, label_mapping, edge_type_mapping, augment=False):
    data_dict = {}
    all_texts = []
    all_labels = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file, encoding='ISO-8859-1')
        df['label'] = df['label'].str.lower().map(label_mapping)
        df['predicted_label'] = df['predicted_label'].map(edge_type_mapping)

        if 'tweet_id' not in df.columns:
            df['tweet_id'] = df.index

        for index, row in df.iterrows():
            tweet_id = row['tweet_id']
            tweet = row['tweet']
            evidence = row['evidence']
            label = row['label']
            predicted_label = row['predicted_label']

            if pd.isna(label) or pd.isna(predicted_label) or pd.isna(tweet) or pd.isna(evidence):
                print(f"Skipping entry due to NaN value at index {index} in file {csv_file}.")
                continue

            if augment:
                aug_choice = random.choice(['synonym_replacement', 'random_deletion', 'random_swap', 'none'])
                if aug_choice == 'synonym_replacement':
                    tweet = synonym_replacement(tweet, n=1)
                elif aug_choice == 'random_deletion':
                    tweet = random_deletion(tweet, p=0.05)
                elif aug_choice == 'random_swap':
                    tweet = random_swap(tweet, n=1)

            evidence_list = [e.strip() for e in str(evidence).split('||')]

            if tweet_id in data_dict:
                data_dict[tweet_id]['evidence_list'].extend(evidence_list)
                data_dict[tweet_id]['predicted_labels'].extend([predicted_label] * len(evidence_list))
            else:
                data_dict[tweet_id] = {
                    'tweet': tweet,
                    'evidence_list': evidence_list,
                    'label': label,
                    'predicted_labels': [predicted_label] * len(evidence_list)
                }

    data_list = []
    for tweet_id, data_item in data_dict.items():
        tweet = data_item['tweet']
        evidence_list = data_item['evidence_list']
        label = data_item['label']
        predicted_labels = data_item['predicted_labels']

        num_nodes = 1 + len(evidence_list)

        edge_index_list = []
        edge_type_list = []
        for i, (evidence, pred_label) in enumerate(zip(evidence_list, predicted_labels)):
            edge_index_list.append([0, i + 1])
            edge_index_list.append([i + 1, 0])
            edge_type_list.append(pred_label)
            edge_type_list.append(pred_label)

        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_type_list, dtype=torch.long)

        graph_data = Data(
            tweet_text=tweet,
            evidence_text=evidence_list,
            edge_index=edge_index,
            edge_type=edge_type,
            y=torch.tensor([label], dtype=torch.long),
            num_nodes=num_nodes
        )

        data_list.append(graph_data)
        all_texts.append(tweet)
        all_labels.append(label)

    return data_list, all_texts, all_labels

def custom_collate(data_list):
    batch = Batch.from_data_list(data_list)
    batch.tweet_text = [data.tweet_text for data in data_list]
    batch.evidence_text = [data.evidence_text for data in data_list]  # 现在 evidence_text 是列表的列表
    return batch

def setup_logging(args):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    save_path = os.path.join(args.log_dir,"training.log")
    logging.basicConfig(
        filename=save_path,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def get_files(dataset_name, gpt, wiki):
    germanwings_crash_files = None
    if dataset_name == 't15':
        train = 'twitter15.train_predicted.csv'
        dev = 'twitter15.dev_predicted.csv'
        test = 'twitter15.test_predicted.csv'
        germanwings_crash_files = {
            "train": [
                gpt + train,
                wiki + train
            ],
            "dev": [
                gpt + dev,
                wiki + dev
            ],
            "test": [
                gpt + test,
                wiki + test
            ]
        }

    if dataset_name == 't16':
        train = 'twitter16.train_predicted.csv'
        dev = 'twitter16.dev_predicted.csv'
        test = 'twitter16.test_predicted.csv'
        germanwings_crash_files = {
            "train": [
                gpt + train,
                wiki + train
            ],
            "dev": [
                gpt + dev,
                wiki + dev
            ],
            "test": [
                gpt + test,
                wiki + test
            ]
        }

    if dataset_name == 'pheme':
        train1 = 'germanwings-crash.train_predicted.csv'
        train2 = 'ottawashooting.train_predicted.csv'
        dev1 = 'ottawashooting.dev_predicted.csv'
        dev2 = 'ottawashooting.dev_predicted.csv'
        test1 = 'ottawashooting.test_predicted.csv'
        test2 = 'ottawashooting.test_predicted.csv'
        germanwings_crash_files = {
            "train": [
                gpt + train1,
                gpt + train2,
                wiki + train1,
                wiki + train2
            ],
            "dev": [
                gpt + dev1,
                gpt + dev2,
                wiki + dev1,
                wiki + dev2
            ],
            "test": [
                gpt + test1,
                gpt + test2,
                wiki + test1,
                wiki + test2
            ]
        }

    return germanwings_crash_files

def create_optimizer(model, optimizer_config):
    opt_type = optimizer_config.get('opt', 'adamw').lower()
    lr = float(optimizer_config.get('lr', 1e-3))
    weight_decay = float(optimizer_config.get('weight_decay', 0.0))
    momentum = float(optimizer_config.get('momentum', 0.0))
    eps = float(optimizer_config.get('eps', 1e-8))

    if opt_type == 'adamw':
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == 'sgd':
        optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif opt_type == 'adam':
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay, eps=eps)
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_type}")

    return optimizer


def create_lr_scheduler(optimizer, scheduler_config, total_steps):
    scheduler_type = scheduler_config.get('type', 'step').lower()
    step_size = scheduler_config.get('step_size', 10)
    gamma = scheduler_config.get('gamma', 0.1)
    t_max = scheduler_config.get('t_max', 50)
    warmup_proportion = scheduler_config.get('warmup_proportion', 0.1)

    if scheduler_type == 'step':
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=t_max)
    elif scheduler_type == 'cosine_with_warmup':
        warmup_steps = int(warmup_proportion * total_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    return scheduler


def clean_generated_text(generated_text):
    cleaned_text = re.sub(r'[^\w\s]', '', generated_text)
    cleaned_text = cleaned_text.lower()
    keywords = cleaned_text.split()
    return keywords

def get_wikipedia_content(keyword):
    search_query = keyword
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": search_query,
        "utf8": 1,
        "srlimit": 1
    }

    while True:
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if 'query' in data and 'search' in data['query']:
                    search_results = data["query"]["search"]
                    if search_results:
                        page_title = search_results[0]["title"]
                        return get_full_wikipedia_content(page_title)
                    else:
                        return None
                else:
                    return None
            else:
                print(f"无法获取关键词 '{keyword}' 的数据，状态码: {response.status_code}。重试中...")
                time.sleep(2)
        except requests.exceptions.RequestException as e:
            print(f"请求失败: {e}。重试中...")
            time.sleep(2)

def get_full_wikipedia_content(title):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "exintro": False
    }

    while True:
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                pages = data["query"]["pages"]
                for page_id in pages:
                    if "extract" in pages[page_id]:
                        return pages[page_id]["extract"]
                    else:
                        return None
            else:
                print(f"无法获取标题 '{title}' 的完整内容，状态码: {response.status_code}。重试中...")
                time.sleep(2)
        except requests.exceptions.RequestException as e:
            print(f"请求失败: {e}。重试中...")
            time.sleep(2)

def split_sentences(text):
    nltk_data_path = os.path.join(os.path.expanduser('~'), 'nltk_data')

    if not os.path.exists(os.path.join(nltk_data_path, 'tokenizers', 'punkt')):
        nltk.download('punkt', download_dir=nltk_data_path)

    nltk.data.path.append(nltk_data_path)

    return nltk.sent_tokenize(text)

def compute_cosine_similarity(sentences, statement):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(sentences + [statement], convert_to_tensor=True)
    cosine_similarities = util.pytorch_cos_sim(embeddings[-1], embeddings[:-1]).flatten().cpu().numpy()
    return cosine_similarities

def process_article_and_statement(article, statement, top_n=5):
    sentences = split_sentences(article)
    cosine_similarities = compute_cosine_similarity(sentences, statement)
    top_sentences = get_top_sentences(sentences, cosine_similarities, top_n)
    return top_sentences

def get_top_sentences(sentences, cosine_similarities, top_n=5):
    top_n = min(top_n, len(sentences))
    top_indices = cosine_similarities.argsort()[-top_n:][::-1]
    top_sentences = [sentences[i] for i in top_indices]

    unique_sentences = []
    seen = set()
    for sentence in top_sentences:
        if sentence not in seen:
            unique_sentences.append(sentence)
            seen.add(sentence)

    return unique_sentences

def process_tweets(tweet_texts, generated_texts, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as outfile:
        fieldnames = ['tweet_text', 'evidence']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(len(tweet_texts)):
            row = {}
            tweet_text = tweet_texts[i]
            row['tweet_text'] = tweet_text
            generated_text = generated_texts[i]

            keywords = clean_generated_text(generated_text)

            evidences = []
            for keyword in keywords:
                evidence = get_wikipedia_content(keyword)
                if evidence:
                    evidences.append(evidence)

            if not evidences:
                print(f"未找到推文任何证据")
                row['evidence'] = ''
                writer.writerow(row)
                continue

            highest_similarity = 0
            best_evidence = ''

            for evidence in evidences:
                top_sentences = process_article_and_statement(evidence, tweet_text, top_n=5)
                combined_sentences = ' '.join(top_sentences)
                cosine_similarities = compute_cosine_similarity([combined_sentences], tweet_text)
                similarity = cosine_similarities[0]
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_evidence = combined_sentences

            row['evidence'] = best_evidence
            writer.writerow(row)

