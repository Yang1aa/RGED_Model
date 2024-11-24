import requests
from openai import OpenAI
import time

import torch
from transformers import BertTokenizer, BertForSequenceClassification

client = OpenAI(
    api_key="",
    base_url=""
)

def gpt_35_api(messages: list) -> str:
    start_time = time.time()
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=500,
        temperature=1
    )
    end_time = time.time()
    response_time = end_time - start_time
    response_message = completion.choices[0].message.content
    print("Response Time:", response_time, "seconds")
    return response_message

def gpt_35_api_stream(messages: list) -> str:
    start_time = time.time()
    response_message = ""
    stream = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=messages,
        stream=True,
        temperature=1
    )
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            response_message += chunk.choices[0].delta.content
    end_time = time.time()
    response_time = end_time - start_time
    print("Response Time:", response_time, "seconds")
    return response_message
def get_named_entities(text):
    messages = [{'role': 'user',
                 'content': f'Based on NER (Named Entity Recognition) results, help me identify the entities in this "{ text }", so that I can use the wiki API to obtain relevant information. Please list each entity as a numbered list, such as 1. xxx 2. xxx, no more than 3 keywords and the entity should not contain url and #.'}]

    response_stream = gpt_35_api_stream(messages)

    entities_list_stream = []
    for line in response_stream.split('\n'):
        line = line.strip()
        if line and '.' in line:
            parts = line.split('. ', 1)
            if len(parts) == 2:
                entities_list_stream.append(parts[1])

    return entities_list_stream

def get_wikipedia_content(entities):
    search_query = ' '.join(entities)
    url = f"https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": search_query,
        "utf8": 1,
        "srlimit": 10
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
                        return "No results found for these entities."
                else:
                    return "No query results found."
            else:
                print(f"Failed to fetch data for entities, status code: {response.status_code}. Retrying...")
                time.sleep(2)
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}. Retrying...")
            time.sleep(2)

def get_full_wikipedia_content(title):
    url = f"https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "exlimit": 1,
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
                        return "No extract found for this entity."
            else:
                print(f"Failed to fetch full content for {title}, status code: {response.status_code}. Retrying...")
                time.sleep(2)
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}. Retrying...")
            time.sleep(2)

def get_entity_information(entities, type):
    if type == 'wiki':
        entity_info = get_wikipedia_content(entities)
        return entity_info
    elif type == 'gpt':
        entity_description_prompt = f"Generate a detailed and concise description for the following entities: {', '.join(entities)}"
        messages = [{'role': 'user', 'content': entity_description_prompt}]
        generated_content = gpt_35_api_stream(messages)
        return generated_content.strip()


def split_text(text, max_length):
    paragraphs = text.split('\n')
    current_length = 0
    current_chunk = []
    chunks = []

    for paragraph in paragraphs:
        paragraph_length = len(paragraph)
        if current_length + paragraph_length <= max_length:
            current_chunk.append(paragraph)
            current_length += paragraph_length
        else:
            chunks.append('\n'.join(current_chunk))
            current_chunk = [paragraph]
            current_length = paragraph_length

    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    return chunks

def filter_relevant_content(statement, evidence, max_length=3000):
    chunks = split_text(evidence, max_length)
    filtered_content = []

    for chunk in chunks:
        messages = [{'role': 'user',
                     'content': f'Filter out irrelevant content from the following evidence based on this statement: "{ statement }". Only keep the relevant parts:\n\n{ chunk }'}]

        filtered_chunk = gpt_35_api_stream(messages)
        filtered_content.append(filtered_chunk.strip())

    return '\n'.join(filtered_content)

def predict(claim, evidence_list):
    model_path = 'fever_model'

    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    results = {}

    for evidence in evidence_list:
        inputs = tokenizer(claim, evidence, truncation=True, padding='max_length', max_length=512, return_tensors='pt')

        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        predicted_class = torch.argmax(logits, dim=1).item()

        labels = {0: 'SUPPORTS', 1: 'REFUTES', 2: 'NOT ENOUGH INFO'}

        results[evidence] = labels[predicted_class]

    return results
