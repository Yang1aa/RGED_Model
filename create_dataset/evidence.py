import csv
import os

from .utils import get_named_entities, get_entity_information, filter_relevant_content


def process_statements(datasets, splits, base_dir, type):
    for dataset in datasets:
        for split in splits:
            csv_path = os.path.join(base_dir, f'{dataset}.{split}.csv')
            if type == 'wiki':
                output_path = os.path.join("mid", 'wiki_mid')
            else:
                output_path = os.path.join("mid", 'gpt_mid')
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            output_path = os.path.join(output_path, f'{dataset}.{split}.csv')
            print(f"Processing {csv_path}...")

            with open(csv_path, 'r', encoding='utf-8-sig') as infile, open(output_path, 'w', newline='', encoding='utf-8-sig') as outfile:
                reader = csv.DictReader(infile)
                fieldnames = reader.fieldnames + ['evidence']
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()

                for row in reader:
                    statement = row['tweet']
                    entities = get_named_entities(statement)
                    evidence = get_entity_information(entities, type)
                    filtered_evidence = filter_relevant_content(statement, evidence)
                    row['evidence'] = filtered_evidence
                    row['id'] = row['id']
                    writer.writerow(row)
                    outfile.flush()

def get_evidence(args, config):
    datasets = config['datasets']
    splits = config['splits']
    base_dir = args.base_dir
    process_statements(datasets, splits, base_dir, 'gpt')
    process_statements(datasets, splits, base_dir, 'wiki')


