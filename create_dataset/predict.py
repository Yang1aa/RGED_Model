import pandas as pd

import os

from create_dataset.utils import predict


def process_file(args, config):
    folder_path = r"mid"
    file_list = []
    for filename in os.listdir(folder_path):
        for fn in os.listdir(os.path.join(folder_path, filename)):
            if fn.endswith('.csv'):
                file_path = os.path.join(folder_path, filename, fn)
                file_list.append(file_path)
    output_dir = 'predict'
    for file_path in file_list:
        df = pd.read_csv(file_path)
        result_df = pd.DataFrame(columns=['id', 'tweet', 'label', 'evidence', 'predicted_label'])
        for index, row in df.iterrows():
            tweet = row['tweet']
            if 'evidence' in row and not pd.isna(row['evidence']):
                evidence_text = row['evidence']
            else:
                continue
            results = predict(tweet, [evidence_text])
            predicted_label = results.get(evidence_text, 'NOT ENOUGH INFO')
            result_df = pd.concat([result_df, pd.DataFrame([{
                'id': row['id'],
                'tweet': tweet,
                'label': row['label'],
                'evidence': evidence_text,
                'predicted_label': predicted_label
            }])], ignore_index=True)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file_name = file_path.lstrip("mid\\\\")
        if output_file_name.startswith("gpt_mid"):
            output_file_name = "gpt_predict" + output_file_name[7:]
        elif output_file_name.startswith("wiki_mid"):
            output_file_name = "wiki_predict" + output_file_name[8:]
        output_file_path = os.path.join(output_dir, os.path.dirname(output_file_name))
        if not os.path.exists(output_file_path):
            os.makedirs(output_file_path)
        output_file_path = os.path.join(output_dir, output_file_name)
        result_df.to_csv(output_file_path, index=False)
        print(f"Processed {file_path} and saved results to {output_file_path}")
