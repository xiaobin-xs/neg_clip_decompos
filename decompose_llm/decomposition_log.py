import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import logging

logging.basicConfig(filename='error_log_13.txt', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

client = OpenAI()

def simplify_title(title):
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": '''Given a long sentence that describe a complex scene, you need to split this sentence into multiple sentences that describe the original scene.
            Each sentence need to be separated by newlines. Do not add additonal words. Each sentence must have a clear subject, don't simply use "him", "he", "her", "she", "it", "they" or "them" as a pronoun, use a specific noun instead.
            You do not need to have the number for each sentence.'''},
            {"role": "user", "content": f'Here is the sentence: \"{title}\"'},
    ]
    )
    simplified = response.choices[0].message.content.strip().split('\n')
    logging.info(f'Original: {title} | Simplified: {simplified}')
    return simplified

data = pd.read_csv('/home/ubuntu/mscoco/train_neg_clip_merged.csv')
new_rows = []

for index, row in tqdm(data.iterrows(), total=data.shape[0], desc="Processing Titles"):
    # if len(row['title'].split()) > 14:
    if len(row['title'].split()) == 13:
        try:
            new_titles = simplify_title(row['title'])
            for new_title in new_titles:
                new_row = row.copy()
                new_row['title'] = new_title
                new_rows.append(new_row)
        except Exception as e:
            logging.error(f'Error processing title: {row["title"]}. Error: {e}')
            new_rows.append(row)
    else:
        new_rows.append(row)

new_data = pd.DataFrame(new_rows)
new_data.to_csv('train_neg_clip_merged_decompos_extra_13.csv', index=False)
