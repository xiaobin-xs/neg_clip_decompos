import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import logging

logging.basicConfig(filename='error_log_neg.txt', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')


client = OpenAI()

def simplify_caption(caption):
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": '''Given a long sentence that describe a complex scene, you need to split this sentence into multiple sentences that describe the original scene.
            Each sentence need to be separated by newlines. Do not add additional words. Each sentence must have a clear subject, don't simply use "him", "he", "her", "she", "it", "they" or "them" as a pronoun, use a specific noun instead.
            You do not need to have the number for each sentence.'''},
            {"role": "user", "content": f'Here is the sentence: \"{caption}\"'},
        ]
    )
    simplified = response.choices[0].message.content.strip().split('\n')
    # Log successful simplification
    logging.info(f'Original: {caption} | Simplified: {simplified}')
    return simplified

data = pd.read_csv('/home/ubuntu/mscoco/train_neg_clip_merged.csv')[:200]
new_rows = []

for index, row in tqdm(data.iterrows(), total=data.shape[0], desc="Processing Captions"):
    captions = eval(row['neg_caption'])
    new_captions = []
    modified = False
    for caption in captions:
        if len(caption.split()) > 16:
            try:
                simplified_captions = simplify_caption(caption)
                new_captions.extend(simplified_captions)
                modified = True
            except Exception as e:
                logging.error(f'Error processing caption: {caption}. Error: {e}')
                new_captions.append(caption)
        else:
            new_captions.append(caption)
    
    if modified:
        new_row = row.copy()
        new_row['neg_caption'] = str(new_captions)
        new_rows.append(new_row)
    else:
        new_rows.append(row)

new_data = pd.DataFrame(new_rows)
new_data.to_csv('train_neg_clip_merged_decompos_neg.csv', index=False)
