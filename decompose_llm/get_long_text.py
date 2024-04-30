import pandas as pd

data = pd.read_csv('/home/ubuntu/mscoco/train_neg_clip_merged.tsv', sep='\t')
filtered_titles = data[data['title'].apply(lambda x: len(x.split()) > 13)]
filtered_titles['title'].to_csv('titles_over_13_words.txt', index=False, header=False)
