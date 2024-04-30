import pandas as pd

df = pd.read_csv('/home/ubuntu/mscoco/train_neg_clip_merged_decompos_extra_more_than13.csv') 
filepath_counts = df['filepath'].value_counts().rename('num_pos_caption')
df = df.merge(filepath_counts, on='filepath', how='left')
df.to_csv('/home/ubuntu/mscoco/train_neg_clip_merged_decompos_extra_more_than13_v2.csv', index=False)
