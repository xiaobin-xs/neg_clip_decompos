import pandas as pd

file_a = pd.read_csv('/home/ubuntu/mscoco/train_neg_clip_merged_decompos_extra_more_than14.csv')
file_b = pd.read_csv('/home/ubuntu/mscoco/train_neg_clip_merged_decompos_extra_13.csv')

unique_filepaths = file_a['filepath'].value_counts()
unique_filepaths = unique_filepaths[unique_filepaths == 1].index.tolist()

file_a_unique = file_a[file_a['filepath'].isin(unique_filepaths)]
file_b_relevant = file_b[file_b['filepath'].isin(unique_filepaths)]

file_a_final = file_a[~file_a['filepath'].isin(unique_filepaths)]

resulting_dataframe = pd.concat([file_a_final, file_b_relevant], ignore_index=True)

resulting_dataframe.to_csv('train_neg_clip_merged_decompos_extra_more_than13.csv', index=False)
