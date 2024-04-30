import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/home/ubuntu/mscoco/train_neg_clip_merged.csv')
word_counts = []
for captions in df['neg_caption']:
    sentences = eval(captions)  
    for sentence in sentences:
        word_count = len(sentence.split())
        word_counts.append(word_count)

plt.figure(figsize=(10, 6))
plt.hist(word_counts, bins=range(min(word_counts), max(word_counts) + 1, 1), alpha=0.75, color='blue')
plt.title('Histogram of Word Counts in Negative Captions')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
plt.savefig("negative.png")


count_above_15 = 0
for captions in df['neg_caption']:
    sentences = eval(captions)  
    for sentence in sentences:
        word_count = len(sentence.split())
        if word_count > 16:
            count_above_15 += 1

print(f"There are {count_above_15} sentences with more than 15 words.")
