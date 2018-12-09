import collections as coll
import pandas as pd
import numpy as np

count = 999
num_col = 100

#Read Data Set 
df = pd.read_json('yelp_academic_dataset_review1K.json',lines = True).head(count)


df2 = df['text'].str.split(' ')

#Counts the frequencies of every word in the dataset
wcount_total = coll.Counter()
for i in range(0,count):
    for j in df2.iloc[i]:
        wcount_total[j] += 1

ignored_words = {'The','the','it','and','a'}

for s in ignored_words:
    if s in wcount_total:
        del wcount_total[s]
        
#List of most frequent words
top_words = sorted(wcount_total, key=wcount_total.get, reverse = True)[0:num_col]

#Count word frequencies by row
line_count = coll.Counter()
s = []
for k in range(0,count):
    for r in range(0,num_col):
        s.append(float(df2.iloc[k].count(top_words[r])))
     
freq_matrix = pd.DataFrame(np.array(s).reshape(-count,num_col),columns = top_words)

norm_freq_matrix = freq_matrix/freq_matrix.sum(axis=0)


norm_freq_matrix['class'] = df['stars']

norm_freq_matrix['class'] = norm_freq_matrix['class'].map({0:0,1:0,2:0,3:1,4:1,5:1})

norm_freq_matrix.to_csv('reviews.csv', sep=',', encoding='utf-8',index=False)

print 'Dataset Created!'
