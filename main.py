# Shaghik Issakhani
# 03-20-22
# HW5

import spacy
import en_core_web_lg
from newsapi import NewsApiClient
import pickle
import pandas as pd
import numpy as np
from collections import Counter
import nltk
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from wordcloud import WordCloud
import string

nlp_eng = en_core_web_lg.load()
newsapi = NewsApiClient(api_key='92a9a24b2baf4cbfb4b335e29567f375')

temp = newsapi.get_everything(q='coronavirus', language='en', from_param='2022-02-28', to='2020-03-20',
                              sort_by='relevancy')

filename = 'covidArticles.pckl'
pickle.dump(temp, open(filename, 'wb'))
filename = 'covidArticles.pckl'
loaded_model = pickle.load(open(filename, 'rb'))
filepath = '/content/covidArticles.pckl'
pickle.dump(loaded_model, open(filepath, 'wb'))

df = pd.DataFrame(temp['articles'])

tokenizer = RegexpTokenizer(r'\w+')


def getKeywords(token):
    result = []
    punctuation = string.punctuation
    stop_words = stopwords.words('english')

    for i in token:
        if (i in stop_words):
            continue
        else:
            result.append(i)
    print(result)
    return result


with open('covidArticles.pckl', 'rb') as f:
    data = pickle.load(f)

print(data)

results = []
for content in df.content.values:
    content = tokenizer.tokenize(content)
    results.append([x[0] for x in Counter(getKeywords(content)).most_common(5)])
df['keywords'] = results

text = str(results)
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

