import urllib
import bs4
import re
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec

source = urllib.request.urlopen("https://en.wikipedia.org/wiki/Global_warming")
soup = bs4.BeautifulSoup(source, 'lxml')

text = ""
for paragraph in soup.find_all('p'):
    text = text + paragraph.text

text = re.sub(r"\[[0-9]*\]", " ", text)
text = re.sub(r"\s+", " ", text)
text = text.lower()
text = re.sub(r"[@#\$%\&\*\(\)\<\>\?\':;\]\[-]", " ", text) # Eliminated regex used to remove all non-words and replaced with special chars
text = re.sub(r"\d", " ", text)
text = re.sub(r"\s+", " ", text)

# Transforming data which is acceptable by gensim word2vec model

sentences = nltk.sent_tokenize(text)
sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

# Removing stopwords to enhance the model
for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]

# Training model
model = Word2Vec(sentences, min_count=1)   # min_count = 1 means we are ignoring frequency less than 1
words = model.wv.vocab                     # words used

# Testing and performance of model
# Vector Representation
vector = model.wv['global']                 # 'global' word vector generation
similar = model.wv.most_similar('warming')    # checking similar words to global
print(similar)