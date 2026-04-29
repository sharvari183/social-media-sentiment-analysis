import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')

def clean_text(text):
    text = re.sub(r"http\S+|[^A-Za-z\s]", "", text)
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

# sample data
data = {
    "text": ["I love this", "Worst product", "It is okay"],
    "sentiment": ["positive", "negative", "neutral"]
}

df = pd.DataFrame(data)

df['cleaned'] = df['text'].apply(clean_text)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned'])

model = LogisticRegression()
model.fit(X, df['sentiment'])

def predict(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]

print(predict("I love this product"))
