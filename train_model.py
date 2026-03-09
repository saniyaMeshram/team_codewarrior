import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier


data = pd.read_csv("news.csv")


data = data.dropna(subset=['text'])


X = data['text']
y = data['label']


vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vectorized = vectorizer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)


model = PassiveAggressiveClassifier(max_iter=1000)
model.fit(X_train, y_train)


accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)


pickle.dump(model, open("fake_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))