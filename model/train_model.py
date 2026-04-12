from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Sample data
texts = ["I love this", "This is bad", "Amazing product", "Worst experience"]
labels = [1, 0, 1, 0]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(X, labels)

# Save model
with open("model/model.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)