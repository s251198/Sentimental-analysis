import nltk
from nltk.corpus import movie_reviews
import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from collections import Counter

# Data Collection
nltk.download('movie_reviews')
nltk.download('stopwords')
nltk.download('punkt')

def load_data():
    documents = [(list(movie_reviews.words(fileid)), category)
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]
    random.shuffle(documents)
    return documents

documents = load_data()

# Data Preprocessing
stop_words = set(stopwords.words('english'))

def preprocess(sentence):
    words = word_tokenize(sentence)
    words = [word.lower() for word in words if word.isalpha()]
    words = [word for word in words if word not in stop_words]
    return words

# Inspecting preprocessed documents
preprocessed_docs = [preprocess(' '.join(doc)) for doc, _ in documents[:5]]
print("Sample Preprocessed Documents:", preprocessed_docs)

# Feature Extraction
def extract_features(documents):
    vectorizer = CountVectorizer(analyzer=lambda x: x)
    features = vectorizer.fit_transform([' '.join(doc) for doc, _ in documents])
    return features, vectorizer

# Check Label Distribution
def check_label_distribution(documents):
    labels = [label for _, label in documents]
    return Counter(labels)

print(check_label_distribution(documents))

# Model Training
def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=200, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

# Evaluate Model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Testing Data Evaluation")
    print(classification_report(y_test, y_pred))

# Evaluate Model on Training Data
def evaluate_model_on_train_data(model, X_train, y_train):
    y_train_pred = model.predict(X_train)
    print("Training Data Evaluation")
    print(classification_report(y_train, y_train_pred))

# Find misclassified samples
def find_misclassified_samples(model, X_test, y_test, documents):
    y_pred = model.predict(X_test)
    misclassified = []
    for i in range(len(y_test)):
        if y_test[i] != y_pred[i]:
            misclassified.append((documents[i][0], y_test[i], y_pred[i]))
    return misclassified

# Prepare Data and Train Model
labels = [label for _, label in documents]
features, vectorizer = extract_features(documents)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

# Check the distribution in training and testing datasets
y_train_labels = Counter(y_train)
y_test_labels = Counter(y_test)
print("Training Labels Distribution:", y_train_labels)
print("Testing Labels Distribution:", y_test_labels)

model = train_model(X_train, y_train)
evaluate_model(model, X_test, y_test)
evaluate_model_on_train_data(model, X_train, y_train)

misclassified_samples = find_misclassified_samples(model, X_test, y_test, documents)
print(f'Misclassified Samples: {misclassified_samples[:5]}')  # Print first 5 misclassified samples for review

# Sentiment Analysis Pipeline
def sentiment_analysis_pipeline(sentence):
    preprocessed_sentence = preprocess(sentence)
    features = vectorizer.transform([' '.join(preprocessed_sentence)])
    prediction = model.predict(features)
    return 'positive' if prediction[0] == 'pos' else 'negative'

# Test Sentiment Analysis
sentence = input("Enter: ")
sentiment = sentiment_analysis_pipeline(sentence)
print(f'Sentiment: {sentiment}')
