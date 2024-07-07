import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Sample data
data = {
    'text': [
        'I love this movie!', 
        'This film was terrible...',
        'What a great experience!',
        'I hated every minute of it.',
        'Absolutely fantastic!',
        'Not my cup of tea.'
    ],
    'sentiment': [
        'positive', 
        'negative', 
        'positive', 
        'negative', 
        'positive', 
        'negative'
    ]
}

# Functional Programming approach
def load_data(data):
    return pd.DataFrame(data)

def split_data(df):
    return train_test_split(df['text'], df['sentiment'], test_size=0.3, random_state=42)

def create_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression())
    ])

def train_model(pipeline, X_train, y_train):
    return pipeline.fit(X_train, y_train)

def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

def main(data):
    df = load_data(data)
    X_train, X_test, y_train, y_test = split_data(df)
    pipeline = create_pipeline()
    model = train_model(pipeline, X_train, y_train)
    accuracy, report = evaluate_model(model, X_test, y_test)
    print(f'Accuracy: {accuracy}')
    print(f'Classification Report:\n{report}')

# Run the main function
main(data)
