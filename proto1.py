positive_words = ['love', 'happy', 'joy', 'excellent', 'good', 'great', 'amazing', 'wonderful', 'positive', 'best']
negative_words = ['hate', 'sad', 'bad', 'terrible', 'awful', 'worst', 'negative', 'horrible', 'poor']

def analyze_sentiment(text: str) -> dict:
    sent = text.lower().split()
    positive_score = sum(1 for word in sent if word in positive_words)
    negative_score = sum(1 for word in sent if word in negative_words)
    return {
        'positive': positive_score,
        'negative': negative_score,
        'compound': positive_score - negative_score
    }

# Function to categorize sentiment
def categorize_sentiment(scores: dict) -> str:
    if scores['compound'] > 0:
        return 'Positive'
    elif scores['compound'] < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Function to get sentiment
def get_sentiment(text: str) -> str:
    return categorize_sentiment(analyze_sentiment(text))

# Main function to test the program
if __name__ == "__main__":
    text = "I love functional programing"
    print(f"Text: {text}")
    sentiment = get_sentiment(text)
    print(f"Sentiment Analysis results: {sentiment}")
