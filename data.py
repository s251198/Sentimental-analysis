from textblob import TextBlob
import nltk
from newspaper import Article

url = "https://www.history.com/topics/21st-century/9-11-timeline"
article = Article(url) #gets the article from the url

article.download() #dowmload the article
article.parse() #parse the article
nltk.download('punkt') #downloading the punkt tokenizer model
article.nlp() #performs nlp tasks

text = article.summary #summarises the text
obj = TextBlob(text) #creates a textblob object
sentiment = obj.sentiment.polarity #retures a object between 1 and -1

if sentiment == 0:
    print("the text is neutral")
elif sentiment > 0:
    print("the text is positive")
else:
    print("the text is negative")