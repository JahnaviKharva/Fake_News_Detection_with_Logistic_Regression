from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk

# Initialize Flask app
app = Flask(__name__)

# Download stopwords
nltk.download('stopwords')

# Load and preprocess dataset
news_dataset = pd.read_csv('train.csv')
news_dataset = news_dataset.fillna('')
news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']

# Initialize PorterStemmer
port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Apply stemming
news_dataset['content'] = news_dataset['content'].apply(stemming)

# Prepare data for model training
X = news_dataset['content'].values
Y = news_dataset['label'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Vectorize text data
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Flask route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    content = data.get('content', '')
    content = vectorizer.transform([content])
    prediction = model.predict(content)
    return jsonify({'prediction': 'real' if prediction[0] == 0 else 'fake'})

if __name__ == '__main__':
    app.run(debug=True)
