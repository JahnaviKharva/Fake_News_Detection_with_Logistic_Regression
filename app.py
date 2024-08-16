from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
import requests
from io import StringIO

# Initialize Flask app
app = Flask(__name__)

# Download stopwords
nltk.download('stopwords')

# URL of your CSV file
csv_url = 'https://docs.google.com/spreadsheets/d/10Cq70VX6PZHo7Plawxok1Hr33cbkG-s3qSFewjV1jCI/edit?pli=1&gid=961256751#gid=961256751'

# Load the CSV file from the URL
response = requests.get(csv_url)
csv_data = StringIO(response.text)
news_dataset = pd.read_csv(csv_data)

# Replace missing values
news_dataset = news_dataset.fillna('')

# Merge the author and title into a single feature
news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']

# Initialize PorterStemmer
port_stem = PorterStemmer()

# Function to apply stemming
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Apply stemming to the content
news_dataset['content'] = news_dataset['content'].apply(stemming)

# Separate the data and the label
X = news_dataset['content'].values
Y = news_dataset['label'].values

# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Vectorize text data
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Initialize and train the model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Flask route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    content = data.get('content', '')
    content = vectorizer.transform([content])
    prediction = model.predict(content)
    result = 'real' if prediction[0] == 0 else 'fake'
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

