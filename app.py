from flask import Flask, request, render_template, jsonify
import joblib

app = Flask(__name__)

# Load the saved model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the text input from the form
        news_text = request.form['news_text']
        
        # Transform the text and predict
        transformed_text = vectorizer.transform([news_text])
        prediction = model.predict(transformed_text)[0]
        
        # Render the result.html template with the prediction result
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
