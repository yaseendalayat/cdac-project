# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, render_template, request
import pickle  # Changed from joblib to pickle
import os  # Import the os module
try:
    import nltk
except ModuleNotFoundError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])
    import nltk
from nltk.stem import WordNetLemmatizer
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ModuleNotFoundError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    from sklearn.feature_extraction.text import TfidfVectorizer
try:
    import pandas as pd
except ModuleNotFoundError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
    import pandas as pd
import random

# Flask constructor takes the name of 
# current module (__name__) as argument.
app = Flask(__name__)

# Initialize lemmatizer and vectorizer
lemmatizer = WordNetLemmatizer()
vectorizer = TfidfVectorizer()

# Construct the absolute path to the model file
model_path = os.path.join(os.path.dirname(__file__), 'svm_model.pkl')
print(f"Model path: {model_path}")  # Print the model path

# Load the SVM model
try:
    # Check if the file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Ensure this loading method is compatible with pickle
    with open(model_path, 'rb') as f:  # Open the file in binary read mode
        model = pickle.load(f)  # Load the model using pickle
    print("Model loaded successfully.")  # Print success message
except FileNotFoundError as e:
    print(f"Error: {e}")
    model = None
except Exception as e:
    print(f"Error loading the model: {e}")
    model = None  # Handle the case where the model fails to load

# Construct the absolute path to the CSV data file
csv_path = os.path.join(os.path.dirname(__file__), 'Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv')
print(f"CSV path: {csv_path}")  # Print the CSV path

# Load the data
try:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    data = pd.read_csv(csv_path)
    data['instruction_lemmatized'], _ = zip(*data['instruction'].apply(lambda x: (' '.join([lemmatizer.lemmatize(w) for w in nltk.word_tokenize(x)]) if isinstance(x, str) else '', '')))
    data['response_lemmatized'], _ = zip(*data['response'].apply(lambda x: (' '.join([lemmatizer.lemmatize(w) for w in nltk.word_tokenize(x)]) if isinstance(x, str) else '', '')))
    all_text = data['instruction_lemmatized'] + ' ' + data['response_lemmatized']
    vectorizer.fit(all_text)
except FileNotFoundError as e:
    print(f"Error: {e}")
    data = None
except Exception as e:
    print(f"Error loading the CSV data: {e}")
    data = None

# Preprocessing function
def preprocess_text(text):
    words = nltk.word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(lemmatized_words)

# The route() function of the Flask class is a decorator, 
# which tells the application which URL should call 
# the associated function.
@app.route('/')
# ‘/’ URL is bound with index() function.
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_message = request.form['user_message']
    if model and data is not None:
        # Preprocess the user input
        preprocessed_message = preprocess_text(user_message)

        # Transform the preprocessed input using the trained vectorizer
        input_tfidf = vectorizer.transform([preprocessed_message])

        # Predict the intent using the loaded SVM model
        predicted_intent = model.predict(input_tfidf)[0]

        # Find a response from your dataset with the predicted intent
        possible_responses = data[data['intent'] == predicted_intent]['response'].tolist()

        # Select a response randomly to avoid repetition
        if possible_responses:  # Check if there are any responses for the intent
            predicted_response = random.choice(possible_responses)
        else:
            predicted_response = "I'm sorry, I don't understand your request."  # Default response if no matching intent found
    elif data is None:
        predicted_response = "Data file not loaded."
    else:
        predicted_response = "Model not loaded."
    return render_template('index.html', user_message=user_message, chatbot_response=predicted_response)

# main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application 
    # on the local development server.
    app.run(debug=True)