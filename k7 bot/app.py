from flask import Flask, render_template, request, jsonify
import spacy
import random

app = Flask(__name__)

# Load the trained chatbot model
nlp = spacy.load("model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.json['user_input']
    
    # Process user input with the chatbot model
    chatbot_response = generate_chatbot_response(user_input)

    return jsonify({"response": chatbot_response})

def generate_chatbot_response(user_input):
    doc = nlp(user_input)
    return random.choice(list(doc.sents)) if doc.sents else "I'm not sure how to respond to that."

if __name__ == '__main__':
    app.run(debug=True)
