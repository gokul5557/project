import spacy
import random
from spacy.training.example import Example
import pickle  # Import the pickle module

# Load a spaCy language model (you can use a pre-trained model or train from scratch)
nlp = spacy.blank("en")

# Define a function to read training data from a file
def read_training_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    # Split lines into question-answer pairs
    train_data = [line.strip().split('\t') for line in lines]
    return train_data

# Define your training data file path
data_file_path = 'dialogs.txt'

# Read training data from the file
train_data = read_training_data(data_file_path)

# Define a custom function for training the chatbot
def train_chatbot(train_data):
    # Create a blank Entity Recognizer component
    ner = nlp.add_pipe("ner")

    # Define labels (only one label for the entire conversation)
    labels = ["CONVERSATION"]

    # Add labels to the NER component
    for label in labels:
        ner.add_label(label)

    # Train the NER component
    optimizer = nlp.begin_training()
    for epoch in range(20):  # You can adjust the number of training iterations
        random.shuffle(train_data)
        losses = {}
        for text, _ in train_data:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, {"entities": []})
            nlp.update([example], drop=0.5, losses=losses)

        # Print the loss for this epoch
        print(f"Epoch {epoch + 1} - Loss: {losses['ner']}")

    # Save the trained model as a .pkl file
    with open("chatbot_model.pkl", "wb") as pkl_file:
        pickle.dump(nlp, pkl_file)

# Train the chatbot model
train_chatbot(train_data)
