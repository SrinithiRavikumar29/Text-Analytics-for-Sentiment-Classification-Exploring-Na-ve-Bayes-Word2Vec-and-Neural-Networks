import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import gensim
from sklearn.preprocessing import LabelEncoder

# Define the neural network
class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation_fn, dropout_rate):
        super(FFNN, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.activation = activation_fn
        self.dropout = nn.Dropout(dropout_rate)
        self.output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.output(x)
        x = self.softmax(x)
        return x

# Function to convert texts to embeddings
def texts_to_embeddings(texts, w2v_model, embedding_dim):
    embeddings = []
    for text in texts:
        words = text.split()
        word_vectors = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
        if word_vectors:
            embeddings.append(np.mean(word_vectors, axis=0))
        else:
            embeddings.append(np.zeros(embedding_dim))
    return np.array(embeddings)

# Function to load data from text file
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = f.readlines()
    sentences = [sentence.strip() for sentence in sentences]
    return sentences

# Function to classify sentences
def classify_sentences(sentences, w2v_model, activation_fn, dropout_rate, model_path):
    # Load pre-trained model
    model = FFNN(w2v_model.vector_size, 128, 2, activation_fn, dropout_rate)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Convert sentences to embeddings
    embeddings = texts_to_embeddings(sentences, w2v_model, w2v_model.vector_size)
    inputs = torch.tensor(embeddings).float()

    # Perform inference
    with torch.no_grad():
        outputs = model(inputs)
        predictions = outputs.argmax(dim=1).numpy()

    return predictions

# Main function
def main(file_path, activation_type):
    # Load word2vec model
    w2v_model_path = "C:/Users/srini/OneDrive/Desktop/MASTERS/SPRING_24/MSCI641/ASSIGNMENTS/A3/data/w2v.model"
    w2v_model = gensim.models.Word2Vec.load(w2v_model_path)

    # Determine activation function and dropout rate
    activation_functions = {
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh()
    }
    activation_fn = activation_functions.get(activation_type)
    dropout_rate = 0.5  # Adjust as needed based on your model training

    # Load sentences from file
    sentences = load_data(file_path)

    # Perform classification
    model_path = f'nn_{activation_type}_{dropout_rate}.model'
    predictions = classify_sentences(sentences, w2v_model, activation_fn, dropout_rate, model_path)

    # Print results
    print("Sentence | Predicted Class")
    print("--------------------------")
    for sentence, prediction in zip(sentences, predictions):
        if prediction == 0:
            print(f"{sentence} | Negative")
        else:
            print(f"{sentence} | Positive")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python inference.py <path_to_sentences_file> <activation_type>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    activation_type = sys.argv[2]
    
    main(file_path, activation_type)
