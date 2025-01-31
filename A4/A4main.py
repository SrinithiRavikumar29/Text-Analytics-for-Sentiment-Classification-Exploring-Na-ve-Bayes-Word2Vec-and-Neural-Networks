import re
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gensim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

w2v_model_path = "C:/Users/srini/OneDrive/Desktop/MASTERS/SPRING_24/MSCI641/ASSIGNMENTS/A3/data/w2v.model"

# Defining the feed forward neural network
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

# Load data
def load_data(data_path, label_path):
    with open(data_path, 'r') as f:
        texts = f.readlines()
    with open(label_path, 'r') as f:
        labels = f.readlines()
    texts = [text.strip() for text in texts]
    labels = [label.strip() for label in labels]
    return texts, labels

# Converting texts to embeddings
def texts_to_embeddings(texts, w2v_model, embedding_dim):
    embeddings = []
    for text in texts:
        words = re.findall(r'\w+', text.lower())
        word_vectors = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
        if word_vectors:
            embeddings.append(np.mean(word_vectors, axis=0))
        else:
            embeddings.append(np.zeros(embedding_dim))
    return np.array(embeddings)

# Preparing data
def prepare_data(data_path, label_path, w2v_model, embedding_dim):
    texts, labels = load_data(data_path, label_path)
    embeddings = texts_to_embeddings(texts, w2v_model, embedding_dim)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    return embeddings, labels

# Training the model
def train_model(train_loader, val_loader, input_size, hidden_size, output_size, activation_fn, dropout_rate, learning_rate, weight_decay, num_epochs):
    model = FFNN(input_size, hidden_size, output_size, activation_fn, dropout_rate)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.float(), targets.long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Validating the model
        model.eval()
        val_loss = 0.0
        val_targets = []
        val_outputs = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.float(), targets.long()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_targets.extend(targets.numpy())
                val_outputs.extend(outputs.argmax(dim=1).numpy())

        val_loss /= len(val_loader)
        val_accuracy = accuracy_score(val_targets, val_outputs)
        print(f'Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    return model

# Main function
def main(data_dir):
    # Load word2vec model
    w2v_model = gensim.models.Word2Vec.load(w2v_model_path)
    embedding_dim = w2v_model.vector_size

    # Hyperparameters
    hidden_size = 5000
    output_size = 2
    dropout_rates = [0.3]
    learning_rate = 0.0001
    weight_decay = 1e-5
    num_epochs = 20
    batch_size = 64

    # Preparing data
    train_data, train_labels = prepare_data(os.path.join(data_dir, 'train.csv'), os.path.join(data_dir, 'train_labels_s.csv'), w2v_model, embedding_dim)
    val_data, val_labels = prepare_data(os.path.join(data_dir, 'val.csv'), os.path.join(data_dir, 'val_labels_s.csv'), w2v_model, embedding_dim)
    test_data, test_labels = prepare_data(os.path.join(data_dir, 'test.csv'), os.path.join(data_dir, 'test_labels_s.csv'), w2v_model, embedding_dim)

    train_dataset = TensorDataset(torch.tensor(train_data), torch.tensor(train_labels))
    val_dataset = TensorDataset(torch.tensor(val_data), torch.tensor(val_labels))
    test_dataset = TensorDataset(torch.tensor(test_data), torch.tensor(test_labels))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Training and saving models with different activation functions
    activation_functions = {
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh()
    }

    results = []

    for name, activation_fn in activation_functions.items():
        for dropout_rate in dropout_rates:
            print(f'Training model with {name} activation and {dropout_rate} dropout rate...')
            model = train_model(train_loader, val_loader, embedding_dim, hidden_size, output_size, activation_fn, dropout_rate, learning_rate, weight_decay, num_epochs)
            model_path = f'nn_{name}_{dropout_rate}.model'
            torch.save(model.state_dict(), model_path)

            # Evaluating on the test set
            model.eval()
            test_targets = []
            test_outputs = []
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.float(), targets.long()
                    outputs = model(inputs)
                    test_targets.extend(targets.numpy())
                    test_outputs.extend(outputs.argmax(dim=1).numpy())

            test_accuracy = accuracy_score(test_targets, test_outputs)
            results.append((name, dropout_rate, test_accuracy))
            print(f'Test Accuracy with {name} activation and {dropout_rate} dropout rate: {test_accuracy:.4f}')

    # Printing results
    print("\nResults:")
    print("Activation Function | Dropout Rate | Test Accuracy")
    for name, dropout_rate, test_accuracy in results:
        print(f"{name:<18} | {dropout_rate:<12} | {test_accuracy:.4f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_data_directory>")
        sys.exit(1)
    data_dir = sys.argv[1]
    main(data_dir)
