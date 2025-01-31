import gensim
from gensim.models import Word2Vec
import re

def preprocess_text(text):
    # Define the set of special characters to be removed
    special_chars = r'!#$%&()*+/:,;.<=>@[\\\]^`{|}~\t\n"'
    # Remove special characters and convert to lowercase
    text = re.sub(f'[{special_chars}]', '', text)
    text = text.lower()
    # Tokenize the text
    tokens = text.split()
    return tokens

def train_word2vec(pos_file, neg_file, output_model):
    sentences = []
    
    # Read positive reviews, preprocess, and tokenize
    with open(pos_file, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = preprocess_text(line)
            sentences.append(tokens)

    # Read negative reviews, preprocess, and tokenize
    with open(neg_file, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = preprocess_text(line)
            sentences.append(tokens)

    # Train Word2Vec model
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, workers=4)
    model.save(output_model)
    print("Word2Vec model trained and saved successfully!")

def get_similar_words(word, model, topn=20):
    try:
        similar_words = model.wv.most_similar(word, topn=topn)
        similar_words = [word for word, _ in similar_words]  # Extract only the words
        return similar_words
    except KeyError:
        return f"Word '{word}' not in vocabulary!"


if __name__ == "__main__":
    # File paths and output model name
    pos_file = 'c:/Users/srini/OneDrive/Desktop/SPRING_24/MSCI641/ASSIGNMENTS/A1/data/pos.txt'
    neg_file = 'c:/Users/srini/OneDrive/Desktop/SPRING_24/MSCI641/ASSIGNMENTS/A1/data/neg.txt'
    output_model = 'c:/Users/srini/OneDrive/Desktop/SPRING_24/MSCI641/ASSIGNMENTS/A3/data/w2v.model'

    train_word2vec(pos_file, neg_file, output_model)

    model = Word2Vec.load(output_model)

    print("\n20 most similar words to 'good':")
    print(get_similar_words('good', model))

    print("\n20 most similar words to 'bad':")
    print(get_similar_words('bad', model))



