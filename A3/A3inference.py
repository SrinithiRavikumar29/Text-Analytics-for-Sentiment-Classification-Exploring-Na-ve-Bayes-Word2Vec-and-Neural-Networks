import sys
import gensim

def load_model(model_path):
    try:
        model = gensim.models.Word2Vec.load(model_path)
        return model
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        sys.exit(1)

def load_words(word_file):
    try:
        with open(word_file, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f.readlines()]
        return words
    except FileNotFoundError:
        print(f"Error: Word file '{word_file}' not found.")
        sys.exit(1)

def get_similar_words(word, model, topn=20):
    try:
        similar_words = model.wv.most_similar(word, topn=topn)
        return [similar_word[0] for similar_word in similar_words]  # Extracting only the words
    except KeyError:
        return f"Word '{word}' not in vocabulary!"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inference.py <word_file>")
        sys.exit(1)

    word_file = sys.argv[1]
    model_path = 'c:/Users/srini/OneDrive/Desktop/SPRING_24/MSCI641/ASSIGNMENTS/A3/data/w2v.model'  # Path to your trained Word2Vec model

    model = load_model(model_path)
    words = load_words(word_file)

    for word in words:
        similar_words = get_similar_words(word, model)
        print(f"\nTop-20 most similar words to '{word}':")
        print(similar_words)
