import os
import sys
import pickle

def load_model(test_model_path):
    with open(test_model_path, 'rb') as file:
        return pickle.load(file)

def classify_sentences(test_model, sentence):
    return test_model.predict(sentence)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python inference.py <path_to_txt_file> <classifier_type>")
        sys.exit(1)

    txt_file_path = sys.argv[1]
    classifier_type = sys.argv[2]
    
    test_model_path = f"{classifier_type}.pkl"
    
    if not os.path.exists(test_model_path):
        print(f"Model file {test_model_path} does not exist.")
        sys.exit(1)
    
    # Load the model
    test_model = load_model(test_model_path)
    
    # Load sentences from the file
    with open(txt_file_path, 'r') as file:
        sentence = [line.strip() for line in file]
    
    # Classify the sentences
    prediction = classify_sentences(test_model, sentence)


    final_labels = []
    for i in range(0, len(prediction)):
        if(prediction[i] == "1"): final_labels.append("positive")
        elif(prediction[i] == "0"): final_labels.append("negative")
    # Output predictions
    for (sent, final_label) in zip(sentence, final_labels):
        print(f"Sentence: {sent}\nPrediction: {final_label}\n")
