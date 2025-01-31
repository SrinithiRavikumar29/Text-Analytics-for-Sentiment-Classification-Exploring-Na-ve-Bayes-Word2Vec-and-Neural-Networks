import os
import re
import sys
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

#Load the data
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    return [line.strip() for line in data]

def load_labels(file_path):
    with open(file_path, 'r') as file:
        labels = file.readlines()
    return [line.strip() for line in labels]

# Define the directory where the data files are located
data_dir = 'c:/Users/srini/OneDrive/Desktop/SPRING_24/MSCI641/ASSIGNMENTS/A1/data/'  

# Paths to data files
train_datafiles = {
    'all_withstopwords': (os.path.join(data_dir, 'train.csv'), os.path.join(data_dir, 'train_labels_s.csv')),
    'all_no_stopwords': (os.path.join(data_dir, 'train_ns.csv'), os.path.join(data_dir, 'train_labels_ns.csv'))
}

val_datafiles = {
    'all_withstopwords': (os.path.join(data_dir, 'val.csv'), os.path.join(data_dir, 'val_labels_s.csv')),
    'all_no_stopwords': (os.path.join(data_dir, 'val_ns.csv'), os.path.join(data_dir, 'val_labels_ns.csv'))
}

def main():
    
    # Configurations for the different models
    configuration = [
         ('mnb_uni_ns', CountVectorizer(ngram_range=(1, 1)),'all_no_stopwords'),
         ('mnb_bi_ns', CountVectorizer(ngram_range=(2, 2)),'all_no_stopwords'),
         ('mnb_uni_bi_ns', CountVectorizer(ngram_range=(1, 2)),'all_no_stopwords'),
         ('mnb_uni', CountVectorizer(ngram_range=(1, 1)),'all_withstopwords'),
         ('mnb_bi', CountVectorizer(ngram_range=(2, 2)),'all_withstopwords'),
         ('mnb_uni_bi', CountVectorizer(ngram_range=(1, 2)),'all_withstopwords')
   
     ]

    results = []

    for configuration_name, vectorizer, stopword_stat in configuration:
    
      # Load training and validation data
      train_data, train_labels = load_data(train_datafiles[stopword_stat][0]), load_labels(train_datafiles[stopword_stat][1])
      val_data, val_labels = load_data(val_datafiles[stopword_stat][0]), load_labels(val_datafiles[stopword_stat][1])

      # Print lengths of loaded data
      #print(f"Config: {configuration_name}")
      #print(f"Training data: {len(train_data)} samples")
      #print(f"Training labels: {len(train_labels)} samples")
      #print(f"Validation data: {len(val_data)} samples")
      #print(f"Validation labels: {len(val_labels)} samples")
    
      # Create and train the model
      model = make_pipeline(vectorizer, MultinomialNB())
      model.fit(train_data, train_labels)
    
      # Validate the model
      val_predictions = model.predict(val_data)
      val_accuracy = accuracy_score(val_labels, val_predictions)
    
      # Save the model
      model_filename = f"{configuration_name}.pkl"
      with open(model_filename, 'wb') as file:
        pickle.dump(model, file)

      # Determine the text feature label
      if 'uni' in configuration_name and 'bi' in configuration_name:
        text_feature = 'unigrams + bigrams'
      elif 'uni' in configuration_name:
        text_feature = 'unigrams'
      else:
        text_feature = 'bigrams'
     
      # Record results
      results.append((stopword_stat, text_feature, val_accuracy))

    # Print results
    print("Stopwords Removed, Text Features, Accuracy")
    for stopword_stat, text_feature, accuracy in results:
        stopword_stat_str = 'yes' if stopword_stat == 'all_withstopwords' else 'no'
        print(f"{stopword_stat_str}, {text_feature}, {accuracy:.4f}")
        
 
pass

if __name__ == "__main__":
    main()