import os
import random

#Stopwords generated.
ENGLISH_STOPWORDS = [
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and",
    "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off",
    "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "s", "t", "can", "will", "just", "don", "should", "now"
]

def tokenize_corpus(data):
    tokenized_data = []
    for sentence in data:
        tokens = sentence.split()
        tokenized_data.append(tokens)
    return tokenized_data

def remove_special_characters(tokens):
    clean_tokens = []
    chars_to_remove = "!#$%&()*+/:,;.<=>@[\\]^`{|}~\t\n\""

    for token in tokens:
         clean_token = ''.join(character for character in token if character not in chars_to_remove)

         if clean_token:
            clean_tokens.append(clean_token)
    return clean_tokens

# print(clean_pos[4])
# print(len(tokenized_neg))
# print(len(clean_neg))

# Remove stopwords
def remove_stopwords(tokens):
    return [token for token in tokens if token.lower() not in ENGLISH_STOPWORDS]


# print(clean_neg[399999])
# print(clean_pos[399999])
# print(clean_neg_no_stopwords[399999])
# print(clean_pos_no_stopwords[399999])
# Data {sentences => {tokens - stopwords - removeSpecialCharacters} + {tokens - removeSpecialCharacters}}



# print(f"All labels = {len(all_no_stopwords_labels)}")

def split_data_to_trainTest(data, labels, test_size, random_state):

    random.seed(random_state)
    indices = list(range(len(data)))
    random.shuffle(indices)
    # print(f"Indices: {len(indices)}")
    # print(f"data_size: {len(data)}")
    # print(f"labels_size: {len(labels)}")

    #1,2,3,4,5,6,6,7
    #b1,b2,b3,b4,b
    #11100
    
    #Shuffling
    shuffled_data = [data[i] for i in indices]
    shuffled_labels = [labels[i] for i in indices]

    split_index = int(len(shuffled_data) * (1 - test_size))

    train_data = shuffled_data[:split_index]
    test_data = shuffled_data[split_index:]
    train_labels = shuffled_labels[:split_index]
    test_labels = shuffled_labels[split_index:]

    return train_data, train_labels, test_data, test_labels

#Outputting csv files
def convert_to_csv(list_to_convert, datadirectory, filename, filetype):

    
    if(filetype == "data"):
        with open(os.path.join(datadirectory, filename), 'w', newline='') as main_file:
         for item in list_to_convert:
            main_file.write(','.join(map(str,item)) + '\n')
    
    elif(filetype == "labels"):
        with open(os.path.join(datadirectory, filename), 'w', newline='') as main_file:
         for item in list_to_convert:
            main_file.write(','.join(str(item)) + '\n')
    return        

def main():
 """Implement your assignment solution here"""   
 args = __import__('sys').argv
 
 if len(args) != 2:
    print("Usage: python script.py filepath")
    return
    
 directory = args[1]
        
# Load positive and negative datasets
 with open(os.path.join(directory, 'pos.txt'), 'r', encoding='utf-8') as f:
    pos_data = f.readlines()

 with open(os.path.join(directory, 'neg.txt'), 'r', encoding='utf-8') as f:
    neg_data = f.readlines()

 tokenized_pos = tokenize_corpus(pos_data)
 tokenized_neg = tokenize_corpus(neg_data)
 clean_pos = [remove_special_characters(tokens) for tokens in tokenized_pos]
 clean_neg = [remove_special_characters(tokens) for tokens in tokenized_neg]
 clean_pos_no_stopwords = [remove_stopwords(tokens) for tokens in clean_pos]
 clean_neg_no_stopwords = [remove_stopwords(tokens) for tokens in clean_neg]
 
 # Concatenate positive and negative datasets
 all_no_stopwords = clean_pos_no_stopwords + clean_neg_no_stopwords
 all_no_stopwords_labels = [1] * len(clean_pos_no_stopwords) + [0] * len(clean_neg_no_stopwords)
 all_withstopwords = clean_pos + clean_neg
 all_withstopwords_labels =  [1] * len(clean_pos) + [0] * len(clean_neg)

# Split nostopwords data into training, validation, and test sets
 train_ns, train_labels_ns , test_ns, test_labels_ns = split_data_to_trainTest(all_no_stopwords, all_no_stopwords_labels , test_size=0.2, random_state=4)
 val_ns, val_labels_ns , test_ns, test_labels_ns = split_data_to_trainTest(test_ns, test_labels_ns , test_size=0.5, random_state=4)

# Split withstopwords data into training, validation, and test sets
 train, train_labels_s, test, test_labels_s = split_data_to_trainTest(all_withstopwords, all_withstopwords_labels , test_size=0.2, random_state=4)
 val, val_labels_s, test, test_labels_s = split_data_to_trainTest(test, test_labels_s , test_size=0.5, random_state=4)

 ##===============NO Stopwords Files=====================
 convert_to_csv(train_ns, directory, "train_ns.csv", "data")
 convert_to_csv(train_labels_ns, directory, "train_labels_ns.csv", "labels")
 convert_to_csv(test_ns, directory, "test_ns.csv", "data")
 convert_to_csv(test_labels_ns, directory, "test_labels_ns.csv", "labels")
 convert_to_csv(val_ns, directory, "val_ns.csv", "data")
 convert_to_csv(val_labels_ns, directory, "val_labels_ns.csv", "labels")
 convert_to_csv(all_no_stopwords, directory, "out_ns.csv", "data")
 convert_to_csv(all_no_stopwords_labels, directory, "out_ns_labels.csv", "labels")

 ##===============With Stopwords Files===================
 convert_to_csv(train, directory, "train.csv", "data")
 convert_to_csv(train_labels_s, directory, "train_labels_s.csv", "labels")
 convert_to_csv(test, directory, "test.csv", "data")
 convert_to_csv(test_labels_s, directory, "test_labels_s.csv", "labels")
 convert_to_csv(val, directory, "val.csv", "data")
 convert_to_csv(val_labels_s, directory, "val_labels_s.csv", "labels")
 convert_to_csv(all_withstopwords, directory, "out.csv", "data")
 convert_to_csv(all_withstopwords_labels, directory, "out_labels.csv", "labels")

 pass
    
if __name__ == "__main__":
    main()