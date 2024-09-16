import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from collections import Counter
import string

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to preprocess text
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())

    # Remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens

# Function to calculate Jaccard similarity
def jaccard_similarity(doc1, doc2):
    # Preprocess both documents
    tokens1 = preprocess_text(doc1)
    tokens2 = preprocess_text(doc2)

    # Calculate Jaccard similarity
    intersection = len(set(tokens1).intersection(tokens2))
    union = len(set(tokens1).union(tokens2))
    similarity = intersection / union

    return similarity

# Example usage
if __name__ == "__main__":
    # Two example documents
    document1 = "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language."
    document2 = "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language."

    # Calculate similarity
    similarity = jaccard_similarity(document1, document2)
    print("Jaccard similarity:", similarity)