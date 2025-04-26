import re
import numpy as np
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

reviews = [
    "I love this product! It's amazing and works perfectly.",
    "Worst experience ever. The product broke after one use.",
    "The quality is decent, but it could be better.",
    "Absolutely fantastic! Highly recommend it.",
    "Not worth the money. I regret buying it."
]

for i, review in enumerate(reviews):
    print(f"Review {i+1}: {review}")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(reviews):
    processed_reviews = []
    tokenized_reviews = []
    
    for i, review in enumerate(reviews):
        # 1. Tokenization and lowercasing
        tokens = word_tokenize(review.lower())
        print(tokens)
        
        # 2. Punctuation removal
        tokens = [token for token in tokens if re.match(r'[a-zA-Z]+', token)]
        print(tokens)
        
        # 3. Stopword removal
        tokens = [token for token in tokens if token not in stop_words]
        print(tokens)
        
        # 4. Lemmatization
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        print(tokens)
        
        processed_reviews.append(' '.join(tokens))
        tokenized_reviews.append(tokens)
    
    return processed_reviews, tokenized_reviews

processed_reviews, tokenized_reviews = preprocess_text(reviews)


# 1. Bag of Words
print("\n1. Bag of Words Representation:")
bow_vectorizer = CountVectorizer()
bow_matrix = bow_vectorizer.fit_transform(processed_reviews)
bow_features = bow_vectorizer.get_feature_names_out()

print("Features:", bow_features)
print("BoW Matrix:")
print(bow_matrix.toarray())

# 2. TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(processed_reviews)
tfidf_features = tfidf_vectorizer.get_feature_names_out()

print("Features:", tfidf_features)
print("TF-IDF Matrix:")
print(np.round(tfidf_matrix.toarray(), 3))  # Rounded for readability

# 3. Word Frequency (Occurrence Matrix)
for i, tokens in enumerate(tokenized_reviews):
    word_freq = Counter(tokens)
    print(f"Review {i+1} word frequencies:", dict(word_freq))

# 4. Co-occurrence Matrix
def create_cooccurrence_matrix(documents, window_size=2):
    vocab = sorted(set(word for doc in documents for word in doc))
    vocab_to_idx = {word: i for i, word in enumerate(vocab)}
    
    cooccurrence_matrix = np.zeros((len(vocab), len(vocab)))
    
    for document in documents:
        for i, word in enumerate(document):
            for j in range(max(0, i-window_size), min(len(document), i+window_size+1)):
                if i != j:  # Avoid self-co-occurrence
                    cooccurrence_matrix[vocab_to_idx[word], vocab_to_idx[document[j]]] += 1
    
    return cooccurrence_matrix, vocab

cooccurrence_matrix, vocab = create_cooccurrence_matrix(tokenized_reviews)

print("vocab: ", vocab)
print(cooccurrence_matrix)

for i, word1 in enumerate(vocab):
    for j, word2 in enumerate(vocab):
        if cooccurrence_matrix[i, j] > 0:
            print(f"'{word1}' co-occurs with '{word2}' {int(cooccurrence_matrix[i, j])} times")