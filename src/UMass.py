from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from tqdm import tqdm

# Function to generate co-occurrence matrix
def create_co_occurrence_matrix(sentences, vocabulary):
    co_occurrence_matrix = defaultdict(int)
    vectorizer = CountVectorizer(vocabulary=vocabulary, binary=True)
    X = vectorizer.fit_transform(sentences)

    for doc in tqdm(X, desc='Creating co-occurrence matrix'):
        words = doc.nonzero()[1]
        for i, word in enumerate(words):
            for j in words[i + 1:]:
                co_occurrence_matrix[(word, j)] += 1

    return co_occurrence_matrix

# Function to calculate UMass coherence score
def calculate_umass_coherence(topics, sentences):
    vectorizer = CountVectorizer()
    sentences_matrix = vectorizer.fit_transform(sentences)
    vocab = vectorizer.get_feature_names_out()

    co_occurrence_matrix = create_co_occurrence_matrix(sentences, vocab)

    coherence = 0
    # print(type(topics))
    for topic in tqdm(topics, desc='Calculating UMass coherence score'):
        
        words = topic.split(',')
        # print(words)
        word_indices = [np.where(vocab == word)[0][0] for word in words if word in vocab]

        if len(word_indices) < 2:
            continue

        total_pairs = 0
        total_score = 0
        for i, word_i in enumerate(word_indices):
            for j, word_j in enumerate(word_indices):
                if i != j:
                    pair_count = co_occurrence_matrix[(word_i, word_j)]
                    total_score += np.log((pair_count + 1) / sentences_matrix[:, word_i].sum())
                    total_pairs += 1

        coherence += total_score / total_pairs if total_pairs > 0 else 0

    coherence /= len(topics)
    return coherence


# Example usage
corpus_filename = '../datasets/yelp/texts.txt'  # Replace with your corpus file
# topics_filename = '../results_yelp/topics.txt'
topics_filename = 'topics_final.txt'

with open(corpus_filename, 'r') as file:
    sentences = [line.strip().split('\t')[1] for line in file.readlines()]

topics = []
with open(topics_filename, 'r') as file:
    # topics = [line.strip() for line in file.readlines()]
    for line in file.readlines():
        topics.append(line.strip().split(' ')[2])

print(topics)

# topics = ['game strategy skill', 'life challenges opportunities', 'learning skills growth']
umass_score = calculate_umass_coherence(topics, sentences)
print(f"UMass coherence score: {umass_score}")