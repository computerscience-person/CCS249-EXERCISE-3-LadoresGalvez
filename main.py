import wikipedia as wk
import nltk
import math
from collections import Counter
from nltk.util import bigrams, trigrams
from nltk.tokenize import word_tokenize

def fetch_wikipedia_text():
    # Retrieve Wikipedia article
    article = wk.page("Python (programming language)")
    tokens = word_tokenize(article.content.lower())  # Tokenize first

    # Remove punctuation tokens (keep only words)
    tokens = [word for word in tokens if word.isalnum()]

    # Limit to 1000 words
    return tokens[:1000]

def plain_text_head():
    return wk.page("Python (programming language)").content[:100]

def bigram_count(tokens):
    return Counter(bigrams(tokens))

def trigram_count(tokens):
    return Counter(trigrams(tokens))

def bigram_probabilities(tokens):
    bigram_counts = Counter(bigrams(tokens))
    unigram_counts = Counter(tokens)
    vocabulary_size = len(unigram_counts)

    # Apply Laplace smoothing
    bigram_probs = {bigram: (count + 1) / (unigram_counts[bigram[0]] + vocabulary_size)
                    for bigram, count in bigram_counts.items()}
    return bigram_probs

def trigram_probabilities(tokens):
    trigram_counts = Counter(trigrams(tokens))
    bigram_counts = Counter(bigrams(tokens))
    vocabulary_size = len(bigram_counts)

    # Apply Laplace smoothing
    trigram_probs = {trigram: (count + 1) / (bigram_counts[(trigram[0], trigram[1])] + vocabulary_size)
                     for trigram, count in trigram_counts.items() if (trigram[0], trigram[1]) in bigram_counts}
    return trigram_probs

def calculate_perplexity(model_probs, test_sentence, ngram_type):
    tokens = word_tokenize(test_sentence.lower())
    N = len(tokens)
    log_prob_sum = 0.0

    if ngram_type == 'bigram':
        for i in range(1, N):
            prob = model_probs.get((tokens[i-1], tokens[i]), 1e-6)  # Apply smoothing
            log_prob_sum += math.log(prob)
    elif ngram_type == 'trigram':
        for i in range(2, N):
            prob = model_probs.get((tokens[i-2], tokens[i-1], tokens[i]), 1e-6)  # Apply smoothing
            log_prob_sum += math.log(prob)

    perplexity = math.exp(-log_prob_sum / N)
    return perplexity

def main():
    tokens = fetch_wikipedia_text()

    # Train bigram and trigram models
    bigram_model = bigram_probabilities(tokens)
    trigram_model = trigram_probabilities(tokens)

    # Define test sentence
    test_sentence = "The quick brown fox jumps over the lazy dog near the bank of the river."

    # Compute perplexities
    bigram_perplexity = calculate_perplexity(bigram_model, test_sentence, 'bigram')
    trigram_perplexity = calculate_perplexity(trigram_model, test_sentence, 'trigram')

    print("Wikipedia Article: Python (programming language)")
    print(f"Raw Text: {plain_text_head()}")
    print(f"Number of Tokens: {len(tokens)}")
    print(f"Preview tokens: {tokens[:10]}")
    # print(f"Bigram counts: {bigram_count(tokens)}")
    # print(f"Trigram counts: {trigram_count(tokens)}")
    # print(f"Bigram Probabilities: {bigram_model}")
    # print(f"Trigram Probabilities: {trigram_model}")
    print(f"Test Sentence: {test_sentence}")
    print(f"Bigram Model Perplexity: {bigram_perplexity}")
    print(f"Trigram Model Perplexity: {trigram_perplexity}")

if __name__ == "__main__":
    main()
