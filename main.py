import wikipedia as wk
import nltk
from collections import Counter
from nltk.util import bigrams, trigrams

def main():
    # Retrieve Wikipedia article
    article = wk.page("Python (programming language)")
    tokzr = nltk.tokenize.RegexpTokenizer(r'\w+')
    article_toks = tokzr.tokenize(article.content.lower())
    print(len(article_toks))
    # Limit to 1000 words
    article_toks = article_toks[:1000]
    print(*(tok for tok in article_toks), sep = " ")

    # Generate bigram and trigram models
    def bigram_probabilities(tokens):
        bigram_counts = Counter(bigrams(tokens))
        unigram_counts = Counter(tokens)

        bigram_probs = {bigram: count / unigram_counts[bigram[0]]
                        for bigram, count in bigram_counts.items()}
        return bigram_probs

    def trigram_probabilities(tokens):
        trigram_counts = Counter(trigrams(tokens))
        bigram_counts = Counter(bigrams(tokens))

        trigram_probs = {trigram: count / bigram_counts[(trigram[0], trigram[1])]
                        for trigram, count in trigram_counts.items() if (trigram[0], trigram[1]) in bigram_counts}
        return trigram_probs

    # print(bigram_probabilities(article_toks))
    print(trigram_probabilities(article_toks))
def bigrams_model():
    ...

def trigram_model():
    ...

if __name__ == "__main__":
    main()
