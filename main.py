from typing import cast
import asyncio
import wikipedia as wk
import nltk
from collections import Counter
from nltk.util import bigrams, trigrams
from pprint import pprint

async def main():
    await create_and_run_models()

async def create_and_run_models():
    # Retrieve Wikipedia article
    article = wk.page("Python (programming language)")
    # Use regular expression tokenizer
    tokzr = nltk.tokenize.RegexpTokenizer(r"\b(?:\w+(?:\.\w+)*|\d+\.\d+)\b")
    article_toks = cast(list[str], tokzr.tokenize(article.content.lower()))
    # Limit to 1000 words
    article_toks = article_toks[:1000]
    # Generate bigram and trigram models
    pprint(bigram_probabilities(article_toks))
    pprint(trigram_probabilities(article_toks))

def bigram_probabilities(tokens: list[str]) -> dict[tuple[str, str], float]:
    bigram_counts = Counter(bigrams(tokens))
    unigram_counts = Counter(tokens)

    bigram_probs = {
        bigram: count / unigram_counts[bigram[0]]
        for bigram, count in bigram_counts.items()
    }
    return bigram_probs


def trigram_probabilities(tokens: list[str]) -> dict[tuple[str, str, str,], float]:
    trigram_counts = Counter(trigrams(tokens))
    bigram_counts = Counter(bigrams(tokens))

    trigram_probs = {
        trigram: count / bigram_counts[(trigram[0], trigram[1])]
        for trigram, count in trigram_counts.items()
        if (trigram[0], trigram[1]) in bigram_counts
    }
    return trigram_probs


if __name__ == "__main__":
    asyncio.run(main())
