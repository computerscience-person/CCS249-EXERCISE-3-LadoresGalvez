from functools import reduce
from math import exp, log
from typing import Callable, cast
import wikipedia as wk
import nltk
from collections import Counter
from nltk.util import bigrams, trigrams
from pprint import pprint


def main():
    # Retrieve Wikipedia article
    article = wk.page("Python (programming language)")
    # Use regular expression tokenizer
    tokzr = nltk.tokenize.RegexpTokenizer(r"\b(?:\w+(?:\.\w+)*|\d+\.\d+)\b")
    article_toks = cast(list[str], tokzr.tokenize(article.content.lower()))
    # Limit to 1000 words
    article_toks = article_toks[:1000]
    bigram_model = bigram_probabilities(article_toks)
    trigram_model = trigram_probabilities(article_toks)
    prompt_model(bigram_model, trigram_model)
    # test_phrase = "Python is a popular programming language, used in web development, machine learning, and data science."
    prompt_perplexity(bigram_model, trigram_model, tokzr.tokenize)


models = dict[tuple[str, str], float] | dict[tuple[str, str, str], float]


def prompt_model(bigram_model, trigram_model):
    user_select_model = input(
        "show model probabilities of [(b)igrams|(t)rigrams|(n)o]: "
    )
    match user_select_model:
        case "b" | "bigrams":
            pprint(bigram_model)
        case "t" | "trigrams":
            pprint(trigram_model)
        case "n" | "no":
            pass
        case _:
            print("bye!")
            exit(0)


def prompt_perplexity(bigram_model, trigram_model, tokenizer):
    user_select_model = input("use [(b)igrams|(t)rigrams|(n)o] model: ")
    user_test_text = input("enter test string: ")
    match user_select_model:
        case "b" | "bigrams":
            print("testing: ", user_test_text)
            print(
                "perplexity score: ",
                perplexity_score(bigram_model, "bigram", tokenizer, user_test_text),
            )
        case "t" | "trigrams":
            print("testing: ", user_test_text)
            print(
                "perplexity score: ",
                perplexity_score(trigram_model, "trigram", tokenizer, user_test_text),
            )
        case "n" | "no":
            pass
        case _:
            print("bye!")
            exit(0)


def perplexity_score(
    model: models, model_type: str, tokenizer: Callable[[str], list[str]], test: str
) -> float:
    """get the perplexity score of a sequence of tokens

    parameters:
    model: the model to be used
    tokenizer: the tokenizer to be used
    test: the sequence of words to be tested

    Returns:
    The perplexity score of the test sequence
    """
    toks = cast(list[str], tokenizer(test.lower()))
    ngrams_counts = (
        list(bigrams(toks))
        if model_type == "bigram"
        else list(trigrams(toks))
        if model_type == "trigram"
        else Counter()
    )
    probs = [model.get(ngram, 1e-9) for ngram in ngrams_counts]
    pprint(toks)
    pprint(ngrams_counts)
    pprint(probs)
    perp = exp(reduce(lambda x, y: abs(log(x) + log(y)), probs) / len(ngrams_counts))
    return perp


def bigram_probabilities(tokens: list[str]) -> dict[tuple[str, str], float]:
    bigram_counts = Counter(bigrams(tokens))
    unigram_counts = Counter(tokens)

    bigram_probs = {
        bigram: count / unigram_counts[bigram[0]]
        for bigram, count in bigram_counts.items()
    }
    return bigram_probs


def trigram_probabilities(
    tokens: list[str],
) -> dict[
    tuple[
        str,
        str,
        str,
    ],
    float,
]:
    trigram_counts = Counter(trigrams(tokens))
    bigram_counts = Counter(bigrams(tokens))

    trigram_probs = {
        trigram: count / bigram_counts[(trigram[0], trigram[1])]
        for trigram, count in trigram_counts.items()
        if (trigram[0], trigram[1]) in bigram_counts
    }
    return trigram_probs


if __name__ == "__main__":
    main()
