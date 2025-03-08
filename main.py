import wikipedia as wk
import nltk

def main():
    # Retrieve Wikipedia article
    article = wk.page("Python (programming language)")
    article_toks = nltk.word_tokenize(article.content.lower())  # Tokenize first

    # Remove punctuation tokens
    words_only = [word for word in article_toks if word.isalnum()]  # Keep only alphanumeric words

    # Limit to 1000 words
    words_only = words_only[:1000]

if __name__ == "__main__":
    main()
