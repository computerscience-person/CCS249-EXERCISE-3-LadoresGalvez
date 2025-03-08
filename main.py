import wikipedia as wk
import nltk

def main():
    # Retrieve Wikipedia article
    article = wk.page("Python (programming language)")
    article_toks = nltk.word_tokenize(article.content.lower())  # Tokenize first

    # Remove punctuation tokens
    article_toks = [word for word in article_toks if word.isalnum()]  # Keep only alphanumeric words

    # Limit to 1000 words
    article_toks = article_toks[:1000]

if __name__ == "__main__":
    main()
