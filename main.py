import wikipedia as wk
import nltk

def main():
    article = wk.page("Text_processing")
    article_plain = article.content
    article_toks = nltk.word_tokenize(article.content)
    print(article_toks)
    print(len(article_toks))


if __name__ == "__main__":
    main()
