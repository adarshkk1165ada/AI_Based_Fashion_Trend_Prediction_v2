from sklearn.feature_extraction.text import TfidfVectorizer


def extract_keywords(text_list, top_n=20):
    """
    Extract important keywords from fashion text using TF-IDF
    """

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=1000,
        ngram_range=(1, 2)
    )

    X = vectorizer.fit_transform(text_list)

    feature_names = vectorizer.get_feature_names_out()

    scores = X.sum(axis=0).A1

    keyword_scores = list(zip(feature_names, scores))

    keyword_scores = sorted(
        keyword_scores,
        key=lambda x: x[1],
        reverse=True
    )

    return keyword_scores[:top_n]