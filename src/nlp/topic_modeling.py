from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def run_lda(text_list, n_topics=5, n_words=8):
    """
    Perform topic modeling on fashion text
    """

    vectorizer = CountVectorizer(
        stop_words="english",
        max_df=0.9,
        min_df=2
    )

    X = vectorizer.fit_transform(text_list)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42
    )

    lda.fit(X)

    words = vectorizer.get_feature_names_out()

    topics = []

    for topic_id, topic in enumerate(lda.components_):

        top_words = [
            words[i] for i in topic.argsort()[-n_words:]
        ]

        topics.append({
            "topic_id": topic_id,
            "keywords": top_words
        })

    topic_matrix = lda.transform(X)

    return topics, topic_matrix