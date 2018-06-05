# from data_helper import *
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
# from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


def get_doc_top_topics_probs(topic2doc, n):

    """

    :param topic2doc: Topic-to-document matrix
    :param n: Top topic probs to be extracted
    :return: Top topics dictionary; key: document index, value: array of topic numbers
    """
    top_topics_dict = []
    for idx in range(topic2doc.shape[0]):
        topic_idxs = np.argpartition(topic2doc[idx], -n)[-n:]
        # topic_probs = topic2doc[idx][topic_idxs]
        # top_topics_dict[idx] = topic_idxs
        top_topics_dict.append(topic_idxs.item(0))
    return top_topics_dict


def get_topic_labels(topic2doc):
    """

    :param topic2doc: Topic-to-document matrix
    :return: List of topic labels
    """
    labels = []
    for n in range(topic2doc.shape[0]):
        topic_most_prob = topic2doc[n].argmax()
        print("doc: {} topic: {}\n".format(n, topic_most_prob))
        labels.append(topic_most_prob)
    return labels

def get_topic_top_words(model, features_vocab, no_words, topic_id):
    """

    :param model: Model instance
    :param features_vocab: Model features
    :param no_words: Number of words to be extracted
    :param topic_id: Topic id number
    :return:
    """
    word_topic_matrix = model.components_
    print("\nTopic #{}:".format(topic_id))
    extracted_words = " ".join([features_vocab[i] for i in word_topic_matrix[topic_id].argsort()[:-no_words-1:-1]])
    return extracted_words


def plot_heatmap(matrix, labels):
    """

    :param matrix:  Data distribution matrix
    :param labels: Topic labels array
    :return:
    """
    row_lbls = ["doc " + str(i) for i in range(matrix.shape[0])]
    # column_lbls = ["topic " + str(i) for i in range(matrix.shape[1])]
    df = pd.DataFrame(data=matrix,
                      index=row_lbls,
                      columns=labels)
    sb.set()
    sb.heatmap(df)
    plt.show()










