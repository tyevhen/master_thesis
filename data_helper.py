import re
import os
import pickle
import pandas as pd
import numpy as np
import nltk
from io import open
from collections import Counter
from nltk import FreqDist, pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt


HEALTHY_CLEAN = "healthy_prep/"
ILL_CLEAN = "ill_prep/"

HEALTHY_RAW = "healthy_blogs/"
ILL_RAW = "ill_blogs/"

RAW_FILES = [HEALTHY_RAW, ILL_RAW]
ALL_POSTS = "all_posts/"

FIRST_PRONOUNS = set(['i', 'me', 'myself', 'mine'])
ENG_STOPWORDS = set(stopwords.words('english'))
CONTR_PUNCT = set(["'ve", "'s", "'d", "'m", "'re", "'ll", ",",
                   "'", "!", "?", "com", "http", "www", "html", "yahoo",
                   "u", "ect", "qi", "b", "lt", "gt", "e", "ce", "q",
                   "act", "sec", "c", "g", "w", "h", "x", "awt", "n",
                   "int", "ar", "us", "j", "cc"]).union(ENG_STOPWORDS)
DROP_LIST = CONTR_PUNCT.difference(FIRST_PRONOUNS)

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r'^https?:\/\/.*[\r\n]*', '', string)
    string = re.sub(r"[^A-Za-z0-9(),!?\']", " ", string)
    string = re.sub(r"([0-9])", "", string)
    string = re.sub(r"\\\\", "", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower().replace('\n', ' ')

def preproc_file(src_file):
    """

    :param src_file: Path to raw text file
    :return: Clean, filtered document str
    """
    loaded_data = open(src_file, "r", encoding="utf8").read()
    clean_doc = clean_str(loaded_data)
    lemm = WordNetLemmatizer()
    clean_tok = []
    for word, tag in pos_tag(word_tokenize(clean_doc)):
        tg = tag[0].lower()
        tg = tg if tg in ['a', 'r', 'n', 'v'] else None
        lemma = lemm.lemmatize(word, tg) if tg else word
        clean_tok.append(lemma)
    for i in range(len(clean_tok)):
        word = clean_tok[i]
        if word not in DROP_LIST:
            clean_tok[i] = word
        else:
            clean_tok[i] = "UNK"
        # clean_tok = [word for word in clean_tok if word not in DROP_LIST]
    clean_flat = " ".join(clean_tok)
    return clean_flat


def write_prep_file(clean_file, target_path):
    """

    :param clean_file: Clean document
    :param target_path: Path to new file
    :return:
    """
    target_file = open(target_path, 'w', encoding="utf8")
    target_file.write(unicode(clean_file))
    target_file.close()

def prepare_corpus(corpus_path, dump_path, merge_authors):
    """

    :param corpus_path: Path to directory with raw documents
    :param dump_path: Path to dump file
    :param merge_authors: TRUE/FALSE flag responsible for merging same blog posts to a single file
    :return:
    """
    docs = []
    if merge_authors:
        for author, posts in group_filenames(corpus_path).items():
            curr_blog = []
            for post in posts:
                doc = open(corpus_path+post, "r", encoding="utf8").read()
                tokens = word_tokenize(doc)
                curr_blog += tokens
                if len(tokens) != 0:
                    flat_doc = " ".join(curr_blog)
                    docs.append(flat_doc)
                else:
                    pass
        dump_corpus(docs, dump_path)
    else:
        data = []
        for file in sorted(os.listdir(corpus_path)):
            doc = open(corpus_path+file, "r", encoding="utf8").read()
            if len(word_tokenize(doc)) != 0:
                docs.append(doc)
                if "healthy" in corpus_path:
                    row = {"post": doc, "label": 0, "probs": 0}
                    data.append(row)
                else:
                    row = {"post": doc, "label": 1, "probs": 0}
                    data.append(row)
            else:
                pass
        df = pd.DataFrame(data)
        dump_corpus(df, dump_path)
        # dump_corpus(docs, dump_path)

def group_filenames(raw_corpus_path):
    """

    :param raw_corpus_path: Path to the directory with raw files
    :return: Dictionary where key is a unique blog name and value is a list of filenames
    """
    names = {}
    for file in os.listdir(raw_corpus_path):
        currname = re.sub(r'\d+', '', file)
        if currname not in names.keys():
            names[currname] = [file]
        else:
            names[currname] += [file]
    return names


def prepare_data_smarter(h_path, i_path, split, fastText):
    ill_names = [name for name in os.listdir(i_path) if os.path.isfile(os.path.join(i_path, name))]
    healthy_names = sorted([name for name in os.listdir(h_path) if os.path.isfile(os.path.join(h_path, name))])

    cnt = Counter()
    i_dict = Counter([re.sub(r'\d+', '', nm) for nm in ill_names])
    h_dict = Counter([re.sub(r'\d+', '', nm) for nm in healthy_names])

    authors_blogs = {}

    for key in i_dict.keys():
        root = key.split('.')[0]
        matches = [os.path.join(i_path, fname) for fname in ill_names if root in fname.split('.')[0]]
        authors_blogs[key] = matches

    for key in h_dict.keys():
        root = key.split('.')[0]
        matches = [os.path.join(h_path, fname) for fname in healthy_names if root in fname.split('.')[0]]
        authors_blogs[key] = matches

    post_sum = 0
    tok_total = sum(i_dict.values())
    observed = []
    for k, v in h_dict.items():
        observed.append(k)
        post_sum += v
        if post_sum > tok_total:
            remove = [key for key in h_dict.keys() if key not in observed]
            for key in remove:
                del h_dict[key]
            break

    i_tup = [(k, v) for k, v in i_dict.items()]
    h_tup = [(k, v) for k, v in h_dict.items()]

    i_tup.sort(key=lambda x: x[1], reverse=True)
    h_tup.sort(key=lambda x: x[1], reverse=True)

    num_tr_ill = int(sum(i_dict.values()) * split)
    num_dev_ill = int(sum(i_dict.values()) * (1-split) / 2)

    num_tr_hlth = int(sum(h_dict.values()) * split)
    num_dev_hlth = int(sum(h_dict.values()) * (1-split) / 2)

    train_pt = 0
    dev_pt = 0
    test_pt = 0

    i_train = {}
    h_train = {}
    i_dev = {}
    h_dev = {}
    i_test = {}
    h_test = {}

    for k in range(len(i_tup)):
        if not (train_pt > num_tr_ill):
            if (i_tup[k][0] and i_tup[len(i_tup)-k-1][0]) not in list(i_test.keys()) + list(i_dev.keys()) + list(i_train.keys()):
                i_train[i_tup[k][0]] = authors_blogs[i_tup[k][0]]
                i_train[i_tup[len(i_tup)-k-1][0]] = authors_blogs[i_tup[len(i_tup)-k-1][0]]

                train_pt += i_tup[k][1]
                train_pt += i_tup[len(i_tup)-k-1][1]
            else:
                pass

        elif not (dev_pt > num_dev_ill):
            if (i_tup[k][0] and i_tup[len(i_tup)-k-1][0]) not in list(i_test.keys()) + list(i_dev.keys()) + list(i_train.keys()):
                i_dev[i_tup[k][0]] = authors_blogs[i_tup[k][0]]
                i_dev[i_tup[len(i_tup)-k-1][0]] = authors_blogs[i_tup[len(i_tup)-k-1][0]]

                dev_pt += i_tup[k][1]
                dev_pt += i_tup[len(i_tup)-k-1][1]
            else:
                pass
        elif not (test_pt > num_dev_ill):
            if (i_tup[k][0] and i_tup[len(i_tup)-k-1][0]) not in list(i_test.keys()) + list(i_dev.keys()) + list(i_train.keys()):
                i_test[i_tup[k][0]] = authors_blogs[i_tup[k][0]]
                i_test[i_tup[len(i_tup)-k-1][0]] = authors_blogs[i_tup[len(i_tup)-k-1][0]]

                test_pt += i_tup[k][1]
                test_pt += i_tup[len(i_tup)-k-1][1]
            else:
                pass
    train_pt = 0
    dev_pt = 0
    test_pt = 0

    for k in range(len(h_tup)):
        if not (train_pt > num_tr_hlth):
            if (h_tup[k][0] and h_tup[len(h_tup)-k-1][0]) not in list(h_test.keys()) + list(h_dev.keys()) + list(h_train.keys()):
                h_train[h_tup[k][0]] = authors_blogs[h_tup[k][0]]
                h_train[h_tup[len(h_tup)-k-1][0]] = authors_blogs[h_tup[len(h_tup)-k-1][0]]

                train_pt += h_tup[k][1]
                train_pt += h_tup[len(h_tup)-k-1][1]
            else:
                pass

        elif not (dev_pt > num_dev_hlth):
            if (h_tup[k][0] and h_tup[len(h_tup)-k-1][0]) not in list(h_test.keys()) + list(h_dev.keys()) + list(h_train.keys()):
                h_dev[h_tup[k][0]] = authors_blogs[h_tup[k][0]]
                h_dev[h_tup[len(h_tup)-k-1][0]] = authors_blogs[h_tup[len(h_tup)-k-1][0]]

                dev_pt += h_tup[k][1]
                dev_pt += h_tup[len(h_tup)-k-1][1]
            else:
                pass

        elif not (test_pt > num_dev_ill):
            if (h_tup[k][0] and h_tup[len(h_tup)-k-1][0]) not in list(h_test.keys()) + list(h_dev.keys()) + list(h_train.keys()):
                h_test[h_tup[k][0]] = authors_blogs[h_tup[k][0]]
                h_test[h_tup[len(h_tup)-k-1][0]] = authors_blogs[h_tup[len(h_tup)-k-1][0]]

                test_pt += h_tup[k][1]
                test_pt += h_tup[len(h_tup)-k-1][1]
            else:
                pass


    train_dict = merge_two_dicts(i_train, h_train)  #{**i_train, **h_train}
    dev_dict = merge_two_dicts(i_dev, h_dev)  #{**i_dev, **h_dev}
    test_dict = merge_two_dicts(i_test, h_test)  #{**i_test, **h_test}

    train_docs = []
    dev_docs = []
    test_docs = []

    train_lbls = []
    dev_lbls = []
    test_lbls = []

    if fastText:
        train_doc = open("tr_fast_last.txt", "w", encoding="utf8")
        for key in train_dict.keys():
            if key in h_dict.keys():
                for blog in train_dict[key]:
                    doc = open(blog, "r", encoding="utf8").read()
                    train_doc.write(" __label__control "+doc+'\n')
            else:
                for blog in train_dict[key]:
                    doc = open(blog, "r", encoding="utf8").read()
                    train_doc.write(" __label__clinical "+doc+'\n')
        train_doc.close()

        dev_doc = open("dev_fast_last.txt", "w", encoding="utf8")
        for key in dev_dict.keys():
            if key in h_dict.keys():
                for blog in dev_dict[key]:
                    doc = open(blog, "r", encoding="utf8").read()
                    # dev_docs.append(doc+"\n")
                    dev_doc.write(" __label__control " + doc + '\n')
                    dev_docs.append(" __label__control " + doc + '\n')
                    dev_lbls.append("control")
            else:
                for blog in dev_dict[key]:
                    doc = open(blog, "r", encoding="utf8").read()
                    dev_doc.write(" __label__clinical " + doc + '\n')
                    dev_docs.append(" __label__clinical " + doc + '\n')
                    dev_lbls.append("clinical")
        dev_doc.close()

        test_doc = open("test_fast_last.txt", "w", encoding="utf8")
        for key in test_dict.keys():
            if key in h_dict.keys():
                for blog in test_dict[key]:
                    doc = open(blog, "r", encoding="utf8").read()
                    # dev_docs.append(doc+"\n")
                    test_doc.write(" __label__control " + doc + '\n')
                    test_docs.append(" __label__control " + doc + '\n')
                    test_lbls.append("control")
            else:
                for blog in test_dict[key]:
                    doc = open(blog, "r", encoding="utf8").read()
                    test_doc.write(" __label__clinical " + doc + '\n')
                    test_docs.append(" __label__clinical " + doc + '\n')
                    test_lbls.append("clinical")
        test_doc.close()

        dump_corpus(dev_docs, "Xdev_fast")
        dump_corpus(test_docs, "Xtest_fast")
        dump_corpus(train_lbls, "Ytrain_fast")
        dump_corpus(dev_lbls, "Ydev_fast")
        dump_corpus(test_lbls, "Ytest_fast")

    else:
        for key in train_dict.keys():
            if key in h_dict.keys():
                for blog in train_dict[key]:
                    doc = open(blog, "r", encoding="utf8").read()
                    train_docs.append(doc)
                    train_lbls.append(0)
            else:
                for blog in train_dict[key]:
                    doc = open(blog, "r", encoding="utf8").read()
                    train_docs.append(doc)
                    train_lbls.append(1)

        for key in dev_dict.keys():
            if key in h_dict.keys():
                for blog in dev_dict[key]:
                    doc = open(blog, "r", encoding="utf8").read()
                    dev_docs.append(doc)
                    dev_lbls.append(0)
            else:
                for blog in dev_dict[key]:
                    doc = open(blog, "r", encoding="utf8").read()
                    dev_docs.append(doc)
                    dev_lbls.append(1)

        for key in test_dict.keys():
            if key in h_dict.keys():
                for blog in test_dict[key]:
                    doc = open(blog, "r", encoding="utf8").read()
                    test_docs.append(doc)
                    test_lbls.append(0)
            else:
                for blog in test_dict[key]:
                    doc = open(blog, "r", encoding="utf8").read()
                    test_docs.append(doc)
                    test_lbls.append(1)

        dump_corpus(train_docs, "train_pt_new")
        dump_corpus(dev_docs, "dev_pt_new")
        dump_corpus(test_docs, "test_pt_new")

        dump_corpus(train_lbls, "Ytrain_new")
        dump_corpus(dev_lbls, "Ydev_new")
        dump_corpus(test_lbls, "Ytest_new")

def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z

def show_corpus_statistics(corpus):
    word_tokens = []
    for sent in corpus:
        word_tokens += word_tokenize(sent)
    rubbish_free = [word for word in word_tokens if word not in DROP_LIST or len(word) >= 2]
    freqdist = FreqDist(rubbish_free)
    top_words = freqdist.most_common()[:30]
    bottom_words = freqdist.most_common()[-20:]
    build_barplot(top_words)
    # build_barplot(bottom_words)
    # print("Most frequent words: ", top_words)
    # print("Least frequent words: {}".format(get_least_frequent_words(freqdist, 5).items()))
    print("Number of documents in corpus: {}".format(len(corpus)))
    print("Number of word tokens: {}".format(len(word_tokens)))
    print("Number of different word tokens: {}".format(len(set(word_tokens))))

def build_barplot(dict):
    x, y = zip(*dict)
    plt.bar(x, y)
    plt.xlabel('Word', fontsize=5)
    plt.ylabel('Counts', fontsize=5)
    plt.xticks(x, fontsize=8, rotation=60)
    plt.title('Word token statistics')
    plt.show()

def get_least_frequent_words(dict, n):
    temp = {word: count for word, count in dict.items() if count<n}
    return temp

def dump_corpus(corpus, filename):
    """

    :param corpus: List of documents (corpus)
    :param filename: Dump file name
    :return:
    """
    dump_file = open(filename, "wb")
    pickle.dump(corpus, dump_file, protocol=2)

def load_dump(path):
    """

    :param path: Path to dump file
    :return: List of documents in str
    """
    filehandler = open(path, 'rb')
    corpus = pickle.load(filehandler)
    return corpus


if __name__ == "__main__":
    # for src in RAW_FILES:
    #     if "ill" in src:
    #         target_dir = ILL_CLEAN
    #     elif "healthy" in src:
    #         target_dir = HEALTHY_CLEAN
    #
    #     for file in os.listdir(src):
    #         raw_file = src + file
    #         prep_path = target_dir + file
    #         clean_file = preproc_file(raw_file)
    #         write_prep_file(clean_file, prep_path)

    # prepare_data_smarter(HEALTHY_CLEAN, ILL_CLEAN, 0.7, True)
    # prepare_data_smarter(HEALTHY_CLEAN, ILL_CLEAN, 0.7, False)
    prepare_corpus(HEALTHY_CLEAN, "hdump_df", False)
    prepare_corpus(ILL_CLEAN, "idump_df", False)


