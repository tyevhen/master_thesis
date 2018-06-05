import requests
import re
from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup

def getHTML(url):
    resp = requests.get(url)
    return resp.text

def getPost(html_content):
    soup = BeautifulSoup(html_content, 'html5lib')
    content = soup.find('div', {'class': 'main_nrn'})
    return content.text

def preprocPost(string):
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\.", ". ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\xa0", ' ', string)
    string = re.sub(r"\n", '', string)
    return string

def writeSampleFile(text_content, filename):
    sents = sent_tokenize(text_content)
    with open(filename, 'w', encoding='utf8') as out_file:
        for sent in sents:
            out_file.write(sent+'\n')
    out_file.close()


html = getHTML('https://www.alz.org/living_with_alzheimers_8637.asp')
post = getPost(html)
clean = preprocPost(post)
writeSampleFile(clean, 'text.txt')