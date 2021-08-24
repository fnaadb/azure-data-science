import numpy as np
import pandas as pd

from tqdm import tqdm
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

list_questions =  ['what is india?','USA is the greatest country']




tfidf = TfidfVectorizer(lowercase=False, )
tfidf.fit_transform(list_questions)


# dict key:word and value:tf-idf score
word2tfidf = dict(zip(tfidf.get_feature_names(), tfidf.idf_))
#print(word2tfidf["india"])


nlp = spacy.load('en_core_web_sm')

vecs1 = []

for qu in list_questions:
    doc1=nlp(qu)
    print("*******************Doc******************************")
    # 384 is the number of dimensions of vectors 
    mean_vec1 = np.zeros([len(doc1), len(doc1[0].vector)])
    #print("initialize mean {}".format(mean_vec1))
    #print("shape of doc {}".format(mean_vec1.shape))
    for word1 in doc1:
        # word2vec
        vec1 = word1.vector
        # fetch df score
        try:
            idf = word2tfidf[str(word1)]
     #       print("idf {}".format(idf))
        except:
            idf = 0
        # compute final vec
        mean_vec1 += vec1 * idf
    #print('mean_vec after idf {}'.format(mean_vec1))
    mean_vec1 = mean_vec1.mean(axis=0)
    #print(mean_vec1.shape)
    #print("mean_vec at the end:{}".format(mean_vec1))
    vecs1.append(mean_vec1)
    print(list(vec1))
    print(vec1.shape)

    #
    # print(help(doc1))
    #print('printing the actual doc {}'.format(len(doc1)))
    #print(len(doc1.vector))
    #print("&&&&&&&&&&")
    #print(len(doc1[0].vector))
   # mean_vec1 = np.zeros([len(doc1), len(doc1[0].vector)])
   # print(mean_vec1.shape)
    #print('printing len of vector {}'.format(doc1.vector))
    #print('first element of doc {}'.format(doc1.vector[0]))
    print("*****************************************")