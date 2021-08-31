
from typing import Dict
import pandas as pd
import numpy as np
#arief
#import distance
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
import debugpy


import re
import time
import warnings
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from pandas.core.frame import DataFrame
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
warnings.filterwarnings("ignore")

#the whole damn spacy thing
import subprocess
import sys
import os
from spacy.cli.download import download as spacy_download


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


from tqdm import tqdm

# exctract word2vec vectors
# https://github.com/explosion/spaCy/issues/1721
# http://landinghub.visualstudio.com/visual-cpp-build-tools
import spacy



class Featurize:

    SAFE_DIV = 0.0001
    STOP_WORDS = stopwords.words("english")
    porter = PorterStemmer()
    pattern = re.compile('\W')
    spacy_download('en_core_web_sm')

    
        

    def a_construct_initial_features_without_preprocessing_with_dataframe(self,df:pd.DataFrame)->pd.DataFrame:
        df=self.__fill_nulls_na(df)
        df.head(2)
        df=self.__construct_obvious_features(df)
        return df

    def b_construct_features_with_preprocessing_with_dataframe(self,df:pd.DataFrame)->pd.DataFrame:
        df=self.__extract_features(df)
        return df

    def c_construct_tfidf_w2vec_features_with_dataframe(self,df:pd.DataFrame)->pd.DataFrame:
        df=self.__fill_nulls_na(df)
        print("starting dictionary")
        dict_tfidf=self.__calculate_tf_idf(df)
        print("completed dictionary")
        df=self.__calculate_tfidf_weighted_question_word2vec(df,dict_tfidf,'question1','q1_feats_m')
        print("calculated q1 tfidf")
        df=self.__calculate_tfidf_weighted_question_word2vec(df,dict_tfidf,'question2','q2_feats_m')
        print("calculated q2 tfidf")
        df3_q1 = pd.DataFrame(df.q1_feats_m.values.tolist(), index= df.index)
        print("----------")
        df3_q2 = pd.DataFrame(df.q2_feats_m.values.tolist(), index= df.index)
        print("what")
        df3_q1['id']=df['id']
        df3_q2['id']=df['id']
        print("id sync")
        finalframe  = df3_q1.merge(df3_q2, on='id',how='left')
        print("merge")
        return finalframe

    def d_construct_final_features(self,dfppro:pd.DataFrame,dfdfnlp:pd.DataFrame,dfword2vec:pd.DataFrame)->pd.DataFrame:
        result = self.__mash_drop_duplicate_fields(dfppro,dfdfnlp,dfword2vec)
        return result


    #public Methods============================================================================================================
    def a_construct_initial_features_without_preprocessing(self, dirpath:str='../data', input_file_name:str='local_train.csv',output_file_name: str='local_df_fe_without_preprocessing_train.csv'):
        df=pd.read_csv("%s/%s"%(dirpath,input_file_name))
        df=self.__fill_nulls_na(df)
        df.head(2)
        df=self.__construct_obvious_features(df)
        df.to_csv("%s/%s"%(dirpath,output_file_name), index=False)
        #df=self.__extract_features(df)
        
    def b_construct_features_with_preprocessing(self,dirpath:str='../data', input_file_name:str='local_df_fe_without_preprocessing_train.csv',output_file_name: str='local_nlp_features_train.csv'):
        df=pd.read_csv("%s/%s"%(dirpath,input_file_name))
        df=self.__extract_features(df)
        df.to_csv("%s/%s"%(dirpath,output_file_name), index=False)

    def c_construct_tfidf_w2vec_features(self,dirpath:str='../data', input_file_name:str='local_train.csv',output_file_name: str='local_word2vec_features.csv'):
        df=pd.read_csv("%s/%s"%(dirpath,input_file_name))
        #calcualte tfidf dictionary
        dict_tfidf=self.__calculate_tf_idf(df)
        df=self.__calculate_tfidf_weighted_question_word2vec(df,dict_tfidf,'question1','q1_feats_m')
        df=self.__calculate_tfidf_weighted_question_word2vec(df,dict_tfidf,'question2','q2_feats_m')
        df3_q1 = pd.DataFrame(df.q1_feats_m.values.tolist(), index= df.index)
        df3_q2 = pd.DataFrame(df.q2_feats_m.values.tolist(), index= df.index)
        df3_q1['id']=df['id']
        df3_q2['id']=df['id']
        
        finalframe  = df3_q1.merge(df3_q2, on='id',how='left')
        finalframe.to_csv("%s/%s"%(dirpath,output_file_name), index=False)

    

    def d_construct_final_features(self,dirpath:str='../data', without_processing_file:str='local_df_fe_without_preprocessing_train.csv',nlp_features_train:str='local_nlp_features_train.csv',word2vecfeatures:str='local_word2vec_features.csv', finalfeaturesfile:str="local_final_features.csv"):
        dfppro=pd.read_csv("%s/%s"%(dirpath,without_processing_file))
        dfdfnlp=pd.read_csv("%s/%s"%(dirpath,nlp_features_train))
        dfword2vec=pd.read_csv("%s/%s"%(dirpath,word2vecfeatures))
        result = self.__mash_drop_duplicate_fields(dfppro,dfdfnlp,dfword2vec)
        result.to_csv("%s/%s"%(dirpath,finalfeaturesfile))


    #main private methods==========================================================================================================    
    

    
    def __mash_drop_duplicate_fields(self,dfppro:pd.DataFrame,nlp_dataframe:pd.DataFrame,dfword2vec:pd.DataFrame)->pd.DataFrame:
        df1 = nlp_dataframe.drop(['qid1','qid2','question1','question2'],axis=1)
        df2 = dfppro.drop(['qid1','qid2','question1','question2','is_duplicate'],axis=1)
        #df3 = dfword2vec.drop(['qid1','qid2','question1','question2','is_duplicate'],axis=1)
        #df3_q1 = pd.DataFrame(df3.q1_feats_m.values.tolist(), index= df3.index)
        #df3_q2 = pd.DataFrame(df3.q2_feats_m.values.tolist(), index= df3.index)

        #df3_q1['id']=df1['id']
        #df3_q2['id']=df1['id']
        df1  = df1.merge(df2, on='id',how='left')
        #df2  = df3_q1.merge(df3_q2, on='id',how='left')
        result  = df1.merge(dfword2vec, on='id',how='left')
        return result

    
    def __construct_obvious_features(self, df: pd.DataFrame)->pd.DataFrame:
        df['freq_qid1'] = df.groupby('qid1')['qid1'].transform('count') 
        df['freq_qid2'] = df.groupby('qid2')['qid2'].transform('count')
        df['q1len'] = df['question1'].str.len() 
        df['q2len'] = df['question2'].str.len()
        df['q1_n_words'] = df['question1'].apply(lambda row: len(row.split(" ")))
        df['q2_n_words'] = df['question2'].apply(lambda row: len(row.split(" ")))
        df['word_Common'] = df.apply(self.__normalized_word_Common, axis=1)
        df['word_Total'] = df.apply(self.__normalized_word_Total, axis=1)
        df['word_share'] = df.apply(self.__normalized_word_share, axis=1)
        df['freq_q1+q2'] = df['freq_qid1']+df['freq_qid2']
        df['freq_q1-q2'] = abs(df['freq_qid1']-df['freq_qid2'])
        return df
        
        

    def __extract_features(self,df:pd.DataFrame)->pd.DataFrame:
        # preprocessing each question
        df["question1"] = df["question1"].fillna("").apply(self.__preprocess)
        df["question2"] = df["question2"].fillna("").apply(self.__preprocess)

        print("token features...")
        
        # Merging Features with dataset
        
        token_features = df.apply(lambda x: self.__get_token_features(x["question1"], x["question2"]), axis=1)
        
        df["cwc_min"]       = list(map(lambda x: x[0], token_features))
        df["cwc_max"]       = list(map(lambda x: x[1], token_features))
        df["csc_min"]       = list(map(lambda x: x[2], token_features))
        df["csc_max"]       = list(map(lambda x: x[3], token_features))
        df["ctc_min"]       = list(map(lambda x: x[4], token_features))
        df["ctc_max"]       = list(map(lambda x: x[5], token_features))
        df["last_word_eq"]  = list(map(lambda x: x[6], token_features))
        df["first_word_eq"] = list(map(lambda x: x[7], token_features))
        df["abs_len_diff"]  = list(map(lambda x: x[8], token_features))
        df["mean_len"]      = list(map(lambda x: x[9], token_features))
        #Computing Fuzzy Features and Merging with Dataset
    
        # do read this blog: http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
        # https://stackoverflow.com/questions/31806695/when-to-use-which-fuzz-function-to-compare-2-strings
        # https://github.com/seatgeek/fuzzywuzzy
        print("fuzzy features..")

        df["token_set_ratio"]       = df.apply(lambda x: fuzz.token_set_ratio(x["question1"], x["question2"]), axis=1)
        # The token sort approach involves tokenizing the string in question, sorting the tokens alphabetically, and 
        # then joining them back into a string We then compare the transformed strings with a simple ratio().
        df["token_sort_ratio"]      = df.apply(lambda x: fuzz.token_sort_ratio(x["question1"], x["question2"]), axis=1)
        df["fuzz_ratio"]            = df.apply(lambda x: fuzz.QRatio(x["question1"], x["question2"]), axis=1)
        df["fuzz_partial_ratio"]    = df.apply(lambda x: fuzz.partial_ratio(x["question1"], x["question2"]), axis=1)
        #arief to avoid custom deployment of the modules.
        #df["longest_substr_ratio"]  = df.apply(lambda x: self.__get_longest_substr_ratio(x["question1"], x["question2"]), axis=1)
        df["longest_substr_ratio"]  = 0.5
        return df
        

    
   
   #----------------------- All utility features______________________________________________ 
    
    #Word to vec Features utility methods.

    def __calculate_tf_idf(self, df:pd.DataFrame)->dict:
        
        # merge texts
        questions = list(df['question1']) + list(df['question2'])

        tfidf = TfidfVectorizer(lowercase=False, )
        tfidf.fit_transform(questions)
        #dict key:word and value:tf-idf score
        word2tfidf = dict(zip(tfidf.get_feature_names(), tfidf.idf_))
        return word2tfidf

    def __calculate_tfidf_weighted_question_word2vec(self, df:pd.DataFrame,word2tfidf:dict,column_to_be_analyzed:str,feature_column_name:str)->pd.DataFrame:
        # en_vectors_web_lg, which includes over 1 million unique vectors.
        #subprocess.call(['python', '-m', "spacy", "download", "en_core_web_sm"])
        #print("before spacy load")
       
        nlp = spacy.load('en_core_web_sm')
        print("arief", nlp("show me the money"))

        vecs1 = []
        # https://github.com/noamraph/tqdm
        # tqdm is used to print the progress bar
        for qu1 in tqdm(list(df[column_to_be_analyzed])):
            doc1 = nlp(qu1) 
            # 384 is the number of dimensions of vectors 
            mean_vec1 = np.zeros([len(doc1), len(doc1[0].vector)])
            for word1 in doc1:
                # word2vec
                vec1 = word1.vector
                # fetch df score
                try:
                    idf = word2tfidf[str(word1)]
                except:
                    idf = 0
                # compute final vec
                mean_vec1 += vec1 * idf
            mean_vec1 = mean_vec1.mean(axis=0)
            vecs1.append(mean_vec1)
        #this is amazing, each list WITHIN the LIST will be aligned with each row of the dataframe, doesn't matter whether you have the list() method called
        #in the front, this is huge coming from java.
        df[feature_column_name] = list(vecs1)
        return df
    
    
    #---------------------------------------Word to Vec features end
    def __fill_nulls_na(self, mydataframe: pd.DataFrame)->pd.DataFrame:
        mydataframe=mydataframe.fillna('')
        return mydataframe

    def __normalized_word_Common(self,row):
            w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
            w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
            return 1.0 * len(w1 & w2)

    def __normalized_word_Total(self,row):
            w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
            w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
            return 1.0 * (len(w1) + len(w2))
    
    def __normalized_word_share(self,row):
            w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
            w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
            return 1.0 * len(w1 & w2)/(len(w1) + len(w2))
    """ arief avoid aml custom deployment of the module
    
    def __get_longest_substr_ratio(self,a, b):
        strs = list(distance.lcsubstrings(a, b))
        if len(strs) == 0:
            return 0
        else:
            return len(strs[0]) / (min(len(a), len(b)) + 1)

    """
    def __preprocess(self,x):
        
        x = str(x).lower()
        x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")\
                            .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
                            .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
                            .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
                            .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
                            .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
                            .replace("€", " euro ").replace("'ll", " will")
        x = re.sub(r"([0-9]+)000000", r"\1m", x)
        x = re.sub(r"([0-9]+)000", r"\1k", x)
        
        

        
        if type(x) == type(''):
            x = re.sub(Featurize.pattern, ' ', x)
        
        
        if type(x) == type(''):
            x = Featurize.porter.stem(x)
            example1 = BeautifulSoup(x)
            x = example1.get_text()
                
        
        return x 

    def __get_token_features(self,q1, q2):
        token_features = [0.0]*10
        
        # Converting the Sentence into Tokens: 
        q1_tokens = q1.split()
        q2_tokens = q2.split()

        if len(q1_tokens) == 0 or len(q2_tokens) == 0:
            return token_features
        # Get the non-stopwords in Questions
        q1_words = set([word for word in q1_tokens if word not in Featurize.STOP_WORDS])
        q2_words = set([word for word in q2_tokens if word not in Featurize.STOP_WORDS])
        
        #Get the stopwords in Questions
        q1_stops = set([word for word in q1_tokens if word in Featurize.STOP_WORDS])
        q2_stops = set([word for word in q2_tokens if word in Featurize.STOP_WORDS])
        
        # Get the common non-stopwords from Question pair
        common_word_count = len(q1_words.intersection(q2_words))
        
        # Get the common stopwords from Question pair
        common_stop_count = len(q1_stops.intersection(q2_stops))
        
        # Get the common Tokens from Question pair
        common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
        
        
        token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + Featurize.SAFE_DIV)
        token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + Featurize.SAFE_DIV)
        token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + Featurize.SAFE_DIV)
        token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + Featurize.SAFE_DIV)
        token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + Featurize.SAFE_DIV)
        token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + Featurize.SAFE_DIV)
        
        # Last word of both question is same or not
        token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
        
        # First word of both question is same or not
        token_features[7] = int(q1_tokens[0] == q2_tokens[0])
        
        token_features[8] = abs(len(q1_tokens) - len(q2_tokens))
        
        #Average Token Length of both Questions
        token_features[9] = (len(q1_tokens) + len(q2_tokens))/2
        return token_features

    # get the Longest Common sub string
  
    