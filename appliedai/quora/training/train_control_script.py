
import pandas as pd
from featurize import Featurize
import os
import spacy


f=Featurize()

df=pd.read_csv("%s/%s"%('../data','train.csv'))
  
#f.a_construct_initial_features_without_preprocessing('../data','local_train.csv','local_df_fe_without_preprocessing_train.csv')
#f.b_construct_features_with_preprocessing('../data','local_df_fe_without_preprocessing_train.csv','local_nlp_features_train.csv')
#f.c_construct_tfidf_w2vec_features('../data','local_train.csv','local_word2vec_features.csv')
#f.d_construct_final_features('../data','local_df_fe_without_preprocessing_train.csv','local_nlp_features_train.csv','local_word2vec_features.csv', "local_final_features.csv")

df1=f.c_construct_tfidf_w2vec_features_with_dataframe(df)

print(df1.shape)

print("done")


