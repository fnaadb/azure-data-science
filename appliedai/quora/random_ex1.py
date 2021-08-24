import numpy as np;
import pandas as pd;
import os
"""
d=np.array([[2,2],[2,2],[1,1],[1,1]])

print(d.shape)
print(d)
m=d.mean(axis=0)
print(m)
print(m.shape)
print(type(m))

"""
"""
d={'a':1,'b':2}
print(type(d))
"""

vec1=[]
a=[1, 2 ,3]
b=[3, 4, 3]
vec1.append(a)
vec1.append(b)
print(vec1)

df=pd.DataFrame()
df["ok"]=vec1
#print(df.head())
print("&&&&&&&&&&&&&&&&&&&&&&&&")
print(df.ok.values.tolist())


"""
df1=pd.DataFrame({"city":['mumbai','ahmedabad'],"1":[1,2]})
df2=pd.DataFrame({"city":['mumbai','ahmedabad'],"1":[3,4]})
df3=pd.DataFrame({"city":['mumbai','ahmedabad'],"1":[5,6]})

newframe=df1.merge(df2, on="city", how="left")
finalframe=newframe.merge(df3, on="city", how="left")
print(finalframe.head())
"""

""" arief
print(os.getcwd())
dfword2vec=pd.read_csv("%s/%s"%('./data','local_word2vec_features.csv'),encoding='latin-1')
newframe=pd.DataFrame(dfword2vec.q1_feats_m.values.tolist(),index=dfword2vec.index)
print(newframe.head())

"""