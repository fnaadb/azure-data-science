
import pandas as pd

from a_featurizer import Featurize

duh =pd.DataFrame(['1','2'])
p=Featurize(duh)
p.callarief()
#prints all the variable and their values
print(p.__dict__)
Featurize.this_class("something")
Featurize.arief

#if you don't have a variable in a class like java and if you can assignment it will create a instance variable.__dict__

p.disco="stuff"
print(p.__dict__)