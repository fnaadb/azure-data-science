import pandas as pd


class Featurize:

    arief= "this is a class variable"
    #-> that's the return type
    def __init__(self, pandasdataframe):
        self.pd=pandasdataframe

    def callarief(self):
        #this is how you call a class variable
        print(Featurize.arief)
        #or you can call using self, first it will check the instance variable then it goes and check the class variable
        print(self.arief)

    @classmethod
    def this_class(cls, name):
        cls.arief=name

    @staticmethod
    #no cls, or self
    def is_work(day):
        print(day)

    @property
    def email(self):
        return 'my at email.com'
    
    @email.setter
    def email(self,email):
        self.email = email

#this is inheritance
class Developer(Featurize):
    def __init__(self, pandasdataframe,firstname, lastname):
        self.firstname=firstname
        self.lastname=lastname
        super().__init__(pandasdataframe)



#__name__ is set to main, special variable, whenever you import a file, it will run the code and
# get's its name
# if it's the main file that's run, that's get the main name, for e.g. if you import a module
# and run with the second module, the second module is the main 


duh =pd.DataFrame(['1','2'])
p=Featurize(duh)
p.callarief()
#prints all the variable and their values
print(p.__dict__)

d= Developer(duh,"arief","bavan")
d.this_class("what")
#tostring()method in java
repr(d)
str(d)
d.email
print(help(Developer))

#property, access method like you are accessing an attribute

