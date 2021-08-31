from azureml.core import run, runconfig, Run

from azureml.core import Workspace, Experiment, ScriptRunConfig, Datastore, Dataset

import os
import sys




from featurize import Featurize


def main():

    
#-----------------------------------------------------------
    os.system(f"python -m spacy download en_core_web_sm") 
    print("spacy installed")
    print("Running aml_train_feature_3.py wor2vec features")
    mounted_input_path = sys.argv[1]
    mounted_output_path = sys.argv[2]
    print("mounted_input_path:",mounted_input_path)
    print("mounted_output_path:", mounted_output_path)
    f=Featurize()
    
    input_dataset = Run.get_context().input_datasets['quora_training']
    type(input_dataset)
    print("got the dataset")
    input_df=input_dataset.to_pandas_dataframe()
    output_df = f.c_construct_tfidf_w2vec_features_with_dataframe(input_df)
    os.makedirs(mounted_output_path,exist_ok=True)
    #os.path.join(mounted_output_path,"df_fe_without_preprocessing_train.csv")
    output_df.to_csv("%s/%s"%(mounted_output_path,"df_word2_vec_features.csv"), index=False)
    print(output_df.head(5))
   
    




if __name__ == '__main__':
    main()
