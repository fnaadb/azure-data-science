from azureml.core import runconfig, Run

from azureml.core import Workspace, Experiment, ScriptRunConfig, Datastore, Dataset

import os
import sys

from featurize import Featurize


def main():
    print("Running aml_train_feature_4 final mashup.py")
    mounted_input_path_1 = sys.argv[1]
    mounted_input_path_2 = sys.argv[2]
    mounted_input_path_3 = sys.argv[3]
    mounted_output_path = sys.argv[4]
     
    print(mounted_input_path_1,":",mounted_input_path_2,":",mounted_input_path_3,":",mounted_output_path)
       

    f=Featurize()
    
    input_dataset_without_preprocessing = Run.get_context().input_datasets['without_preprocess']
    input_dataset_with_preprocessing = Run.get_context().input_datasets['with_preprocess']
    input_dataset_word2vec = Run.get_context().input_datasets['word2vec']
    #final_output_dataset = Run.get_context().input_datasets['final_output']

    Run.get_context().out
    
    print("got the dataset")
    input_dataset_without_preprocessing_df=input_dataset_without_preprocessing.to_pandas_dataframe()
    input_dataset_with_preprocessing_df=input_dataset_with_preprocessing.to_pandas_dataframe()
    input_dataset_word2vec_df=input_dataset_word2vec.to_pandas_dataframe()
    
    final_dataset_df = f.d_construct_final_features(input_dataset_without_preprocessing_df,input_dataset_with_preprocessing_df,input_dataset_word2vec_df)

    os.makedirs(mounted_output_path,exist_ok=True)
    #os.path.join(mounted_output_path,"df_fe_without_preprocessing_train.csv")
    final_dataset_df.to_csv("%s/%s"%(mounted_output_path,"df_word_2_vec_features.csv"), index=False)
    print(final_dataset_df.head(5))
    




if __name__ == '__main__':
    main()
