
from azureml.core import runconfig, Run

from azureml.core import Workspace, Experiment, ScriptRunConfig, Datastore, Dataset

import os
import sys


from featurize import Featurize


def main():
    print("Running aml_train_feature_2.py processed features.")
    mounted_input_path = sys.argv[1]
    mounted_output_path = sys.argv[2]
    print("mounted_input_path:",mounted_input_path)
    print("mounted_output_path:", mounted_output_path)
    f=Featurize()
    
    input_dataset = Run.get_context().input_datasets['quora_training']
    type(input_dataset)
    print("got the dataset")
    input_df=input_dataset.to_pandas_dataframe()
    output_df = f.b_construct_features_with_preprocessing_with_dataframe(input_df)
    os.makedirs(mounted_output_path,exist_ok=True)
    #os.path.join(mounted_output_path,"df_fe_without_preprocessing_train.csv")
    output_df.to_csv("%s/%s"%(mounted_output_path,"df_processed_features.csv"), index=False)
    print(output_df.head(5))
    
    




if __name__ == '__main__':
    main()
