
from azureml.core import runconfig, Run

from azureml.core import Workspace, Experiment, ScriptRunConfig, Datastore, Dataset

import os
import sys


from featurize import Featurize


def main():
    mounted_input_path = sys.argv[1]
    mounted_output_path = sys.argv[2]
    print("mounted_input_path:",mounted_input_path)
    print("mounted_output_path:", mounted_output_path)
    print("Running aml_train_feature_1.py")
    f=Featurize()
    
    input_dataset = Run.get_context().input_datasets['quora_training']
    type(input_dataset)
    print("got the dataset")
    input_df=input_dataset.to_pandas_dataframe()
    output_df = f.a_construct_initial_features_without_preprocessing_with_dataframe(input_df)
    os.makedirs(mounted_output_path,exist_ok=True)
    #os.path.join(mounted_output_path,"df_fe_without_preprocessing_train.csv")
    output_df.to_csv("%s/%s"%(mounted_output_path,"df_fe_without_preprocessing_train.csv"), index=False)
    print(output_df.head(5))
    




if __name__ == '__main__':
    main()
