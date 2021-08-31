
from azureml.core import Workspace, Experiment, ScriptRunConfig, Datastore, Dataset
from azureml.core import runconfig
from azureml.core.runconfig import RunConfiguration, DEFAULT_CPU_IMAGE 
from azureml.core.conda_dependencies import CondaDependencies


from azureml.data.data_reference import DataReference
from azureml.data import OutputFileDatasetConfig
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig
from azureml.data.tabular_dataset import TabularDataset

from azureml.pipeline.core import PipelineData, Pipeline, PipelineParameter
from azureml.core.compute import AmlCompute, ComputeTarget
import os
from azureml.pipeline.steps import PythonScriptStep

from azureml.pipeline.steps import CommandStep


ws = Workspace.get(
       name='ariefmlworkspace',
       subscription_id='31241946-ad24-4495-85e2-31248c60d164',
        resource_group='ariefinternal')

#ws.write_config(path='../../amlconfig/',file_name="ws_config.json")
default_datastore= ws.get_default_datastore()
compute_name="arief-cluster"
if compute_name in ws.compute_targets:
    compute_target = ws.compute_targets[compute_name]


compute_config = RunConfiguration()
compute_config.target = "arief-cluster"
compute_config.environment.python.user_managed_dependencies = False

dependencies =CondaDependencies.create(conda_packages=['scikit-learn','numpy','pandas','nltk','fuzzywuzzy','bs4','tqdm','spacy','debugpy'])
dependencies.add_pip_package("azureml-core")
compute_config.environment.python.conda_dependencies = dependencies






####----------------------------------------------datastore and dataset-------------------------------------------------------------------------------------------------
source_directory = "./training"


# doesn't like the datareference object, cannot be json serializable
# great reference https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.data.dataset_consumption_config.datasetconsumptionconfig?view=azure-ml-py


"""
blob_input_data = DataReference(
    datastore=def_blob_store,
    data_reference_name="quora_training",
    path_on_datastore="quora/train.csv")
print("DataReference object created")
"""
# first get the datastore #obtain the default blob store where we have stored the quora/train.csv
quora_train_blob_store = ws.get_default_datastore()
# get the dataset from the store, since we have already registered as a dataset
quora_train_dataset =  Dataset.get_by_name(ws,"quora_training",1)
type(quora_train_dataset)
file_pipeline_param = PipelineParameter(name="file_ds_param", default_value=quora_train_dataset)
#represents how to deliver the dataset to a compute target: DatasetConsumptionConfig
"""
   import sys
   download_location = sys.argv[1]

   # The download location can also be retrieved from input_datasets of the run context.
   from azureml.core import Run
   mount_or_download(this is actually a dataset) = Run.get_context().input_datasets['input_1']
   df = mount_or_download.to_pandas_dataframe()
"""
dataset_input =  DatasetConsumptionConfig("input_1", file_pipeline_param)
"""
#OutputDataSetConfig - > OutputFileDataSetConfig and OutputTabularDatasetConfig
You should not call this constructor directly, but instead should create a OutputFileDatasetConfig and 
then call the corresponding read_* methods to convert it into a OutputTabularDatasetConfig.


"""
#this is for the output of a run to be treated as a FileDataset, not the actual file you want to register as an output
without_preprocessed_features_1 = OutputFileDatasetConfig(destination=(quora_train_blob_store, 'quora/without_preprocessed')).register_on_complete(name='quora_features_without_preprocessing')
"""
Different types of run, ScriptRunConfig, PipelineRun and AutoMLRun
in order for me to have this configuration work, to get access to input data and output data it will not work, as this arg parameters need to be converted into dataset
to get access to the dataframe, so i will be moving to the pipelinerun
ws = Run.get_context().experiment.workspace
script_run_config = ScriptRunConfig(source_directory=os.getcwd(),
                                    script="aml_train_feature_1.py",
                                    arguments =["--input","quora_training","--output", "without_preprocessed_features_1"],      
                                    run_config=compute_config)    

experiment = Experiment(workspace=ws, name="arief_quora_experiment")
run = experiment.submit(config=script_run_config)
run.wait_for_completion(show_output=True)                                     
                      
"""
spacy_install_command=CommandStep(name='spacy_download_model',
                                         command= 'pip install spacy',
                                         source_directory=os.getcwd(),
                                         compute_target = compute_target)
spacy_model_download_command=CommandStep(name='spacy_download_model',
                                         command= 'python -m spacy download en_core_web_sm',
                                         source_directory=os.getcwd(),
                                         compute_target = compute_target)

#as dataset promotes the Pipeline data to Dataset ta
a_without_preprocessed_features =  PipelineData('a_without_preprocessed', datastore = default_datastore).as_dataset()
b_preprocessed_features =  PipelineData('b_preprocessed', datastore = default_datastore).as_dataset()
c_word2vec_features =  PipelineData('c_word2vec_features', datastore = default_datastore).as_dataset()
d_final_features =  OutputFileDatasetConfig(destination=(quora_train_blob_store, 'quora/quora_final_features')).register_on_complete(name="quora_final_features")



features_without_preprocessing = PythonScriptStep(
    script_name="aml_train_feature_1.py",
    arguments=["quora_training",a_without_preprocessed_features],
    inputs=[quora_train_dataset.as_named_input("quora_training")],
    outputs=[a_without_preprocessed_features],
    runconfig = compute_config,
    compute_target=compute_target,
    source_directory=os.getcwd()
)


features_with_preprocessing = PythonScriptStep(
    script_name="aml_train_feature_2.py",
    arguments=["quora_training", b_preprocessed_features],
    inputs=[quora_train_dataset.as_named_input("quora_training")],
    outputs=[b_preprocessed_features],
    runconfig = compute_config,
    compute_target=compute_target,
    source_directory=os.getcwd(),
    allow_reuse=True
)

word2vec_feature_processing = PythonScriptStep(
    script_name="aml_train_feature_3.py",
    arguments=["quora_training", c_word2vec_features],
    inputs=[quora_train_dataset.as_named_input("quora_training")],
    outputs=[c_word2vec_features],
    runconfig = compute_config,
    compute_target=compute_target,
    source_directory=os.getcwd()
)
"""
word2vec_feature_processing_debug = PythonScriptStep(
    script_name="aml_train_feature_3.py",
    arguments=["quora_training", c_word2vec_features,'--remote_debug', '--remote_debug_connection_timeout', 300,'--remote_debug_client_ip','70.106.210.186','--remote_debug_port',5678],
    inputs=[quora_train_dataset.as_named_input("quora_training")],
    outputs=[c_word2vec_features],
    runconfig = compute_config,
    compute_target=compute_target,
    source_directory='./appliedai/quora/training',
    allow_reuse=False
)
"""
final_mashing_steps = PythonScriptStep(
    script_name="aml_train_feature_4.py",
    arguments=["without_preprocess","with_preprocess","word2vec", d_final_features],
    inputs=[a_without_preprocessed_features.as_named_input("without_preprocess"),b_preprocessed_features.as_named_input("with_preprocess"),c_word2vec_features.as_named_input("word2vec")],
    outputs=[d_final_features],
    runconfig = compute_config,
    compute_target=compute_target,
    source_directory=os.getcwd()
)


pipeline1 = Pipeline(workspace=ws, steps=[word2vec_feature_processing])
#pipeline1 = Pipeline(workspace=ws, steps=[features_without_preprocessing,features_with_preprocessing,word2vec_feature_processing,final_mashing_steps])
pipeline_run1 = Experiment(ws, 'arief_quora_experiment').submit(pipeline1)
pipeline_run1.wait_for_completion()


print("done")







