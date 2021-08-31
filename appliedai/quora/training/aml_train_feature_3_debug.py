from azureml.core import run, runconfig, Run

from azureml.core import Workspace, Experiment, ScriptRunConfig, Datastore, Dataset

import os
import sys
import debugpy
import argparse


from featurize import Featurize


def main():
#------------------------------ added for debugging
    parser= argparse.ArgumentParser()
    parser.add_argument('--remote_debug', action='store_true')
    parser.add_argument('--remote_debug_connection_timeout', type=int,
                    default=300,
                    help=f'Defines how much time the AML compute target '
                    f'will await a connection from a debugger client (VSCODE).')
    parser.add_argument('--remote_debug_client_ip', type=str,
                    help=f'Defines IP Address of VS Code client')
    parser.add_argument('--remote_debug_port', type=int,
                    default=5678,
                    help=f'Defines Port of VS Code client')
    global run
    run = Run.get_context()

    args = parser.parse_args()

    # Start debugger if remote_debug is enabled
    if args.remote_debug:
        print(f'Timeout for debug connection: {args.remote_debug_connection_timeout}')
    # Log the IP and port
    # ip = socket.gethostbyname(socket.gethostname())
        try:
            ip = args.remote_debug_client_ip
        except:
            print("Need to supply IP address for VS Code client")
        print(f'ip_address: {ip}')
        debugpy.listen(address=(ip, args.remote_debug_port))
        # Wait for the timeout for debugger to attach
        debugpy.wait_for_client()
        print(f'Debugger attached = {debugpy.is_client_connected()}')

    
#-----------------------------------------------------------
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
