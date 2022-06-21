import argparse
import os
import pandas as pd
import mlflow
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def select_first_file(path):
    """Selects first file in folder, use under assumption there is only one file in folder
    Args:
        path (str): path to directory or file to choose
    Returns:
        str: full path of selected file
    """
    files = os.listdir(path)
    return os.path.join(path, files[0])

# Start Logging
mlflow.start_run()

# enable autologging
mlflow.sklearn.autolog()

os.makedirs("./outputs", exist_ok=True)

def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="path to train data")
    parser.add_argument("--test_data", type=str, help="path to test data")
    parser.add_argument("--n_estimators", required=False, default=100, type=int)
    parser.add_argument("--learning_rate", required=False, default=0.1, type=float)
    parser.add_argument("--registered_model_name", type=str, help="model name")
    parser.add_argument("--model", type=str, help="path to model file")
    args = parser.parse_args()

    train_df =  pd.read_csv(select_first_file(args.train_data))
    train_label=train_df["Price"]
    train_df=train_df.drop(columns=['Price'])
    
    test_df = pd.read_csv(select_first_file(args.test_data))
    test_label=test_df["Price"]
    test_df=test_df.drop(columns=['Price'])

    #convert into array
    X_train=train.df.values
    y_train=train_label.values

    X_test=test_df.values
    y_test=test_label.values

    #fit the model
    lr=LinearRegression()
    lr.fit(X_train,y_train)

    y_pred=lr.predict(X_test)
    print(r2_score(y_test,y_pred))
    
    #register the model to the workspace
    print("registering the model")
    mlflow.sklearn.log_model(
        sk_model=lr,
        registered_model_name=args.registered_model_name,
        artifact_path=args.registered_model_name,
    )

    #saving the model
    mlflow.sklearn.save_model(
        sk_model=lr,
        path=os.path.join(args.model,"trained_model"),
    )

    #stop logging
    mlflow.end_run()

if __name__ == "__main__":
    main()








