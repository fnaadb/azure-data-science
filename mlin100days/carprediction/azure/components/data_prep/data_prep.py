
import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import mlflow
import seaborn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

def main():
    """Main function of the script
    got it"""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--test_train_ratio", type=float, help="split ratio", required=False, default=0.2)
    parser.add_argument("--train_data", type=str, help="path to train data")
    parser.add_argument("--test_data", type=str, help="path to test data")

    args = parser.parse_args()

    #start logging
    mlflow.start_run()
    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))
    print("input data:",args.data)


    car= pd.read_csv(args.data,header=1, index_col=0)
    print("initial size data", car.shape)
    
    #remove any price greater than 6000000
    car=car[car['Price']<6000000]
    print("Size after removing rows with price greater than 6000000 ", car.shape)
     
    mlflow.log_metric("initial_num_samples", car.shape[0])
    mlflow.log_metric("initial_num_features", car.shape[1] - 1)

    X=car[['name','company','year','kms_driven','fuel_type']]
    y=car['Price'] 

    #train test split
    X_train,X_test,y_train,y_test=train_test_split(car,y, test_size=args.test_train_ratio)
   
    #get all the categories from all the data
    ohe=OneHotEncoder()
    ohe.fit(car[['name','company','fuel_type']])

    #make column transformer for all the categorical variables.
    column_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']), remainder='passthrough')
    # fit transform and convert from sparse array
    X_train =column_trans.fit_transform(X_train).toarray()
    X_test = column_trans.fit_transform(X_test).toarray()
    
    #convert to dataframe and merge with price, please convert the series into dataframe by using to_frame() function
    train_df=pd.DataFrame(X_train)
    train_df.merge(y_train.to_frame(), left_index=True, right_index=True)
    test_df =pd.DataFrame(X_test)
    test_df.merge(y_test.to_frame(), left_index=True, right_index=True)

    #log the metrics
    mlflow.log_metric("TRANSFORMED_TRAIN_num_samples", train_df.shape[0])
    mlflow.log_metric("TRANSFORMED_TRAIN_num_features", train_df.shape[1] - 1)

    mlflow.log_metric("TRANSFORMED_TEST_num_samples", test_df.shape[0])
    mlflow.log_metric("TRANSFORMED_TEST_num_features", test_df.shape[1] - 1)

    # output paths are mounted as folder, therefore, we are adding a filename to the path
    train_df.to_csv(os.path.join(args.train_data, "data.csv"), index=False)
    test_df.to_csv(os.path.join(args.test_data, "data.csv"), index=False)

    mlflow.end_run()


if __name__ == "__main__":
    main()







