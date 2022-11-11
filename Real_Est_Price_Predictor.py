import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from joblib import dump


# Read dataset:

data= pd.read_csv("./Training Data/real_est.csv")

# Dropping sold_date:
data=data.drop("sold_date",axis=1)

# Dropping data where price is not defined:
data=data[pd.notnull(data["price"])]

# Converting non-numeric data to numeric:
def handle_non_numeric_data(df):
    columns= df.columns.values

    # For each column check if dtype is non-numeric and update with numeric data:
    for column in columns:
        # Init a dict to map vals to numerics:
        txt_dig_val={}
        def convert_to_int(val):
            return txt_dig_val[val]

        # Check if the values are not int and float:
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            col_data=df[column].values.tolist()
            uniq_data=set(col_data)

            x=0
            # Create a corresponding numeric value for each uniq element:
            for uniq in uniq_data:
                if uniq not in txt_dig_val:
                    txt_dig_val[uniq]=x
                    x+=1

            # Map the column values to its corresponding numeric value:
            df[column]=list(map(convert_to_int,df[column]))

    return df

data=handle_non_numeric_data(data)

# Train- Test Splitting:
train_set,test_set=train_test_split(data,test_size=0.2,random_state=42)

# Drop the lables and save it:
train_data=train_set.drop("price",axis=1)
train_labels=train_set["price"].copy()

# Handling missing data and feature scaling:

# Creating a pipeline:
est_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ('std_scaler',StandardScaler())
])

# Applying the pipeline to train set
train_features=est_pipeline.fit_transform(train_data)

# Training the model:

model=LinearRegression()
# model=DecisionTreeRegressor()
# model=RandomForestRegressor()

model.fit(train_features,train_labels)

# Evaluating the model:

scores= cross_val_score(model,train_features,train_labels,scoring="neg_mean_squared_error")
rmse_scores=np.sqrt(-scores)

def print_scores(scores):
    print("Scores:", scores)
    print("Mean:",scores.mean())
    print("Standard deviation:",scores.std())

print_scores(rmse_scores)

# Testing the model on test dataset:

# Drop the lables and save it:
test_data=test_set.drop("price",axis=1)
test_labels=test_set["price"].copy()

# Applying the pipeline on test data:
test_features=est_pipeline.fit_transform(test_data)
print(list(test_features[0]))

# test Predictions:

test_predictions=model.predict(test_features)
test_mse=mean_squared_error(test_labels,test_predictions)
test_rmse=np.sqrt(test_mse)

print("Test Results (RMSE): ", test_rmse)

dump(model, 'RealEstPricePredictor.joblib') 