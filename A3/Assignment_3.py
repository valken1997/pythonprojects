import pandas as pd
import datetime as dt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, median_absolute_error

def load_data():
    df = pd.read_csv("data/data_full.csv", header = 0)
    return df

def prepare_data(df):
    #Convert the numbered date into a datetime
    df["DATE"] = pd.to_datetime(df["DATE"], format='%Y%m%d')

    #Divide temperatures by 10 to show in degrees
    df["MEAN_TEMP"] = df["MEAN_TEMP"]/10
    df["MIN_TEMP"] = df["MIN_TEMP"]/10
    df["MAX_TEMP"] = df["MAX_TEMP"]/10

    #Divide windspeed by 10 to show in m/s
    df["MEAN_WIND_SPEED"] = df["MEAN_WIND_SPEED"]/10
    df["MAX_WIND_SPEED"] = df["MAX_WIND_SPEED"]/10
    df["MIN_WIND_SPEED"] = df["MIN_WIND_SPEED"]/10

    #Divide rainfall by 10 to show in mm
    df["SUM_RAINFALL"] = df["SUM_RAINFALL"]/10

    #for all features, add the features from 4 previous days 
    for col in df.columns:
        if col != "DATE":
            for i in range(1,5):
                get_n_prior_days(col,df,i)

    #Drop all rows with NaN values
    df = df.dropna()

    #Convert all objects to numerics
    df = df.apply(pd.to_numeric, errors="coerce")
    return df

    #For each row, this method creates a new column with the previous n datapoints for all features
    #After this method, each row contains the weather data for 5 days
def get_n_prior_days(feature, df, n):
    rows = df.shape[0]
    m = [None]*n + [df[feature][i-n] for i in range(n, rows)]
    col_name = "{}_{}".format(feature, n)
    df[col_name] = m

    #This method returns the linear correlation regarding a certain attribute for all other attributes
def get_pearson_coefficient(df, attribute):
    return df.corr()[[attribute]].sort_values(attribute)

    #This method returns an array of features that have a linear correlation that is above 0.5
def get_linear_predictors(adf):
    preds = []
    for index, row in adf.iterrows():
        if row["MEAN_TEMP"] > 0.5 and index != "MAX_TEMP" and index != "MIN_TEMP":
            preds.append(index)
    return preds
    
    #This method trains and tests on the dataset using linear regression, after training it calculates the score and error
def linear_regression(df, pre):
    pre.remove("MEAN_TEMP")
    X = df[pre]
    y = df["MEAN_TEMP"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    pred = regressor.predict(X_test)

    print("R^2 Score:", regressor.score(X_test, y_test))
    print("Mean absolute error:", mean_absolute_error(y_test, pred))
    print("Median absolute error:", median_absolute_error(y_test, pred))


def main():
    df = prepare_data(load_data())
    pearson_coefficient = get_pearson_coefficient(df, "MEAN_TEMP")
    predictors = get_linear_predictors(pearson_coefficient)
    lin_df = df[predictors].copy()
    linear_regression(lin_df, predictors)

main()