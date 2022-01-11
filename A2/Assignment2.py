import pandas as pd
from sklearn.model_selection import StratifiedKFold 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi
import time
import numpy as np


#This function loads the data from the datafile and converts it to a pandas dataframe
def load_data():
    df = pd.read_csv('data/spambaseHeaders.data', header = 0)
    return df

#This function performs trains and tests using a decision tree and 10 fold cross validation
#Return the accuracy, f-measure and time for each fold
def decision_tree(df):
    skf = StratifiedKFold(n_splits=10, random_state=1)
    results_acc = []
    results_f = []
    results_time= []
    X = df.drop("spam", axis=1)
    y = df["spam"]
    
    for train_index, test_index in skf.split(X, y):
        elapsed_time = 0.0
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        clf = DecisionTreeClassifier()
        s = time.process_time()
        clf = clf.fit(X_train,y_train)
        e = time.process_time()
        elapsed_time += (e-s)
        y_pred = clf.predict(X_test)
        results_acc.append(metrics.accuracy_score(y_test, y_pred))
        results_f.append(metrics.f1_score(y_test, y_pred))
        results_time.append(elapsed_time)

    return results_acc, results_f, results_time

#This function performs trains and tests using naive bayes and 10 fold cross validation
#Return the accuracy, f-measure and time for each fold
def naive_bayes(df):
    skf = StratifiedKFold(n_splits=10, random_state=1)
    results_acc = []
    results_f = []
    results_time = []
    X = df.drop("spam", axis=1)
    y = df["spam"]
    
    for train_index, test_index in skf.split(X, y):
        elapsed_time = 0.0
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        clf = GaussianNB()
        s = time.process_time()
        clf = clf.fit(X_train,y_train)
        e = time.process_time()
        elapsed_time += (e-s)
        y_pred = clf.predict(X_test)
        results_acc.append(metrics.accuracy_score(y_test, y_pred))
        results_f.append(metrics.f1_score(y_test, y_pred))
        results_time.append(elapsed_time)

    return results_acc, results_f, results_time

#This function performs trains and tests using nearest neighbour and 10 fold cross validation
#Return the accuracy, f-measure and time for each fold
def nearest_neighbor(df):
    skf = StratifiedKFold(n_splits=10, random_state=1)
    results_acc = []
    results_f = []
    results_time = []
    X = df.drop("spam", axis=1)
    y = df["spam"]
    
    for train_index, test_index in skf.split(X, y):
        elapsed_time=0
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        clf = KNeighborsClassifier()
        s = time.process_time()
        clf = clf.fit(X_train,y_train)
        e = time.process_time()
        elapsed_time += (e-s)
        y_pred = clf.predict(X_test)
        results_acc.append(metrics.accuracy_score(y_test, y_pred))
        results_f.append(metrics.f1_score(y_test, y_pred))
        results_time.append(elapsed_time)


    return results_acc, results_f, results_time

#This function prints the results for a given measurement (Accuracy, F-Measure, Time) for each algorithm
def print_results(clf1, clf2, clf3):
    dat = {
        "Decision Tree": clf1,
        "K Nearest Neighbor": clf2,
        "Naive Bayes": clf3
        }
    df = pd.DataFrame(dat)
    df.loc["avg"] = df.mean()
    df.loc["stdev"] = df.std()  
    print(df, "\n")

#This function conducts the friedman test, if the result of this test is smaller than 0.05
#Nimenyi analysis will be performed
def friedman_test(d1, d2, d3, time):
    dat = {"Decision tree":d1,
           "K Nearest Neighbor":d2,
           "Naive bayes": d3}
    df = pd.DataFrame(dat)

    if time == True:
        df_ranked = df.rank(axis=1, method="first", ascending= True)
    else:
        df_ranked = df.rank(axis=1, method="first", ascending= False)

    alpha = 0.05
    stats, p = friedmanchisquare(df_ranked.iloc[0:10,0], df_ranked.iloc[0:10,1], df_ranked.iloc[0:10,2])
    
    df_print = pd.DataFrame()
    df_print["Decision tree"] = df["Decision tree"].astype(str).str.cat(df_ranked["Decision tree"].astype(str), sep = " - ")
    df_print["K Nearest Neighbor"] = df["K Nearest Neighbor"].astype(str).str.cat(df_ranked["K Nearest Neighbor"].astype(str), sep = " - ")
    df_print["Naive bayes"] = df["Naive bayes"].astype(str).str.cat(df_ranked["Naive bayes"].astype(str), sep = " - ")

    print("Friedman:")
    print(stats,p)
    print(df_print, "\n")
    if stats < 7.8:
        print("There are no significant differences between algorithms, null hypothesis can not be rejected")
    else:
        print("There are significant differences between algorithms, null hypothesis can be rejected")
        a = [list(df_ranked.iloc[0:10,0]),list(df_ranked.iloc[0:10,1]), list(df_ranked.iloc[0:10,2])]
        print("Nemenyi:")
        print(posthoc_nemenyi(a),"\n")

#The main function calls the necessary funtions for training, testing and comparing the 3 different measurements.
def main():
    data = load_data()
    tree_acc, tree_f, tree_time = decision_tree(data)
    knn_acc,knn_f, knn_time = nearest_neighbor(data)
    gnb_acc, gnb_f, gnb_time = naive_bayes(data)
    
    print("----------ACCURACY----------\n")
    print_results(tree_acc, knn_acc, gnb_acc)
    friedman_test(tree_acc, knn_acc, gnb_acc, False)

    print("----------F-MEASURE----------\n")
    print_results(tree_f, knn_f,gnb_f)
    friedman_test(tree_f, knn_f,gnb_f, False)

    print("----------TIME----------\n")
    print_results(tree_time, knn_time, gnb_time)
    friedman_test(tree_time, knn_time, gnb_time, True)

main()