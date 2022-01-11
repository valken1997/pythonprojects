import pandas as pd
from sklearn.model_selection import train_test_split

#this collection contains all attributes that will be treated as boolean values
words = ["word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d", "word_freq_our","word_freq_over","word_freq_remove","word_freq_internet","word_freq_order", 
              "word_freq_mail","word_freq_receive","word_freq_will","word_freq_people","word_freq_report","word_freq_addresses","word_freq_free","word_freq_business","word_freq_email",
              "word_freq_you","word_freq_credit","word_freq_your","word_freq_font","word_freq_000","word_freq_money","word_freq_hp","word_freq_hpl","word_freq_george","word_freq_650",
              "word_freq_lab","word_freq_labs","word_freq_telnet","word_freq_857","word_freq_data","word_freq_415","word_freq_85","word_freq_technology","word_freq_1999","word_freq_parts",
              "word_freq_pm","word_freq_direct","word_freq_cs","word_freq_meeting","word_freq_original","word_freq_project", "word_freq_re","word_freq_edu","word_freq_table","word_freq_conference",
              "char_freq_;", "char_freq_(","char_freq_[","char_freq_!","char_freq_$","char_freq_#"]

#this collection contains all attributes that will be binned
lengths = ["capital_run_length_average","capital_run_length_longest","capital_run_length_total"]

def load_data():
    data = pd.read_csv('data/spambaseHeaders.data', header = 0)
    return data

def prepare_data(df):
    #Extract training data from all data, 70% of all 'spam == 1' values
    spam_df = df[df.spam != 0]
    td, test_df = train_test_split(spam_df,test_size=0.3, random_state=21)
    train_data = td.copy(deep=True)

    #Test data contains 30% of all 'spam == 1' and 100% of 'spam == 0' values
    test_data = test_df.append(df[df.spam != 1])
    for column in train_data.columns:
        if column in words:
            #All values that are in the "words" collection that are bigger than 0 will be replaced with 1
            mask = train_data[column] == 0
            train_data.loc[mask, column] = 0
            mask = train_data[column] > 0
            train_data.loc[mask, column] = 1
        elif column in lengths:
            #All values that are in the "lengths" collection will be binned into 4 bins with an equal amount of data points
            train_data[column] = pd.qcut(train_data[column],4, duplicates="drop")
    return train_data, test_data

#This function compares a data instance to the model, and returns all attributes and values that are equal
def find_least_general_conjunction(dict1, dict2):
    dict3 = dict()
    for x in dict2:
        if x in dict1 and x != "spam":
                #If the instance (dict2) contains an attribute that exists in the model (dict1)
                #And the corresponding values are equal, this attribute and value will be added to 
                #dict3, which represents the least general conjunction up to this point
            if dict2[x] == 0 and dict2[x]==dict1[x]:
                dict3[x] = dict2[x] 

    return dict3

#This function loops trough all data instances and compares them to eachother to find the 
#least general conjunction using the find_least_general_conjunction function
def find_least_general_generalisation(d):
    x = d[0]
    H = x
    for i in d:
        #The method stops trying to find a more general model if the model consists of 5 or less elements,
        #This is done to make sure the model won't return empty.
        if len(H) > 5:
            x = i
            H = find_least_general_conjunction(H, x)
    return H
    
#This method classifies an instance. By comparing all attributes to the model,
#it determines whether an instance can be classified as spam or not.
def classify_instance(instance, model):
    l = len(model)
    cnt = 0
    for i in instance:
        if i in words:
            if i in model and instance[i] == model[i]:
                cnt+=1
        elif i in lengths:
            if i in model and instance[i] in model[i]:
                cnt+=1
    if cnt == l:
        #instance has all attributes of a spam email
        return True
    else:
        #instance lacks some attributes of spam
        return False


def calculate_hypothesis_space():
    a = 4**(len(lengths)) + 2**(len(words))
    print("Hypothesis space: 2^", a)

def calculate_no_concepts():
    x = 3**(len(words))
    y = 5**(len(lengths))
    print("Number of concepts: ", x+y)

def calculate_matrix(TP, FN, FP,TN):
    recall = TP/(TP+FN)
    print("Recall ",recall)

    precision = TP/(TP+FP)
    print("Precision: ",precision)

    f_measure = (2*recall*precision)/(recall+precision)
    print("F-measure: ", f_measure)

def main():
    calculate_no_concepts()
    calculate_hypothesis_space()
    df = load_data()
    train_data, test_data = prepare_data(df)
   
    dict_data = train_data.to_dict(orient="records")
    
    m = find_least_general_generalisation(dict_data)

    TP = 0
    FN = 0
    FP = 0
    TN = 0

    dict_test = test_data.to_dict(orient="records")
    i = 0
    for x in dict_test:
        if x["spam"] == 1:
            if classify_instance(x, m):
                #True Positive, predicted 1 & actual 1
                TP+=1
            else:
                #False Negative, predicted 0 & actual 1
                FN+=1
        else:
            if classify_instance(x, m):
                #False Positive, predicted 1 & actual 0
                FP+=1
            else:
                #True Negative, predicted 0 & actual 0
                TN+=1

    print("TP", TP)
    print("FP", FP)
    print("TN", TN)
    print("FN", FN)
    calculate_matrix(TP, FN, FP, TN)
    print("Model: ", m)


main()



