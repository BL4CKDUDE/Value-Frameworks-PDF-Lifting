"""
Author: MJ
Copy DateTime: 15 Friday March 2024 18:55
Task: Unifi Zindi PDF Lifting Data Science Competition

Contact: mojela74@gmail.com
"""

import numpy as np
import pandas as pd
import datetime
import time
import re
import os
import sys

#Modelling
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#Evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve

from bisect import bisect_left
import matplotlib.pyplot as plt
import seaborn as sns



class CONSTANTS:
    def __init__(self):
        self.SEARCH_MSG = 1
        self.OVERLAP_MSG = 1
        
    def get_search_msg(self):
        return self.SEARCH_MSG
    
    def get_overlap_msg(self):
        return self.OVERLAP_MSG
    
    def set_overlap_msg(self, item):
        self.OVERLAP_MSG = item
        
    def set_search_msg(self, item):
        self.SEARCH_MSG = item
        
TODAY = str(datetime.datetime.now()).split(".")[0]

constants = CONSTANTS()

def desc():
    functions = ["is_null(item)", "append_to_df(df, row)", "merge(df1, df2, col1, col2, method={‘left’, ‘right’, ‘outer’, ‘inner’, ‘cross’})", "show_all_cols()", "show_all_rows()"
                 "timestamp()","data_scale(dfx)","data_balance_binary(df, col)","count_nans(df, col)", "count_dups(df, col)","lists_identical(list1, list2)", "mi_classif(X,y)", "search_dfa(item, df, col)", "search_dfb(item, df, col)", 
                 "extract_overlap(df, overlap_column, overlap_values)","cut_nulls(df, col_name)", "cut_dups(df, list_columns)","stop()",
                 "strip_string(item, method={spec, specspace, num, alpha})", "update_row(df, index, col, value)","seaborn_dist(df_data, col)",
                 "binary_search_array(item, array)","date_extension()","grade_model_clf(y_true, y_pred, identity)", "join_df_below(df1, df2)", "seaborn_corr_matrix(df_data, annot_size, label_size, outputname)"]
    
    print("This is a toolkit with " + str(len(functions)) + " universal helper functions.\n")
    print("====")
    for f in functions:
        print(f)
    print("====")
    
# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
    
def show_all_cols():
    pd.options.display.max_columns = None
    
def show_all_rows():
   pd.options.display.max_rows = None

def cut_nulls(df, col_name):
    print("Before: "+str(df.shape))
    df_result = df.dropna(subset=[col_name]).copy()
    print("After: "+str(df_result.shape))
    print("Left-Over: "+str(df.shape[0]-df_result.shape[0]))
    return df_result

def cut_dups(df, list_columns):
    print("Before: "+str(df.shape))
    df_result = df.drop_duplicates(subset=[list_columns], keep="first").copy()
    print("After: "+str(df_result.shape))
    print("Left-Over: "+str(df.shape[0]-df_result.shape[0]))
    return df_result

def count_nans(df, col):
    nans = df[col].apply(is_null).sum()
    try:
        nan_ratio = np.round(float(nans/df.shape[0]), 3)
        print("Missing: "+str(nans))
        print(str(nan_ratio*100)+" %")
        return nans
    except:
        print("Message: Division by zero :(")
        
def count_dups(df, col):
    dups = df.duplicated(subset=col, keep='first').sum()  
    try:
        dup_ratio = np.round(float(dups/df.shape[0]), 3)
        print("Duplicates: "+str(dups))
        print(str(dup_ratio*100)+" %")
        return dups
    except:
        print("Message: Division by zero :(")

def strip_string(item, method):
    new_item = ""
    if method == 'specspace':
        new_item = re.sub(r"[^A-za-z0-9 ]", "", item)
    elif method == 'num':
        new_item = re.sub(r"[^A-za-z]", "", item)
    elif method == 'alpha':
        new_item = re.sub(r"[^0-9]", "", item)
    elif method == 'spec':
        new_item = re.sub(r"[^A-za-z0-9]", "", item)
    return new_item

def mi_classif(X,y, n_features):
     mi = mutual_info_classif(X.values, y.values)
     sr_mi = pd.Series(mi)
     sr_mi.index = X.columns
     sr_mi = sr_mi.sort_values(ascending=False)
     print(sr_mi)

     select = SelectKBest(mutual_info_classif, k=n_features)
     select.fit(X, y)
     select_cols = select.get_support()
     cols = X.columns[select_cols]
     return sr_mi , cols
 
def timestamp():
    return time.time()

def merge(df1, df2, col1, col2, method):   
    anchor = "Anchor_tk"
    df1[anchor] = df1[col1]
    df2[anchor] = df2[col2]
    
    df1[anchor] =  df1[anchor].astype(str)
    df2[anchor] =  df2[anchor].astype(str)
    
    df_merge = pd.merge(df1, df2, on=anchor, how=method)
    print("Merge column name: "+str(anchor))
    print("Shape: "+str(df_merge.shape))
    
    if df1.shape[0] == df_merge.shape[0]:
        print("Full match with first DF :)")
    else:
        print("Message: Imperfect merge with first DF :(")
        blockPrint()
        if count_dups(df1, col1) > 0 or count_dups(df2, col2) > 0:
            enablePrint()
            print("Message 2: Possible duplicates in df1 or df2")
        
            
    return df_merge
        
def seaborn_dist_xx_CONSTRUCTION_xx(df_data, col, label_size, outputname):
    ## IN CONSTRUCTION ************
    plt.figure(figsize=(18, 18), dpi=80)
    #sns.set(font_scale=fontsize)
    output = sns.histplot(data=df_data, x=col)
    print(df_data.value_counts([col], ascending=False))
    output.set_xticklabels(output.get_xmajorticklabels(), fontsize = label_size)
    output.set_yticklabels(output.get_ymajorticklabels(), fontsize = label_size)
    plt.savefig(str(outputname)+"_Corr_Matrix.png")
    plt.show()

def seaborn_corr_matrix(df_data, annot_size, label_size, outputname):
    plt.figure(figsize=(18, 18), dpi=80)
    corr_data = df_data.corr()
    output = sns.heatmap(corr_data, cmap="YlGnBu", annot=True, annot_kws={"size": annot_size})
    output.set_xticklabels(output.get_xmajorticklabels(), fontsize = label_size)
    output.set_yticklabels(output.get_ymajorticklabels(), fontsize = label_size)
    plt.savefig(str(outputname)+"_Corr_Matrix.png")
    plt.show()
    
def search_dfa(item, df, col):
    item = str(item)
    result = df.loc[np.where((df[col].astype(str).values == item))]
    if constants.SEARCH_MSG == 1:
        print("Message: SEARCH_DFB (slightly slower than SEARCH_DFA): Make sure the item you are searching for is of type STRING and the col is of datatype OBJECT(string)")
        constants.set_search_msg(0) 
    return result

def search_dfb(item, df, col):
    item = str(item)
    result = df.query(col + ' == @item')
    if constants.SEARCH_MSG == 1:
        print("Message: SEARCH_DFB (slightly quicker than SEARCH_DFA): Make sure the item you are searching for is of type STRING and the col is of datatype OBJECT(string).")
        constants.set_search_msg(0) 
    return result

def grade_model_clf(y_true, y_pred, identity):
    print(str(identity))
    print("\n")
    acc = accuracy_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print(str(identity)+" Accuracy: "+str(acc))
    print(str(identity)+" MSE: "+str(mse))
    print(str(identity)+" F1 Score: "+str(f1))
    print(str(identity)+" Confusion Matrix:")
    
    cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0, 1])
    cm_display.plot()
    plt.show()
    
    print("\n")
    print("ROC")
    fpr1, tpr1, thresh1 = roc_curve(y_true, y_pred, pos_label=1)
    plt.plot(fpr1, tpr1, linestyle='--',color='orange', label= identity)
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.show()
    
    return acc, f1, mse
    
def data_scale(dfx):
    min_max_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()
    dfx_min_max = min_max_scaler.fit_transform(dfx)
    dfx_standard = standard_scaler.fit_transform(dfx)
    print("Min Max Shape: "+str(dfx_min_max.shape))
    print("Standardize Shape: "+str(dfx_standard.shape))
    return dfx_min_max, dfx_standard
    
def data_balance_binary(df, col):
    classes = list(set(df[col]))
    df[col] = df[col].astype(str).copy()
    if len(classes) == 2:
        classes = [str(x) for x in classes]
        class_a = classes[0]
        class_b = classes[1]
        class_a_count = df.value_counts(col)[class_a]
        class_b_count = df.value_counts(col)[class_b]
        print(df.value_counts(col))
        print(class_a+" %: " + str(np.round(float(class_a_count/df.shape[0])*100, 2)))
        print(class_b+" %: " + str(np.round(float(class_b_count/df.shape[0])*100, 2)))
    else:
        print("Column =="+col+"== is not binary. :(")
    
def append_to_df(df, row):
    row = dict(row)
    df_dict = pd.DataFrame([row])
    return pd.concat([df, df_dict], ignore_index=True)

def extract_overlap(df, overlap_column, overlap_values):
    df_ovlp = df[df[overlap_column].isin(overlap_values)]
    print("Shape: "+str(df_ovlp.shape))
    if constants.OVERLAP_MSG == 1:
        print("Message: EXTRACT OVERLAP: This is an independent overlap extraction, not an inner merger.")
        constants.set_overlap_msg(0)
    return df_ovlp

def join_df_below(df1, df2):
    df_join = pd.concat([df1, df2], ignore_index=True)
    print("Shape: "+str(df_join.shape))
    return df_join

def is_null(item):
    item = str(item).lower()
    item = item.replace(" ", "")
    if item == '""':
        return True
    if item == "''":
        return True
    if item == "":
        return True
    if item == np.nan:
        return True
    if item == "nan":
        return True
    if item == "null":
        return True
    if pd.isnull(item):
        return True
    if item == None:
        return True
    return False

def update_row(df, index, col, value):
    df[col].iloc[index] = value
    return df

def binary_search_array(item, array):
    i = bisect_left(array, item)
    if i != len(array) and array[i] == item:
        # Return Index
        return i
    else:
        # Not Found
        return -1
    
def stop():
    raise Exception("Message: Code has been stopped on purpose via tk_essentials.")
    
def lists_identical(list1, list2):
    len1 = len(list1)
    len2 = len(list2)
    
    compare1 = len(set(list1))
    compare2 = len(set(list2))
    
    if len1 == len2 and compare1 == compare2:
        return True
    else:
        return False                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               

def date_extension():
    return TODAY.split(" ")[0].replace("-", "")




#Testing
