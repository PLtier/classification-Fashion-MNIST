import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd


#Import the ML models
import sys
import os
module_path = os.path.abspath(os.path.join('..', 'models'))
sys.path.append(module_path)
import template_matching

MODELS = [template_matching.template_matching]  
data_raw = np.load(r"..\data\raw\fashion_train.npy")
# data_img = [np.reshape(image[:784],(28,28)) for image in data_raw]

#Standardize the data
row_scaled_data = (data_raw[:,:-1] - np.mean(data_raw[:,:-1],axis=1)[:,np.newaxis]) / np.std(data_raw[:,:-1],axis=1)[:,np.newaxis]
data_scaled = np.column_stack((row_scaled_data,data_raw[:,-1]))


# Helper function
def get_scores_main(mean_conf_matrix):
    TP = np.diag(mean_conf_matrix)
    FP = np.sum(mean_conf_matrix,axis=0)-TP
    FN = np.sum(mean_conf_matrix,axis=1)-TP
    accuracy = sum(TP)/sum(TP+FP)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    F1 = 2*((precision*recall)/(precision+recall))
    return [accuracy, precision, recall, F1]
    
# get_scores(mean_conf_matrix)

def info(data, MODELS):
    NUMBER_OF_CLASSES = 5
    kf = KFold(n_splits=5, shuffle=True)
    kf.get_n_splits(data)

    for model in MODELS:
        print(f"Model {model.__name__}")
        cms = []
        all_scores = []
        
        for fold_id,(train_idx, test_idx) in enumerate(kf.split(data)):
            y_test, y_pred = model(train_idx, test_idx, data, NUMBER_OF_CLASSES,cms,all_scores)
            
            matrix = confusion_matrix(y_test, y_pred)
            
            cms.append(matrix)
            mean_conf_matrix = np.mean(cms,axis=0)
            var_conf_matrix = np.var(cms,axis=0)
            
            fold_scores = get_scores_main(matrix)
            all_scores.append(fold_scores)
            
        print("Mean confusion matrix\n", mean_conf_matrix)
        print()
        print("Variance of mean confusion matrix\n", var_conf_matrix)

        print("\nAccuracy:", round(np.mean([all_scores[i][0] for i in range(5)],axis=0),4))
        print("Std Accuracy:", round(np.std([all_scores[i][0] for i in range(5)]),4))
        print()
        print("Precision:",list(map(lambda x:round(float(x),4),np.mean([all_scores[i][1] for i in range(5)],axis=0))))
        print("Std Precision:",list(map(lambda x:round(float(x),4),np.std([all_scores[i][1] for i in range(5)],axis=0))))
        print()
        print("Recall:",list(map(lambda x:round(float(x),4),np.mean([all_scores[i][2] for i in range(5)],axis=0))))
        print("Std Recall:",list(map(lambda x:round(float(x),4),np.std([all_scores[i][2] for i in range(5)],axis=0))))
        print()
        print("F1:",list(map(lambda x:round(float(x),4),np.mean([all_scores[i][3] for i in range(5)],axis=0))))
        print("Std F1:",list(map(lambda x:round(float(x),4),np.std([all_scores[i][3] for i in range(5)],axis=0))))

info(data_scaled, MODELS)