import os
import lightgbm
import pandas as pd
import glob
import sklearn as sk
import skops.io as sio
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier

'''csv_input = pd.read_csv(r'C:/Users/Ilya/PycharmProjects/NIRB_ML/newnew_fake_fragments_img.csv')
csv_input['Class'] = float(1)
csv_input.to_csv('new_train/newnewf.csv', index = False)
csv_input1 = pd.read_csv(r'C:/Users/Ilya/PycharmProjects/NIRB_ML/newnew_original_fragments_img.csv')
csv_input1['Class'] = 0
csv_input1.to_csv('new_train/newnewor.csv', index = False)
#combine all files in the list
all_filenames = os.listdir(r'C:/Users/Ilya/PycharmProjects/NIRB_ML/new_train')
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
#export to csv
combined_csv.to_csv('new_train/train_haralick.csv', index=False)
df = pd.concat(
    map(pd.read_csv, ['or.csv', 'f.csv']), ignore_index=True)
print(df)'''

fragments = pd.read_csv('new_train/train_train_haralick.csv', sep=',', header=0, low_memory=False)
fragments.head()
test = pd.read_csv('test_fragments_img.csv', sep=',', header=0)
x_t = test
X = fragments.iloc[:,:-1]
y = fragments['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=27)
print('X_test',X_test)
print('y_test',y_test)
#print(x_t)'''
#print(y_test)

##########################


'''
LGBM = LGBMClassifier(n_estimators=30000, objective="binary", class_weight=None)
LGBM.fit(X_train, y_train)
sio.dump(LGBM, 'Haralick_LGBM_model.pkl')

RF = RandomForestClassifier(n_jobs=-1, min_samples_split=2)
RF.fit(X_train, y_train)
sio.dump(RF, 'Haralick_RF_model.pkl')


DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)
AdaBoost = AdaBoostClassifier(base_estimator=DT, algorithm="SAMME.R")
AdaBoost.fit(X_train, y_train)
sio.dump(AdaBoost, 'Haralick_DT_with_AB_model.pkl')

ExtraTree = ExtraTreesClassifier(n_jobs=-1, min_samples_split=2)
ExtraTree.fit(X_train, y_train)
sio.dump(ExtraTree, 'Haralick_ET_model.pkl')

AdaBoost = AdaBoostClassifier(base_estimator=RF, algorithm="SAMME", learning_rate=1)
AdaBoost.fit(X_train, y_train)
sio.dump(AdaBoost, 'Haralick_RF_with_AB_model.pkl')


BaggingClassifier = BaggingClassifier(KNeighborsClassifier(), n_jobs=-1)
BaggingClassifier.fit(X_train, y_train)
sio.dump(BaggingClassifier, 'Haralick_KNN_BC_model.pkl')'''

#########################################

with open('Haralick_DT_with_AB_model.pkl', 'rb') as f:
    AdaBoostDT = sio.load(f, trusted=True)
scoreAdaBoostDT = AdaBoostDT.score(X_test, y_test)
scoreAdaBoostDTTrain = AdaBoostDT.score(X_train, y_train)
print("Score Test DecisionTreeClassifier with AdaBoost:", scoreAdaBoostDT)
print("Score Train DecisionTreeClassifier with AdaBoost:", scoreAdaBoostDTTrain)

with open('Haralick_KNN_BC_model.pkl', 'rb') as f:
    BaggingClassifier = sio.load(f, trusted=True)
scoreBaggingClassifier = BaggingClassifier.score(X_test, y_test)
scoreBaggingClassifierTrain = BaggingClassifier.score(X_train, y_train)
print("Score Test BaggingClassifier with KNeighborsClassifier:", scoreBaggingClassifier)
print("Score Train BaggingClassifier with KNeighborsClassifier:", scoreBaggingClassifierTrain)


with open('Haralick_LGBM_model.pkl', 'rb') as f:
    LGBM = sio.load(f, trusted=True)
scoreLGBM = LGBM.score(X_test, y_test)
scoreLGBMTrain = LGBM.score(X_train, y_train)
print("Score Test LGBMClassifier:", scoreLGBM)
print("Score Train LGBMClassifier:", scoreLGBMTrain)

with open('Haralick_RF_model.pkl', 'rb') as f:
    RF = sio.load(f, trusted=True)
scoreRF = RF.score(X_test, y_test)
scoreRFTrain = RF.score(X_train, y_train)
print("Score Test RandomForestClassifier:", scoreRF)
print("Score Train RandomForestClassifier:", scoreRFTrain)

with open('Haralick_ET_model.pkl', 'rb') as f:
    ExtraTree = sio.load(f, trusted=True)
scoreExtraTree = ExtraTree.score(X_test, y_test)
scoreExtraTreeTrain = ExtraTree.score(X_train, y_train)
print("Score Test ExtraTreeClassifier:", scoreExtraTree)
print("Score Train ExtraTreeClassifier:", scoreExtraTreeTrain)

with open('Haralick_RF_with_AB_model.pkl', 'rb') as f:
    AdaBoostRF = sio.load(f, trusted=True)
scoreAdaBoostRF = AdaBoostRF.score(X_test, y_test)
scoreAdaBoostRFTrain = AdaBoostRF.score(X_train, y_train)
print("Score Test RandomForestClassifier with AdaBoost:", scoreAdaBoostRF)
print("Score Train RandomForestClassifier with AdaBoost:", scoreAdaBoostRFTrain)