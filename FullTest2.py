import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from imblearn.over_sampling import ADASYN,SMOTE

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# Import the data
data = pd.read_csv("norm_data__non_log.txt",sep='\t').T
label = pd.read_csv("sample_list.csv",sep=';')

# Log transform on the data, keep both datasets. Use log for logistic and the features 
# for the booster or SVM
data_log = data.apply(np.log).values

# Conversion of string to bool
mapping = {'Non-LCa':0,'LCa':1}
target = label.Disease.map(mapping).values


clf = SVC(class_weight='balanced')

param_dist = {"C": [np.random.randint(1,1e5,size=1)],
              'gamma':[1,0.1,0.01,0.001,0.0001,1e-4,1e-5,1e-6,1e-7],
              'kernel':['rbf']}

# run randomized search
n_iter_search = 100
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search, cv=5, scoring='precision',iid=False)
random_search.fit(data_log,target)
res = pd.DataFrame(random_search.cv_results_)
print(res[['param_C','param_gamma','mean_test_score','rank_test_score']])


