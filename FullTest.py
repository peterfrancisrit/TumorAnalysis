# Packages
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

from sklearn.model_selection import GridSearchCV

class EstimatorSelectionHelper:

    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, cv=3, n_jobs=3, verbose=1, scoring=None, refit=False):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring, refit=refit,
                              return_train_score=True)
            gs.fit(X,y)
            self.grid_searches[key] = gs    

    def score_summary(self, sort_by='mean_score'):
        def row(key, scores, params):
            d = {
                 'estimator': key,
                 'min_score': min(scores),
                 'max_score': max(scores),
                 'mean_score': np.mean(scores),
                 'std_score': np.std(scores),
            }
            return pd.Series({**params,**d})

        rows = []
        for k in self.grid_searches:
            print(k)
            params = self.grid_searches[k].cv_results_['params']
            scores = []
            for i in range(self.grid_searches[k].cv):
                key = "split{}_test_score".format(i)
                r = self.grid_searches[k].cv_results_[key]        
                scores.append(r.reshape(len(params),1))

            all_scores = np.hstack(scores)
            for p, s in zip(params,all_scores):
                rows.append((row(k, s, p)))

        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]

        df[columns].to_csv('RESULTS_OF_FULL_MODELS_FDR09.csv')
        return df[columns]
# Import the data
data = pd.read_csv("norm_data__non_log.txt",sep='\t').T
label = pd.read_csv("sample_list.csv",sep=';')

# Log transform on the data, keep both datasets. Use log for logistic and the features 
# for the booster or SVM
data_log = data.apply(np.log).values

# Conversion of string to bool
mapping = {'Non-LCa':0,'LCa':1}
target = label.Disease.map(mapping).values


# For other models, normalise the non-log data only used in LR
# scaler = StandardScaler().fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)


# Logistic regression, fitting intercept automatically. No standardisation needed
model_lr = LogisticRegression(class_weight='balanced',solver='liblinear')
# model_svm = SVC(class_weight='balanced').fit(X_train,y_train)
# model_rf = RandomForestClassifier(class_weight='balanced').fit(X_train,y_train)

# Prediction of the models

# y_pred_svm = model_svm.predict(X_test)
# y_pred_rf = model_rf.predict(X_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFdr, chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_selection import SelectFdr, chi2, SelectFromModel,f_classif,SelectKBest,mutual_info_classif
from sklearn.feature_selection import SelectPercentile, chi2

X_new = SelectPercentile(mutual_info_classif,percentile=40).fit_transform(data_log,target)
X_new_non = SelectPercentile(mutual_info_classif,percentile=40).fit_transform(data,target)

X_new = TSNE

print(X_new.shape)
scoring = {'AUC': 'roc_auc', 'Accuracy':'accuracy','Recall':'recall','Precision':'precision'}
models1 = {
	'LR': LogisticRegression(class_weight='balanced'),
	'NN': KNeighborsClassifier(),
    'SVC': SVC(class_weight='balanced'),
    'RF': RandomForestClassifier(class_weight='balanced'),
     'GB': GradientBoostingClassifier(),
     'NB': GaussianNB(),
     'BAG': BaggingClassifier()
}

params1 = {
	'GB': {'max_features': ['sqrt','log2',None], 'learning_rate':[0.1,0.001,0.0001], 'n_estimators':[10,20,30,40,50,100]},
	'LR': {'C':[1,10,100]},
	'NN': {'weights':['uniform','distance'],'n_neighbors':[5,10,20,50]},
	'NB': {'priors':[None]},
	'BAG': {'n_estimators':[10,15,20,25,30,60,100],'max_features':[1,0.9,0.8,0.5,0.2,0.1]},
	'RF': {'n_estimators':[10,15,20,25,30,60,100]},
    'SVC': [{'kernel':['poly'], 'C':[1,10,100,1000,1e4],'gamma': [0.01,0.001, 0.0001,0.00001,1e-6,1e-7,1e-8]},
        {'kernel': ['rbf'], 'C': [1, 10,100,1000,1e4], 'gamma': [0.01,0.001, 0.0001,0.00001,1e-6,1e-7,1e-8]}
        ]
}

helper1 = EstimatorSelectionHelper(models1, params1)
helper1.fit(X_new, target, scoring="f1", n_jobs=-1)

print(helper1.score_summary(sort_by='max_score'))

# helper2 = EstimatorSelectionHelper(models1,params1)
# helper2.fit(X_new_non,target,scoring='f1',n_jobs=-1)






































