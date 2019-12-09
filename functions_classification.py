import pandas
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
np.random.seed(42)



def param_selection(X, y, nfolds):
    
    # SVC
    Cs = [0.01, 1, 5]
    #gammas = [0.001]
    kernels = ['linear']
    param_grid = {'C': Cs}
    grid_search = GridSearchCV(SVC(kernel = 'linear', class_weight='balanced'), param_grid, cv=nfolds, scoring='balanced_accuracy')
    grid_search.fit(X, y)
    grid_search.best_params_
    print("Best SVC params: {}".format(grid_search.best_params_))
    
    # LR
    param_grid = {'C': [ 0.01, 0.1, 1] }
    grid_search_lr = GridSearchCV(LogisticRegression(class_weight='balanced', solver='lbfgs', multi_class='auto'), param_grid, cv=nfolds, scoring='balanced_accuracy')
    grid_search_lr.fit(X, y)
    grid_search_lr.best_params_
    print("Best Logistic regression params: {}".format(grid_search_lr.best_params_))
    
    # Random Forest
    estimators = [200,300]
    param_grid_rf = {'n_estimators':estimators}
    grid_search_rf = GridSearchCV(RandomForestClassifier(class_weight='balanced'), param_grid_rf, cv=nfolds, scoring='balanced_accuracy')
    grid_search_rf.fit(X, y)
    grid_search_rf.best_params_
    print("Best RandomForest params: {}".format(grid_search_rf.best_params_))
    
    return grid_search.best_estimator_ , grid_search_lr.best_estimator_ , grid_search_rf.best_estimator_




def plot_models(model_svc, model_lr, model_rf, X, y, score_func):
    models = []
    models.append(('SVC', model_svc))
    models.append(('LR', model_lr))
    models.append(('RF', model_rf))

    results = []
    names = []
    scoring = score_func
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=42)
        cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: mean: %f std: (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    # boxplot algorithm comparison
    fig = plt.figure(figsize=(6,6));
    fig.suptitle('Algorithm Comparison');
    ax = fig.add_subplot(111);
    plt.boxplot(results);
    ax.set_xticklabels(names);
    plt.show();