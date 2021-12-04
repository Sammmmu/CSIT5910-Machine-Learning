import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from warnings import filterwarnings
class Regressor:
    
    def __init__(self, model = "Lasso"):
        self.model_type = model
        if self.model_type == "Elastic":
            self.model = ElasticNet()
        elif self.model_type == "Lasso":
            self.model = Lasso()
        elif self.model_type == "SVR":
            self.model = SVR(C=0.21,
                                kernel='linear',
                                cache_size=150,
                                max_iter=-1,
                                epsilon=0.1)
        elif self.model_type == "RandomForest":
            self.model = RandomForestRegressor()
        else:
            raise Exception("Model type doest no exist")
    
    def fit(self, train_x, train_y):
        train_y = np.ravel(train_y)
        self.model.fit(train_x, train_y)
    
    def score_cv(self, test_x, test_y):
        test_y = np.ravel(test_y)
        return self.model_cv.score(test_x, test_y)
        
    def score(self, test_x, test_y):
        test_y = np.ravel(test_y)
        return self.model.score(test_x, np.ravel(test_y))

    def fit_cv(self, train_x, train_y):
        filterwarnings('ignore')
        train_y = np.ravel(train_y)
            
        if self.model_type == "Elastic":
            params = {'l1_ratio' :[0,0.1, 0.5, 0.7, 0.9, 0.95, 1],
                       'alpha' : [0.1,0.01,1,10,100],
                       'normalize' : [True, False], 
                       'max_iter' : [1000,5000],
                       'positive' :[True, False]}
            
            self.model_cv = GridSearchCV(estimator = self.model, 
                                            param_grid = params,
                                            scoring = 'r2').fit(train_x, train_y)
            self.cv_weight = self.model_cv.best_estimator_.coef_
            return
        
        if self.model_type == "Lasso":
            
            params = { 'alpha' : [0.1,0.01,1,10,100],
                       'normalize' : [True, False], 
                       'max_iter' : [1000,5000],
                       'positive' :[True, False]}
            
            self.model_cv = GridSearchCV(estimator = self.model, 
                                            param_grid = params,
                                            n_jobs = -1,
                                            scoring = 'r2').fit(train_x, train_y)
            self.cv_weight = self.model_cv.best_estimator_.coef_
            return 
        
        if self.model_type == "SVR":
            filterwarnings('ignore')
            params = { 'kernel': ['linear', 'poly', 'rbf'],
                      'C' : np.logspace(-2,2,num=20),
                      'degree' : [1,2,3,4],
                      'gamma' : ['scale', 'auto']}
    
            self.model_cv = GridSearchCV(estimator = self.model, 
                                            param_grid = params,
                                            scoring = 'r2').fit(train_x, train_y)
            self.cv_weight = self.model_cv.best_estimator_.coef_
            return
        
        if self.model_type == "RandomForest":
            
            params = { 'n_estimators' : [50, 100, 200, 300,1000],
                      'max_features' : ["auto", "sqrt", "log2"],
                      'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]
                    }
    
            self.model_cv = GridSearchCV(estimator = self.model, 
                                            param_grid = params,
                                            scoring = 'r2').fit(train_x, train_y)
            self.cv_weight = self.model_cv.best_estimator_.feature_importances_
            return
        
    def predict(self, test_x):
        return self.model.predict(test_x);
    
    def predict_cv(self, test_x):
        return self.model_cv.predict(test_x)