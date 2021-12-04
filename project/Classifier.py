import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

class Classifier:
    
    def __init__(self, model_type = "Logistic"):    
        if model_type == "Logistic" or model_type == "LogisticRegression":
            self.model_type = 'LG'
            self.model = LogisticRegression(max_iter=10000)
        
        elif model_type == "MLP" or model_type == "MLPClassifier":
            self.model_type = 'MLP'
            self.model = MLPClassifier(alpha=1, max_iter=1000)
        elif model_type == "RandomForest" or model_type == "RandomForestClassifier":
            self.model_type = "RF"
            self.model = RandomForestClassifier(max_depth=5, n_estimators=20)
        elif model_type == "Gaussian" or model_type == "GaussianNB":
            self.model_type = "GNB"
            self.model = GaussianNB()
        else:
             raise Exception('Model type not exist in classifer class')
        
    def fit(self, train_x, train_y):
        new_train_x = np.copy(train_x)
        new_train_y = np.copy(train_y)
        new_train_x[:,12] =  np.sign(new_train_x[:,12])
        new_train_y =  np.ravel(np.sign(new_train_y))
        self.model.fit(new_train_x, new_train_y)
        
    def predict(self, test_x):
        new_test_x = np.copy(test_x)
        new_test_x[:,12] =  np.sign(new_test_x[:,12])
        return self.model.predict(new_test_x)

    def predict_cv(self, test_x):
        new_test_x = np.copy(test_x)
        new_test_x[:,12] =  np.sign(new_test_x[:,12])
        return self.model_cv.predict(new_test_x)
    
    def score(self, test_x, test_y):
        new_test_x = np.copy(test_x)
        new_test_x[:,12] =  np.sign(new_test_x[:,12])
        new_test_y = np.copy(test_y)
        new_test_y =  np.ravel(np.sign(new_test_y))
        return self.model.score(new_test_x, new_test_y)
    
    def score_cv(self, test_x, test_y):
        new_test_x = np.copy(test_x)
        new_test_y = np.copy(test_y)
        new_test_x[:,12] =  np.sign(new_test_x[:,12])
        new_test_y =  np.sign(new_test_y)
        return self.model_cv.score(new_test_x, new_test_y)    
    
    def fit_cv(self, train_x, train_y):
        x = np.copy(train_x)
        y = np.copy(train_y)
        x[:,12] =  np.sign(x[:,12])
        y =  np.ravel(np.sign(y))

        if self.model_type == "LG":
            params = {'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                      'dual': [True, False],
                      "solver" : ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                      "multi_class" : ['auto', 'ovr', 'multinomial']}
            self.model_cv = GridSearchCV(estimator = self.model,
                                            param_grid=params,
                                            scoring='r2',
                                            max_iter = 1000).fit(x, y)

        elif self.model_type == "GNB":
            params = {'var_smoothing': [1e-9,1e-8,1e-7]}
            
            self.model_cv = GridSearchCV(estimator=self.model, 
                                         param_grid=params, 
                                         scoring='r2').fit(x, y)
            
        elif self.model_type == "RF":
            params = { 'n_estimators' : [50, 100, 200, 300,1000],
                      'max_features' : ["auto", "sqrt", "log2"],
                      'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]}
            
            self.model_cv = GridSearchCV(estimator = self.model, 
                                               param_grid = params, 
                                               scoring='r2').fit(x,y)
        return
