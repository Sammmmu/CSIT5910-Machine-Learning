from Classifier import *
from Regressor import *
from clean_data import *

# def select_country(train_x, train_y, test_x, test_y, Ctype = "RandomForest"):
class train:
    
    def __init__(self): 
        self.classifier = Classifier("RandomForest")
        self.regressor = Regressor("RandomForest")
    
    def generate_data_by_classifier(self, start = 1996, end = 2012, model_type = "RandomForest", cv = False):
        
        self.classifier = Classifier(model_type)
        train_x, train_y, test_x, test_y,country = data_preprocessing(start, end)
        train_x_after_classified = []
        train_y_after_classified = []
        test_x_after_classified = []
        test_y_after_classified = []
        selected_country = []
        k_fold = (end - start - 4) / 4
        size = len(train_y)
        step_size = int (size / k_fold)
        
        for step_start in range(0, size, step_size):
            test_index = list(range(step_start, step_start + step_size))
            train_index = list(range(0,step_start)) + list(range(step_start + step_size, size))
            if cv:
                self.classifier.fit_cv(train_x[np.array(train_index)], train_y[np.array(train_index)])
                predict_index = self.classifier.predict_cv(train_x[test_index])
            else:
                self.classifier.fit(train_x[np.array(train_index)], train_y[np.array(train_index)])
                predict_index = self.classifier.predict(train_x[test_index])
            selected_index = [i for i, e in enumerate(predict_index) if e == 1]
            train_x_after_classified = [*train_x_after_classified, *train_x[selected_index]]
            train_y_after_classified = [*train_y_after_classified, *train_y[selected_index]]

        if cv:
            self.classifier.fit_cv(train_x, train_y)
            predict_test_index = self.classifier.predict_cv(test_x)
        else:
            self.classifier.fit(train_x, train_y)
            predict_test_index = self.classifier.predict(test_x)

        selected__test_index = [i for i, e in enumerate(predict_test_index) if e == 1]
        test_x_after_classified = test_x[selected__test_index]
        test_y_after_classified = test_y[selected__test_index]
        selected_country = country[selected__test_index]
        print(model_type + " classifier accuracy is " + str(self.classifier.score(test_x, test_y)))
        return train_x_after_classified, train_y_after_classified,test_x_after_classified, test_y_after_classified, selected_country

    def generate_regressor_result(self,train_x_after_classified, train_y_after_classified,test_x_after_classified, test_y_after_classified,country, model_type = "RandomForest", cv = False):
        self.regressor = Regressor(model_type)
        if cv:
            self.regressor.fit_cv(train_x_after_classified, train_y_after_classified)
            print(model_type + " regressor accuracy is " + str(self.regressor.score_cv(test_x_after_classified, test_y_after_classified)))
            print("The coefficient is ")
            print(self.regressor.cv_weight)
            score = self.regressor.predict_cv(test_x_after_classified)
            actual_medal = [x for _,x in sorted(zip(score,test_y_after_classified))]
            zip_info = [x for _,x in sorted(zip(score,country))]
            score = sorted(score)
            for i in range(len(score)):
                print(zip_info[i], score[i], actual_medal[i])
            return self.regressor.predict_cv(test_x_after_classified), self.regressor.score_cv(test_x_after_classified, test_y_after_classified)
                
        else:
            self.regressor.fit(train_x_after_classified, train_y_after_classified)
            print(model_type + " regressor accuracy is " + str(self.regressor.score(test_x_after_classified, test_y_after_classified)))
            score = self.regressor.predict(test_x_after_classified)
            actual_medal = [x for _,x in sorted(zip(score,test_y_after_classified))]
            zip_info = [x for _,x in sorted(zip(score,country))]
            score = sorted(score)
            for i in range(len(score)):
                print(zip_info[i], score[i],actual_medal[i])
            return self.regressor.predict(test_x_after_classified), self.regressor.score(test_x_after_classified, test_y_after_classified)
    
    def predict(self,test_x, cv = False):
        
        if cv:
            return self.regressor.predict_cv(test_x)
        else:
            return self.regressor.predict(test_x)
    
    def score(self,test_x,test_y, cv = False):
        if cv:
            return self.regressor.score_cv(test_x,test_y)
        else:
            return self.regressor.score(test_x,test_y)