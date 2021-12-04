# MSCIT-Machine-Learning-Project
Author: Ruxin Qiu, Daci Ma, Fangxu Yuan, Lishan Huang


Component: 
- **clean_data**
- **Classifier**
- **Regressor**
- **train**
- **main**


 
 This project can provide machine learning model to predict the result of 2020 Tokyo Olympic games.
 It firstly filters the country that is predicted to obtain at least one medal by classifier. Then it predicts the number of medals each country can obtain by regressor.
 
 To excute the project, you can download all files in a repository, then run the main function.
 This project contains 3 parts
  1. **clean_data**


      the clean_data python file default to read Olympic game data from 1992 to 2016, and generate csv file for each year.
      and then it can generate the training data from 1996 to 2012 Olympic game and test data from 2016 Olympic game
      there are 13 features for each data
      
      ['Athelete_per',"GDP_norm","population_norm","average_capita_norm","GDP_Growth", "average_capita", "population", "GDP", "Host",'Age', 'Athelete_x','Athelete_y',"Medal_y"]
      
      where medal_y is the medal obtained from previous Olympic game, and Athlete_y is the number of athlete in the previous Olympic game
      
 2. **Classifier**


     There are 4 classifiers in the Classifier class: Logistic, MLPClassifier,  RandomForestClassifier and GaussianNB,
     the method ends with _cv means the model is obtained from gridsearch method
     
 3. **Regressor**


     There are 4 Regressors in the Regressor class: Lasso, Elastic,  RandomForestrRegressor and SVR,
     the method ends with _cv means the model is obtained from gridsearch method
  
  
  The training file can filter the country that is predicted to obtain at least one medal in the Olympic game
  and generate_regressor_result method can provide predict result for the testing year.
  
  

