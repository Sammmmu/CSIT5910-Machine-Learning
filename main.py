from train import *
from Classifier import *
import matplotlib.pyplot as plt
import seaborn as sns

def main():

    print("*******************Classifier result******************")
    model = train()
    model.generate_data_by_classifier(model_type="RandomForest")
    model.generate_data_by_classifier(model_type="MLP")
    model.generate_data_by_classifier(model_type="Gaussian")
    model.generate_data_by_classifier(model_type="Logistic")
    
    print("\n\n\n\n\n*******************Regressor result******************")
    train_x, train_y, test_x, test_y, country = model.generate_data_by_classifier(model_type ="Logistic")
    train_x, train_y, test_x1, test_y1, country1 = data_preprocessing()
    Lasso_predi_result, Lasso_score = model.generate_regressor_result(train_x, train_y, test_x, test_y,country,model_type = "Lasso",cv = True)
    # plot_data(Lasso_predi_result, test_y, "Lasso")
    
    print("\n\n\n***************")
    train_x, train_y, test_x, test_y, country = model.generate_data_by_classifier(model_type ="Logistic")
    train_x, train_y, test_x1, test_y1, country1 = data_preprocessing()
    Random_Forest_predi_result, Random_Forest_score = model.generate_regressor_result(train_x, train_y, test_x, test_y,country,cv = True)
    # plot_data(Random_Forest_predi_result, test_y, "Random Forest")
    
    print("\n\n\n***************")
    train_x, train_y, test_x, test_y, country = model.generate_data_by_classifier(model_type ="Logistic")
    train_x, train_y, test_x1, test_y1, country1 = data_preprocessing()
    SVR_predi_result, SVR_score = model.generate_regressor_result(train_x, train_y, test_x, test_y,country,model_type = "SVR", cv = False)
    # plot_data(SVR_predi_result, test_y, "SVR")
    
    print("\n\n\n***************")
    train_x, train_y, test_x, test_y, country = model.generate_data_by_classifier(model_type ="Logistic")
    train_x, train_y, test_x1, test_y1, country1 = data_preprocessing()
    Elastic_result, Elastic_score = model.generate_regressor_result(train_x, train_y, test_x, test_y,country,model_type = "Elastic",cv = True)
    plot_data(Elastic_result, test_y, "Elastic")
    plot_coeff()
    # plot_combine_result()

def plot_coeff():
    
    plt.figure()
    
    random_forest = [0.18478826,0.05688605,0.01755174, 0.01028271, 0.00717037,0.01408368, 0.03489484, 0.04312658,0.01552939, 0.00562163, 0.20772936, 0.13140758, 0.2709278 ]
    elastic = [ 0, 0, 0, 0, 0.09546002, -0.03164552,0,0.01196964, 0, 0.05860956, 0.02805756, -0.01525213, 0.6869521 ]
    Lasso = [ 0, 0,0, 0,0.10262326,-0.031034,0, 0.00839684,0, 0.06832373,0.02788342,-0.01573728,0.70416855]
    SVR = [0.00721599, -0.00160031, -0.15883072, 0.0369509, 0.00222954, -0.00484106, -0.71103475,0.00903526,0.52514441,0.00959085,0.03192953,-0.01939097,0.69320729]
    
    heap_map = [random_forest, elastic, Lasso, SVR]
    
    plt.imshow(heap_map)
    plt.xlabel("Features",fontsize=11,fontname='Comic Sans MS')
    plt.title("Features selection result from different Model",fontsize=11,fontname='Comic Sans MS')
    plt.yticks(list(range(0,4)),["random_forest","elastic","Lasso","SVR"],fontsize=11,fontname='Comic Sans MS')
    plt.xticks(list(range(0,13)),['AtheN',"GDPN","popN","AvgGDPN","GDP+", "AvgGDP", "POP", "GDP", "Host",'Age', 'Ath','Athy',"Medaly"],fontsize=11,fontname='Comic Sans MS')
    plt.ylabel("Regressor Models",fontsize=11,fontname='Comic Sans MS')
    plt.colorbar()
    plt.show()
    plt.savefig("Features selection result from different Model.png")
    
def plot_data(predict_data, true_data, model_type):
    
    plt.figure()
    
    plt.plot(predict_data, true_data,'o')
    m, b = np.polyfit(predict_data, true_data, 1)
    plt.plot(predict_data, m*predict_data, + b)
    plt.xlabel("Predict Result")
    plt.ylabel("True Result")
    plt.title(model_type + " predict result")
    plt.savefig(model_type + " predict result" + ".png")

def plot_combine_result():
    
    plt.figure()

    pf = pd.DataFrame({
        'random_forest' : [0.8576203528160548],
        'elastic' : [0.9162009711435177],
        'Lasso' : [0.9182583404641756],
        'SVR' : [0.9108735147515485]})
    
    sns.barplot(data=pf, palette='summer')
    plt.xlabel("Model Type")
    plt.ylabel("Predict Accuracy")
    plt.xticks(list(range(0,4)),["random_forest","elastic","Lasso","SVR"],fontsize=11,fontname='Comic Sans MS')
    plt.show()


plt.show()
if __name__ == "__main__":
    main()