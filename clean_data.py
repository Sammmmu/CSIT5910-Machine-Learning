import csv  
import pandas as pd
import numpy as np

"""

This File provides a method to read Olympic history data and NOC_regions data
then generate a clean data that will be trained in the future

"""
GDP_growth_File = "data_set/GDP_growth.csv"
GDP_Per_Capita_File = "data_set/GDP_per_capita.csv"
population_File = "data_set/population.csv"
Host_dict = {1992:"ESP", 1996:"USA", 2000:"AUS", 2004:"GRE",2008:"CHN", 2012:"GBR", 2016:"BRA", 2020:"JPN"}

def data_map(input_file):
    result = dict()
    try:
        with open(input_file) as csv_file:  
            csv_reader = csv.reader(csv_file, delimiter = ',')
            count = 0
            for row in csv_reader:
                if count == 0:
                    count+=1
                    continue
                result[row[0]] = dict()
                for i in range(1989, 2021):

                    result[row[0]][i] = row[i - 1988]
    except:
        print("The input file " + input_file + " can not be opened.")
    return result

def clean_data(input_file, output_file, map_name, choice_sex,start=1992,end=2012):
    
    """[summary]

    Args:
        input_file ([String]): [The location and name of input data file]
        output_file ([String]): [The location and name of output file]
        map_name ([type]): [noc and origions mapping file name]
        choice_sex ([String]): [ select data by Athlete sex,  M for male, F for femal, T for all people]
    """

    """
        This part is going to read the noc_regions file and create a map to store all map pair of noc - region
    """
    gpd_growth_map = data_map(GDP_growth_File)
    gpd_per_map = data_map(GDP_Per_Capita_File)
    population_map = data_map(population_File)
    noc_name = dict()
    try:
        with open(map_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter = ',')
            for row in csv_reader:
                value = row[1: len(row)]
                value = ' '.join(value).strip().strip(',')
                noc_name[row[0]] = value
    except IOError:
        print("NOC name mapping file" + map_name + "does not exist")

    """
        This part will read the input data file,
        then combine the athletes data by country or origins, and save all data in a dictionary,
        where the key is NOC, value is  a list of 'Age', 'Medal', 'Gold' and 'Total Athlete
    """
    for curr_year in range(start,end + 4,4):
        total_data= dict()
        try:
            with open(input_file) as csv_file:
                
                csv_reader = csv.reader(csv_file, delimiter = ',')
                line_count = 0;
                for row in csv_reader:
                    if line_count == 0:
                        line_count +=1
                        continue
                    
                    age = row[3].strip()
                    team = row[7].strip()
                    season = row[10].strip()
                    medal = row[14].strip()
                    year = row[9].strip()
                    sex = row[2].strip()
                    year = int(row[9].strip())
                    item = row[13].strip()
                    gold = 0
                    silver = 0
                    bronze = 0
                    if (sex == choice_sex or choice_sex == "T" ) and (year == curr_year):
                        if season == "Summer" and age != 'NA':
                            
                            if medal == "NA":
                                medal_numb = 0
                            else:
                                if medal == "Gold":
                                    gold = 1
                                elif medal == "Bronze":
                                    bronze = 1
                                elif medal == "Silver":
                                    silver = 1
                                
                            Gold_team_info = set()
                            Silver_team_info = set()
                            Brozen_team_info = set()
                            
                            
                            if team not in total_data:
                                info = [int(age),1,gold,silver,bronze,Gold_team_info,Silver_team_info,Brozen_team_info] 
                                if gold == 1 :
                                    Gold_team_info.add(item)
                                if Silver_team_info == 1 :
                                    Silver_team_info.add(item)   
                                if Brozen_team_info == 1 :
                                    Brozen_team_info.add(item)
                                total_data[team] = info
                                
                            else:
                                for i in range(2):
                                    total_data[team][0] += int(age)
                                    total_data[team][1] += 1
                                if item not in total_data[team][5] and gold != 0:
                                    total_data[team][2] += gold
                                    total_data[team][5].add(item)
                                elif item not in total_data[team][6] and silver != 0:
                                    total_data[team][3] += silver
                                    total_data[team][6].add(item)
                                elif item not in total_data[team][7] and bronze != 0:
                                    total_data[team][4] += bronze
                                    total_data[team][7].add(item)

        except IOError:
            print("Input file " + input_file +"does not exist")
            return
        
        """
            This part is going to save the dictionary data in the target output file
            each column is for 'Country','NOC','Age', 'Medal', 'Gold', 'Total Athlete' respectively 
        """ 
        
        try:
            with open( str(curr_year) +"_Total_clean_data.csv" , 'w+') as f:

                header = ['Country/Region','NOC', "GDP_Growth", "average_capita", "population", "GDP", "Host",'Age', 'Athlete','Gold','Medal']
                writer  = csv.writer(f)
                writer.writerow(header)
                total_data = {k: v for k,v in sorted(total_data.items(), key=lambda item: item[0])} 
            
                for key,value in total_data.items():
                    
                    value[0] = round(value[0] / value[1], 2)
                    gdp_growth = 0
                    average_gdp = 0
                    population = 0
                    host = 0
                    
                    for i in range(curr_year - 3, curr_year+1):
                        if key not in gpd_growth_map or gpd_growth_map[key][i] == "..":
                            gdp_growth = 0
                        else:
                            gdp_growth +=  float(gpd_growth_map[key][i])
                        if key not in gpd_per_map or gpd_per_map[key][i] == "..":
                            average_gdp = 0
                        else:
                            average_gdp += float(gpd_per_map[key][i])
                        if key not in population_map or population_map[key][i] == ".." or population_map[key][i] == "":
                            population = 0
                        else:
                            population += float(population_map[key][i])
                            
                    if Host_dict[curr_year] == key:
                        host = 1
                    value[3] = value[2] + value[3] + value[4]
                    if key == "SGP":
                        value = ["Singapore"] + ["SGP"]  + [gdp_growth/4, average_gdp/4, population/4, average_gdp*population/4, host] + value[0:4]
                    else:
                        value = [noc_name[key]] + [key]  + [gdp_growth/4, average_gdp/4, population/4,average_gdp*population/4, host] + value[0:4]
                    writer.writerow(value)
                    
        except IOError:
            print("Output file " + str(curr_year) + "_" + output_file + "can not be saved")
            return

def data_preprocessing(start = 1996, end = 2012):
    data = []
    for i in range(start, end,4):
        file_name = str(i) + "_Total_clean_data.csv"
        previous = str(i-4) + "_Total_clean_data.csv"
        
        prevdata = pd.DataFrame(pd.read_csv(previous))
        prevmedal = pd.DataFrame(prevdata, columns= ['NOC','Medal'])
        prevdata = prevdata.dropna(how='all')
        prevmedal = prevmedal.dropna(how='all')
        curr_data = pd.DataFrame(pd.read_csv(file_name))
        curr_data = pd.merge(curr_data,prevmedal, on="NOC")        
        curr_data['Athlete_per'] = curr_data['Athlete'] / curr_data['Athlete'].max()
        prevAthlete = pd.DataFrame(prevdata, columns= ['NOC','Athlete'])
        curr_data = pd.merge(curr_data,prevAthlete, on="NOC")
        curr_data = curr_data.fillna(0)
        curr_data.population = curr_data.population / 1000000000
        curr_data.GDP = curr_data.GDP / 1000000000000
        curr_data.average_capita = curr_data.average_capita / 1000
        
        curr_data["average_capita_norm"] = curr_data["average_capita"] / curr_data["average_capita"].max()
        curr_data["population_norm"] = curr_data["population"] / curr_data["population"].max()
        curr_data["GDP_norm"] = curr_data["GDP"] / curr_data["GDP"].max()
        
        data.append(curr_data)
        
    train_data = pd.concat(data)
    train_data = train_data.dropna()
    previous = str(end) + "_Total_clean_data.csv"
    prevdata = pd.DataFrame(pd.read_csv(previous))
    prevmedal = pd.DataFrame(prevdata, columns= ['NOC','Medal'])
    prevdata = prevdata.dropna(how = 'all')
    
    test = pd.DataFrame(pd.read_csv (str(end + 4) + "_Total_clean_data.csv"))
    test = test.dropna()
    test = pd.merge(test,prevmedal, on="NOC")
    test['Athlete_per'] = test['Athlete'] / test['Athlete'].max()
    prevAthlete = pd.DataFrame(prevdata, columns= ['NOC','Athlete'])
    test = pd.merge(test,prevAthlete, on="NOC")

    test.population = test.population / 1000000000
    test.GDP = test.GDP / 1000000000000
    test.average_capita = test.average_capita / 1000
    
    test["average_capita_norm"] = test["average_capita"] / test["average_capita"].max()
    test["population_norm"] = test["population"] / test["population"].max()
    test["GDP_norm"] = test["GDP"] / test["GDP"].max()
    test.population = test.population / 1000000000
    test.GDP = test.GDP / 1000000000000
    test.average_capita = test.average_capita / 1000
    test_x = pd.DataFrame(test, columns= ['Athlete_per',"GDP_norm","population_norm","average_capita_norm","GDP_Growth", "average_capita", "population", "GDP", "Host",'Age', 'Athlete_x','Athlete_y',"Medal_y"])
    test_y = pd.DataFrame(test, columns= ['Medal_x'])
    train_x = pd.DataFrame(train_data, columns= ['Athlete_per',"GDP_norm","population_norm","average_capita_norm","GDP_Growth", "average_capita", "population", "GDP", "Host",'Age', 'Athlete_x','Athlete_y',"Medal_y"])
    train_y = pd.DataFrame(train_data, columns= ['Medal_x'])
    country = pd.DataFrame(test,columns=['Country/Region'])
    return train_x.to_numpy(), train_y.to_numpy(), test_x.to_numpy(), test_y.to_numpy(), country.to_numpy()
