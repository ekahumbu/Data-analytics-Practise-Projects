import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt  
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics  import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv('500_Person_Gender_Height_Weight_Index.csv') 
print(df.describe())

def give_names_to_indices(ind):
    if ind == 0: return 'Extremely Weak'
    elif ind == 1: return 'Weak'
    elif ind == 2: return 'Normal'
    elif ind == 3: return 'OverWeight'
    elif ind == 4: return 'Obesity'
    elif ind == 5: return 'Extremely Obese'
    
df['Index'] = df['Index'].apply(give_names_to_indices) 
sns.lmplot('Height', 'Weight', df, hue='Index', size=7, aspect=1, fit_reg=False) 
people = df['Gender'].value_counts()
categories = df['Index'].value_counts()

df[df['Gender']=='Male']['Index'].value_counts()
df[df['Gender']=='Female']['Index'].value_counts()

df2 = pd.get_dummies(df['Gender'])
df.drop('Gender', axis=1, inplace=True)
df = pd.concat([df,df2],axis=1)

scaler = StandardScaler()
df = scaler.fit_transform(df)
df = pd.DataFrame(df)

x_train, x_test, y_train, y_test = train_test_split(df,y, test_size=0.3, random_state=101)
param_grid = ('n_estimators': [100,200,300,400,500,600,700,800,1000])
grid_cv = GridSearchCV(RandomForestClassifier(random_state=101),param_grid, verbose=3)
grid_cv.fit(x_train, y_train)
print(grid_cv.best_params_)  

pred = grid_cv.predict(x_test)

print(classification_report(y_test, pred))
print('\n')
print(confusion_matrix(y_test, pred))
print('\n')
print('Accuracy is: ',accuracy_score(y_test, pred)*100)
print('\n')

def lp(details):
    gender = details[0]
    height = details[1]
    weight = details[2]

    if gender == 'Male':
        details=np.array([np.float(height), np.float(height), 0.0, 1, 1.0])
    elif gender == 'Female':
        details=np.array([np.float(height), np.float(weight), 1.0,0,0.11)
    
    y_pred = grid_cv.predict(scaler.transform(details))
    return (y_pred[0])
    
    your_details = ['Male', 175, 80]
    print(lp(your_details))
    
                                                      