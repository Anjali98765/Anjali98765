import numpy as np
import pandas as pd
from sklearn.linear_model import  RandomForestClassifier  
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")

data = pd.read_csv("WildFire_Prediction_Data_Set.csv")
data = np.array(data)

X = data['NDVI','LST','BURNED_AREA']
y = data['CLASS']

from sklearn.preprocessing import LabelEncoder  
label_encoder_x= LabelEncoder()  
y= label_encoder_x.fit_transform(y)  

y = y.astype('int')
X = X.astype('int')
# print(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
classifier = RandomForestClassifier(n_estimators= 10, criterion="entropy") 


classifier.fit(X_train, y_train)

pickle.dump(classifier,open('model.pkl','wb'))