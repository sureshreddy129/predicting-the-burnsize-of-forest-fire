#!/usr/bin/env python
# coding: utf-8

# # Final notebook of Case study One

# In[45]:


## importing required libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
### Setting the figsize for a better vizualization 
plt.rcParams['figure.figsize']=(20,10)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from scipy.sparse import hstack
import pickle
import joblib


# In[49]:


def predicting_the_area(Data):
     
    Data_nml=Data.copy()  
    
    #### Creating a new feature M_D

    M_D=[]
    
    
    #### Giving very less weightage to these months
    if (Data['month']=='jan') or (Data['month']=='may') or (Data['month']=='nov'):
        M_D.append(np.random.normal(0.0,0.3,1)[0])
        
    ### Giving more weightage to these months  
    if (Data['month']=='aug') or (Data['month']=='sep'):
        M_D.append(np.random.normal(0.7,1,1)[0])
        
    ### Giving moderate weightage to this month
    if Data['month']=='jul':
        M_D.append(np.random.normal(0.6,0.7,1)[0])
        
    ### Giving less weightage to these months  
    if Data['month']=='feb' or Data['month']=='mar' or Data['month']=='apr' or Data['month']=='jun' or Data['month']=='oct' or Data['month']=='dec':         
        M_D.append(np.random.normal(0.3,0.6,1)[0]) 
    
    Data['M_D']=M_D[0]
    
    mean_month_area_cbrt=joblib.load('mean_month_area_cbrt')  ### Importing the mean encoding values for month,day
    mean_day_area_cbrt=joblib.load('mean_day_area_cbrt')
  
    Data['month']=mean_month_area_cbrt[Data['month']]   ### Replacing the month,day with mean encoding values
    Data['day']=mean_day_area_cbrt[Data['day']]
    
    Data['RH_cbrt']=np.cbrt(Data['RH'])            ### Transforming the features
    Data['wind_sqrt']=np.sqrt(Data['wind'])
    
    ### Creating new feature X_Y by giving more importance to the corresponding coordinates

    X_Y=[]
    
    for i in range(1):
        if Data['Y']<7:
            new=0.6*Data['X']+0.4*Data['Y']
            X_Y.append(new)
        if Data['Y']>=7:
            new=0.9*Data['X']+0.1*Data['Y']
            X_Y.append(new)
            
    Data['X_Y']=X_Y[0]
    

    
    ### We have already seen in above plots the contribution of temp,rh and wind 
    ## Here we are creating the new feature combinig the three of them in the following manner

    ## TRW=0.55*temp+0.3*RH+0.15*wind

    Data['TRW']=Data['temp']*0.4+0.4*Data['RH']+0.2*Data['wind']
    
    ### BUI is the fire behaviour index
     ## BUI is the combination of DMC and DC dominated by DMC we don't the weightages exactly
     ## As Bui dominated by DMC giving more weightage to DMC 
     ## let BUI=0.85*DMC+0.15*DC

    Data['BUI']=0.85*Data['DMC']+0.15*Data['DC']
    
    ## Distinguishing the fire intensity based on the FFMC value
    ### These distinguished values are from https://www.youtube.com/watch?v=Uy_1V20j_L8&ab_channel=NorthwestKnowledgeHub
    ### Depending up on the FFMC code we can distinguish the fire intensity into low,moderate,high,very high and extreme categories
    ### 0-80 Low, Let 1
    ### 81-87 moderate, Let 2
    ### 88-90 High, Let 3
    ### 91-92 Very High, Let 4
    ### 93+   Extreme, Let 5
  
    if (Data.FFMC.round()>=0) & (Data.FFMC.round()<=80):
        Data['FFMC_intensity']=1
    if (Data.FFMC.round()>=81) & (Data.FFMC.round()<=87):
        Data['FFMC_intensity']=2
    if (Data.FFMC.round()>=88) & (Data.FFMC.round()<=90):
        Data['FFMC_intensity']=3
    if (Data.FFMC.round()>=91) & (Data.FFMC.round()<=92):
        Data['FFMC_intensity']=4
    else:
        Data['FFMC_intensity']=5
    
    
    ## Distinguishing the fire intensity based on the DMC value
    ### These distinguished values are from https://www.youtube.com/watch?v=Uy_1V20j_L8&ab_channel=NorthwestKnowledgeHub
    ### Depending up on the DMC code we can distinguish the fire intensity into low,moderate,high,very high and extreme categories
     ### 0-12 Low, Let 1
     ### 13-27 moderate, Let 2
     ### 28-41 High, Let 3
     ### 42-62 Very High, Let 4
     ### 63+   Extreme, Let 5

    if (Data.DMC.round()>=0) & (Data.DMC.round()<=12):
        Data['DMC_intensity']=1
    if (Data.DMC.round()>=13) & (Data.DMC.round()<=27):
        Data['DMC_intensity']=2
    if (Data.DMC.round()>=28) & (Data.DMC.round()<=41):
        Data['DMC_intensity']=3
    if (Data.DMC.round()>=42) & (Data.DMC.round()<=62):
        Data['DMC_intensity']=4
    else:
        Data['DMC_intensity']=5

    ## Distinguishing the fire intensity based on the DC value
    ### These distinguished values are from https://www.youtube.com/watch?v=Uy_1V20j_L8&ab_channel=NorthwestKnowledgeHub
    ### Depending up on the DC code we can distinguish the fire intensity into low,moderate,high,very high and extreme categories
    ### 0-79 Low, Let 1
    ### 80-209 moderate, Let 2
    ### 210-274 High, Let 3
    ### 275-359 Very High, Let 4
    ### 360+   Extreme, Let 5

    if (Data.DC.round()>=0) & (Data.DC.round()<=79):
        
        Data['DC_intensity']=1
    if (Data.DC.round()>=80) & (Data.DC.round()<=209):
        Data['DC_intensity']=2
    if (Data.DC.round()>=210) & (Data.DC.round()<=274):
        Data['DC_intensity']=3
    if (Data.DC.round()>=275) & (Data.DC.round()<=359):
        Data['DC_intensity']=4
    else:
        Data['DC_intensity']=5

        
    ## Distinguishing the fire intensity based on the ISI value
    ### These distinguished values are from https://www.youtube.com/watch?v=Uy_1V20j_L8&ab_channel=NorthwestKnowledgeHub
    ### Depending up on the ISI values we can distinguish the fire intensity into low,moderate,high,very high and extreme categories
    ### 0-1.9 Low, Let 1
    ### 2-4.9 moderate, Let 2
    ### 5-7.9 High, Let 3
    ### 8.0-10.9 Very High, Let 4
    ### 11+   Extreme, Let 5

 
    if (Data['ISI']>=0) & (Data['ISI']<=1.9):
        Data['ISI_intensity']=1
    if (Data['ISI']>=1.9) & (Data['ISI']<=4.9):
        Data['ISI_intensity']=2
    if (Data['ISI']>=5.0) & (Data['ISI']<=7.9):
        Data['ISI_intensity']=3
    if (Data['ISI']>=8.0) & (Data['ISI']<=10.9):
        Data['ISI_intensity']=4
    if Data['ISI']>=11 :
        Data['ISI_intensity']=5

        
    ### FFMC is the fuel moisture code we can derive moisture content of litter from this and can add as a new feature
    ## This feature explains the moisture content of litter which is the initial layer of ground upto a depth of 5cm
    ## This equation is from https://www.youtube.com/watch?v=LqqcngYc4Ks&ab_channel=NorthwestKnowledgeHub

    ## MC=147.2(101-FFMC)/(59.5+FFMC)

    Data['FFMC_MC']=(147.2*(101-Data['FFMC']))/(59.5+Data['FFMC'])
    Data['log_FFMC_MC']=np.log(Data['FFMC_MC'])   ### Transforming the features
    
    ### DMC is the Duff moisture code we can derive moisture content  from this and can add as a new feature
    ## This feature explains the moisture content of duff layer which is the beneath the litter upto a depth of 5cm to 10cm
    ## This equation is from https://www.youtube.com/watch?v=qsWSbNOoh8I&ab_channel=NorthwestKnowledgeHub
    ## MC=exp[(DMC-244.7)/-43.4]+20

    Data['DMC_MC']=np.exp((Data['DMC']-244.7)/(-43.4))+20
    Data['log_DMC_MC']=np.log(Data['DMC_MC'])   ### Transforming the features
    
    
    ## Let's add another feature as MC_ratio i.e FFMC_MC/DMC_MC
    ## MC_ratio is the moisture content ratio
    ## This is adding as our own feature

    Data['MC_ratio']=Data['FFMC_MC']/Data['DMC_MC']
    Data['log_MC_ratio']=np.log(Data['MC_ratio'])                     ### Transforming the features
    
   
    col=['X','Y','RH','wind','ISI','FFMC_MC','DMC_MC','MC_ratio']
    for i in col:
        del Data[i]  ### Deleting the unnecessary features
    
    ### Standardizing the Transformed data
    
    scalers_transform=joblib.load('scalers_transform')     ### Importing the scalers for transformed data
    
    keys=list(scalers_transform.keys())

    for i in range(len(keys)-1):
     
        Data[keys[i+1]]=scalers_transform[keys[i+1]].transform(np.array(Data[keys[i+1]]).reshape(-1,1))

    
  ### Importing the best model i.e Linear regression model  for Transformed data
    
    model_t=joblib.load('lir_reg')
    
    ### Since the area is cbrt transformed now we are retransforming with cube
    predicted_y=(model_t.predict(Data.values.reshape(1,-1))) **3          ### Predicted area for transformed data
                                            
    
#### PRedicting for the original data 
    
    mean_month_area=joblib.load('mean_month_area')        ### importing mean encoding values
    mean_day_area=joblib.load('mean_day_area','rb')
 
    
    Data_nml['month']=mean_month_area[Data_nml['month']]             #### Replacing month and day with mean encoding values
    Data_nml['day']=mean_day_area[Data_nml['day']] 
    
   
### Standardizing the  data
    
    scalers_normal=joblib.load('scalers_normal')         ### Importing the scalers for  data
    
    keys=list(scalers_normal.keys())
    
    
    for i in range(len(keys)):
        Data_nml[keys[i]]=scalers_normal[keys[i]].transform(np.array(Data_nml[keys[i]]).reshape(-1,1))

    
    ### Importing the best model i.e Support vector Regressor for data
    
    model=joblib.load('svr_reg_mae_nml')
    
    predicted_y_nml=model.predict(Data_nml.values.reshape(1,-1))   ### Predicting the area for normal data
    
    return predicted_y,predicted_y_nml


# In[50]:


def metric_score(y,predicted_values):
    mae_transformed=abs(y-predicted_values[0])   ## Since we took only one value we are not doing mean
    mae_normal=abs(y-predicted_values[1])
    
    return mae_transformed,mae_normal


# In[51]:


for i in range(10):
    data_=pd.read_csv('forestfires.csv')

    x=data_.iloc[np.random.randint(0,517,1)[0]].drop('area')
    y=data_.iloc[np.random.randint(0,517,1)[0]]['area']

    print(y)
    area_t,area=predicting_the_area(x)
    print(area_t,area)
    mae_t,mae=metric_score(y,[area_t,area])
    print(mae_t,mae)


# In[52]:


def predict():
    data=pd.DataFrame({'X':X,'Y':Y,'month':month,'day':day,'FFMC':FFMC,'DMC':DMC,'DC':DC,'ISI':ISI,'wind':wind,'RH':RH,'rain':rain,'temp':temp},index=[0])
    print(data)
    predicted_t,predicted=predicting_the_area(data.iloc[0])
    
    return {'area_for_transformed_data':predicted_t[0],'area_for_normal_data':predicted[0]}


# In[53]:


X=5
Y=6
month='sep'
day='sat'
FFMC=85
DMC=180
DC=650
ISI=10
wind=5
RH=55
rain=0
temp=25
print(predict())


# In[ ]:




