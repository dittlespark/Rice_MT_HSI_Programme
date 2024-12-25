# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:09:36 2022

@author: Administrator
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso,LassoCV
from sklearn.metrics import r2_score                
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings("ignore")



#--------SLR regression-------#
def stepregression(X,Y):
    
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size =0.2,random_state = 0)  #random_state = 0
    data=pd.concat([x_train,y_train],axis=1)   # Merge by column
    print(data)
    variate=set(x_train.columns) # Get column names
    selected = [] # Final set of independent variables
    current_score, best_new_score = float('inf'), float('inf')  
    while variate:
        score_with_variate = [] 
		# Traverse the independent variables
        for candidate in variate:
            formula = "{}~{}".format(c3[k-1][s-1],"+".join(selected+[candidate]))  # combination
            model = smf.ols(formula=formula, data=data).fit()
            score = smf.ols(formula=formula, data=data).fit().aic
            score_with_variate.append((score, candidate)) 
        score_with_variate.sort(reverse=True)  
        best_new_score, best_candidate = score_with_variate.pop()  
        if current_score > best_new_score:  
            variate.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
            print(selected)

        else: 
            break
       
    formula = "{}~{}".format(c3[k-1][s-1],"+".join(selected))
    model = smf.ols(formula=formula, data=data).fit()
    y_pred = model.predict(x_test)
    print(y_pred)
    print(formula)
    print("R^2:",model.rsquared)
    print(model.params)
    print(model.summary())
    
    x = np.arange(20,23)
    fig = plt.figure(figsize = (10,4))
    ax1 = fig.add_subplot(1,2,1)
    ax1.scatter(y_test,y_pred)
    plt.plot(x,x,color = 'red')
    plt.xlim(20.8,21.5)
    plt.ylim(20.8,21.5)
    plt.xlabel('test',size=20)
    plt.ylabel('pred',size=20)
    plt.title(str(c2),size=20)
    number.append(len(selected))
    daixie.append(str(c3[k-1][s-1]))
    R.append(str(model.rsquared))
    return 0

    

#-------PLSR regression---------
def PLSR(X,Y):
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size =0.2,random_state = 0)
    pls_model_setup = PLSRegression(scale=True)
    param_grid = {'n_components': range(1, 20)}
    #GridSearchCV Optimize parameters and train models
    gsearch = GridSearchCV(pls_model_setup, param_grid,cv=10)
    try:
        pls_model = gsearch.fit(x_train, y_train)
        y_pred = pls_model.predict(x_test)
        print(str(c2))
        print("R2:",r2_score(y_test,y_pred))
        number.append(len(c3[k-1][0:s-1]))
        daixie.append(str(c3[k-1][s-1]))
        R.append(str(r2_score(y_test,y_pred)))
        
    except ValueError:
        return 0
      
    return 0

#---------Radomfroest regression -----
def randomforest(X,Y):
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size =0.2,random_state = 0)
    parameters = {'n_estimators':range(1,20)}
    clf = RandomForestRegressor()
    rfr = GridSearchCV(estimator=clf,param_grid=parameters,cv=10)
    rfr.fit(x_train, y_train)  
    y_pred = rfr.predict(x_test)
    print(str(c2))
    print("R2:",r2_score(y_test,y_pred))
    number.append(len(c3[k-1][0:s-1]))
    daixie.append(str(c3[k-1][s-1]))
    R.append(str(r2_score(y_test,y_pred)))
    return 0
    
    
#---------SVM regression----------
def SVRregression(X,Y):
    x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2, random_state = 0)
    svr_rbf=GridSearchCV(SVR(kernel='rbf', gamma=0.1),{"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)},cv=10)
    svr_rbf.fit(x_train, y_train)
    y_pred=svr_rbf.predict(x_test)
    print(str(c2))
    print("R2:",r2_score(y_test,y_pred))
    number.append(len(c3[k-1][0:s-1]))
    daixie.append(str(c3[k-1][s-1]))
    R.append(str(r2_score(y_test,y_pred)))
    return 0
   
     
#--------Ridge regression--------
def Ridgeregression(X,Y):
    
    x_train, x_test, y_train, y_test=train_test_split(X,Y,test_size = 0.2,random_state = 0)
    model = Ridge(normalize=True)
    alpha_can = np.logspace(-3, 2, 10)
    ridge=GridSearchCV(model, param_grid={'alpha': alpha_can}, cv=10)
    ridge.fit(x_train,y_train) 
    y_pred=ridge.predict(x_test)
    print(str(c2))
    print("R2:",r2_score(y_test, y_pred))
    number.append(len(c3[k-1][0:s-1]))
    daixie.append(str(c3[k-1][s-1]))
    R.append(str(r2_score(y_test,y_pred)))
    return 0
        

    
#--------lasso regression--------     
def Lassoregression(X,Y):
    

    x_train, x_test, y_train, y_test=train_test_split(X,Y,test_size = 0.2)
    model = Lasso(normalize=True)
    alpha_can = np.logspace(-5, 2, 10)
    lasso=GridSearchCV(model, param_grid={'alpha': alpha_can}, cv=10)
    lasso.fit(x_train,y_train)

    y_pred=lasso.predict(x_test)
    coef = pd.Series(lasso.best_estimator_.coef_,index=x_train.columns)
    a=coef[coef != 0].abs().sort_values(ascending = False)
    print(str(c2))
    print("selected:"+str(list(a.index)))
    print("R2:",r2_score(y_test, y_pred))
    print("rmse:", sqrt(mean_squared_error(y_test, y_pred)))
   
    number.append(len(a.index))
    daixie.append(str(c3[k-1][s-1]))
    R.append(str(r2_score(y_test, y_pred))) 
    RMSE.append(sqrt(mean_squared_error(y_test, y_pred)))
    #f2.close()         
    return 0
   

    
    
        

if __name__=="__main__": 
    
    k=0
    c3=[]
    daixie=[]
    number=[]
    R=[]   
    RMSE=[]
    data1=pd.read_csv('G:\hyper.csv',encoding='utf-8')
    data2=pd.read_csv('G:\metabolite.csv',encoding='utf-8')
    #data2=data2[abs(data2-data2.mean())<3*data2.std()]  #3σ
    #data2.to_csv("代谢processed.csv")
    data1=data1.iloc[:,1:]
    data2=data2.iloc[:,1:]

    
    
    for j in range(0,887):    #887
        flag2=0
        flag1=0
        c1=[]     
        for i in range(0,1848):        
            y=data2.iloc[:, j:j+1]
            x=data1.iloc[:, i:i+1]
            #print(y)
            #print(x)
            T4=np.array(y.T)
            T5=np.array(x.T)
            t=np.corrcoef(T5,T4)
            #print(list(x),list(y),t[0][1]) 
            if(t[0][1]>0.3):  
                #print(list(x),list(y),t[0][1])   
                c1=list(x)+c1
                flag1=1    
        if(flag2==0 and flag1==1):
            c2=list(y)
            flag2=1
        if(flag1==1):
            c3.append(list(c1+c2))
            k=k+1
            s=len(c3[k-1])
            X=data1[c3[k-1][0:s-1]] 
            Y=data2[c3[k-1][s-1]]
            #Y=np.log(Y)  #
            #X = (X - X.mean())/np.std(X) #
            #Y = (Y - Y.mean())/np.std(Y)
            #stepregression(X,Y)
            #PLSR(X,Y)
            #Ridgeregression(X,Y)
            Lassoregression(X,Y)
            #randomforest(X,Y)
            #SVRregression(X,Y)
            #Bayes(X,Y)
            #KNN(X,Y)
  
number=pd.DataFrame(number,columns=['number'])
daixie=pd.DataFrame(daixie,columns=['daixie'])
R=pd.DataFrame(R,columns=['R2'])
RMSE=pd.DataFrame(RMSE,columns=['RMSE'])
total=pd.concat([number,daixie],axis=1)
output=pd.concat([total,RMSE],axis=1)   
#output=pd.concat([output,R],axis=1)   
print(output)
#output.to_csv('rice.csv')   
