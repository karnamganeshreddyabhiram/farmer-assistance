import pandas as pd
import numpy as np
df = pd.read_csv('nellore.csv')
features=['maxtempC','mintempC','humidity','precipMM','pressure','tempC']
df_f=df[features].values
df_f1=df['tempC'].values
df_f2=df['precipMM'].values
df_f3=df['humidity'].values

X=[]
y=[]
y1=[]
y2=[]
for i in range(30,len(df_f),1):
    X.append(df_f[i-30:i])
    y.append(df_f1[i])
    y1.append(df_f2[i])
    y2.append(df_f3[i])
    
X=np.array(X)
y=np.array(y)
y1=np.array(y1)
y2=np.array(y2)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y1, test_size = 0.2)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y2, test_size = 0.2)

X_train=X_train.reshape(len(X_train),-1)
X_test=X_test.reshape(len(X_test),-1)
X_train1=X_train1.reshape(len(X_train1),-1)
X_test1=X_test1.reshape(len(X_test1),-1)
X_train2=X_train2.reshape(len(X_train2),-1)
X_test2=X_test2.reshape(len(X_test2),-1)

print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', y_test.shape)

from sklearn.metrics import mean_absolute_error
y_pred=[y_train.mean()]*len(y_train)
print("baseline mae for temperature: ",round(mean_absolute_error(y_train,y_pred),5))

y_pred1=[y_train1.mean()]*len(y_train1)
print("baseline mae for precipitation : ",round(mean_absolute_error(y_train1,y_pred1),5))

y_pred2=[y_train2.mean()]*len(y_train2)
print("baseline mae for humidity: ",round(mean_absolute_error(y_train2,y_pred2),5))


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
rf = make_pipeline(StandardScaler(),RandomForestRegressor())
rf1 = make_pipeline(StandardScaler(),RandomForestRegressor())
rf2 = make_pipeline(StandardScaler(),RandomForestRegressor())

rf.fit(X_train, y_train)
rf1.fit(X_train1, y_train1)
rf2.fit(X_train2, y_train2)
print('Random forest regressor model training mae for temperature: ', mean_absolute_error(y_train, rf.predict(X_train)))
print('Random forest regressor model testing mae for temperature: ', mean_absolute_error(y_test, rf.predict(X_test)))

print('Random forest regressor model training mae for precipitation: ', mean_absolute_error(y_train1, rf1.predict(X_train1)))
print('Random forest regressor model testing mae for precipitation: ', mean_absolute_error(y_test1, rf1.predict(X_test1)))

print('Random forest regressor model training mae for humidity: ', mean_absolute_error(y_train2, rf2.predict(X_train2)))
print('Random forest regressor model testing mae for humidity: ', mean_absolute_error(y_test2, rf2.predict(X_test2)))

y_pred=rf.predict(X_test)
y_pred=y_pred.reshape(len(X_test),1)
errors=abs(y_pred-y_test)
mape=100*(errors/y_test)
accuracy=100-np.mean(mape)
print("Random forest model accuracy for temperature ", round(accuracy,2),"%")

y_pred=rf1.predict(X_test1)
y_pred=y_pred.reshape(len(X_test1),1)
errors=abs(y_pred-y_test1)
mape=100*(errors/(y_test1+0.0001))
accuracy=100-np.mean(mape)
print("Random forest model accuracy for precipitation ", round(accuracy,2),"%")

y_pred=rf2.predict(X_test2)
y_pred=y_pred.reshape(len(X_test2),1)
errors=abs(y_pred-y_test2)
mape=100*(errors/y_test2)
accuracy=100-np.mean(mape)
print("Random forest model accuracy for humidity ", round(accuracy,2),"%")


d1=rf.predict([df_f[1:31].reshape(-1)])
d2=rf1.predict([df_f[1:31].reshape(-1)])
d3=rf2.predict([df_f[1:31].reshape(-1)])