import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




modelo = joblib.load('modelo_entrenado.plk')

ve = 120
scaler = MinMaxScaler( feature_range=(0, 1) )

def preprocessing(data):
  data = data.drop(['Date','Adj Close'], axis=1)
  data = data.round(2)

  #transform dataframe to numpy array
  data = data.values

  #normalize data
  data = scaler.fit_transform(data)

  X_train = []
  y_train = []

  for i in range(ve, len(data)):
      X_train.append(data[i-ve:i, :])
      y_train.append(data[i, 0])
 
  X_train, y_train = np.array(X_train), np.array(y_train)
  


  return X_train, y_train


df = pd.read_csv('AMZN.csv')

X_df, y_df = preprocessing(df)

real = df.loc[120:, ['Open']].values.round(2)

nnr = modelo.predict(X_df)


nnr_list = []

for i in range(len(nnr)):
    x = (nnr[i,0]*(scaler.data_max_[0]-scaler.data_min_[0]))+scaler.data_min_[0]
    nnr_list.append(x)



plt.plot(real, color = 'red', label = 'Real Stock Price')
plt.plot(nnr_list, color = 'blue', label = 'Predicted Stock Price')
plt.title('Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
