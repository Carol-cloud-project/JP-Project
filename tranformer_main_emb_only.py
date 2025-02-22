import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import os
os.makedirs('result', exist_ok=True)  
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore") 
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error
from scipy.stats import mstats
from keras.optimizers import Adam


seed=42
# Set the random seed for reproducibility
np.random.seed(seed)  # For NumPy
import random
random.seed(seed)  # For Python's built-in random module

# For TensorFlow/Keras
import tensorflow as tf
tf.random.set_seed(seed)  # For TensorFlow/Keras


# MAPE,MAPE
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100
def smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100

def split_data(dataset, timesteps=10, predict_steps=2):  
    datax = []  # 构造x
    datay = []  # 构造y
    for each in range(len(dataset) - timesteps - predict_steps):
        x = dataset[each:each + timesteps, :]
        y = dataset[each + timesteps:each + timesteps + predict_steps, -1:]
        datax.append(x)
        datay.append(y)
    datax = np.array(datax).reshape(len(datax), -1)
    datay = np.array(datay).reshape(len(datay), -1)
    return np.array(datax),np.array(datay)  #
# 导入数据
# x_data = pd.read_table('/home/rliuaj/balance_sheet/text_embeddings.csv',sep=',',index_col=0)
# #x_data = x_data.drop(['Market Cap', 'Industry', 'City'],axis=1)
# y_data = pd.read_table('/home/rliuaj/balance_sheet/df_y',sep=',',index_col=0)
x_data = pd.read_table('/home/rliuaj/balance_sheet/df_X', sep=',', index_col=0, usecols=lambda column: column != 'Unnamed: 0')
x_data=x_data.iloc[:,:-1]
y_data = pd.read_table('/home/rliuaj/balance_sheet/df_y', sep=',', index_col=0, usecols=lambda column: column != 'Unnamed: 0')
y_data=y_data.iloc[:,:-1]
x_data = x_data.interpolate()
y_data = y_data.interpolate()

# 划分输入与输出
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=seed) # 20

# # Apply winsorization (example: 1st and 99th percentile)
# for col in X_train.columns:
#     X_train[col] = mstats.winsorize(X_train[col], limits=[0.05, 0.05])  # limits=[0.01, 0.01] means 1st and 99th percentiles
# for col in y_train.columns:
#     y_train[col] = mstats.winsorize(y_train[col], limits=[0.05, 0.05])  # limits=[0.01, 0.01] means 1st and 99th percentiles
# # X\y归一化
# #x_scaler = MinMaxScaler()
# x_scaler = StandardScaler()
# X_train = x_scaler.fit_transform(X_train)
# X_test = x_scaler.transform(X_test)
y_scaler = MinMaxScaler()
y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)
# 输入格式
X_train = np.array(X_train).reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = np.array(X_test).reshape((X_test.shape[0], 1, X_test.shape[1]))
print('X_train.shape, y_train.shape, X_test.shape, y_test.shape')
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
import tensorflow as tf
def custom_loss(y_true, y_pred):
    l_sum = 0
    for i in range(5):
        l = tf.square(y_true[:, i] - y_pred[:, i])
        l_sum = l_sum+l
    s_sum = tf.square((y_pred[:, 0]+y_pred[:, 1]) - (y_pred[:, 2]+y_pred[:, 3]+y_pred[:, 4]))
    return l_sum + s_sum
# -----------------Transformer-------------------
def transformer_encoder(inputs):
    # 标准化与注意
    x = MultiHeadAttention(
        key_dim=256, num_heads=4, dropout=0.2)(inputs, inputs)
    x = Dropout(0.2)(x)
    res = x + inputs
    # 前馈部分
    x = Dense(units=128, activation="relu")(res)
    x = Dropout(0.2)(x)
    x = Dense(inputs.shape[-1])(x)
    x = Dropout(0.2)(x)
    return x + res
inputs = Input(shape=(X_train.shape[-2:]))
x = inputs
x = transformer_encoder(x)
x = GlobalAveragePooling1D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.2)(x)
# 输出层
outputs = Dense(5)(x)
model = Model(inputs, outputs)
# Define the learning rate
learning_rate = 0.0001  # Example: setting learning rate to 0.001
# Use the Adam optimizer with a custom learning rate
optimizer = Adam(learning_rate=learning_rate)
#model.compile(loss=custom_loss, optimizer='adam',metrics=['mse'])
model.compile(loss=custom_loss, optimizer=optimizer, metrics=['mse'])

# 模型训练
history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                    validation_data=(X_test, y_test), verbose=2)
# Loss
plt.figure()
plt.plot(history.history['loss'], label='train-loss')
plt.plot(history.history['val_loss'], label='test-loss')
plt.legend()
plt.title('Loss')
plt.savefig(os.getcwd() + '/result/Transformer_Loss_emb_only')
plt.show()
# MAE
plt.figure()
plt.plot(history.history['mse'], label='train-mse')
plt.plot(history.history['val_mse'], label='test-mse')
plt.legend()
plt.title('Loss')
plt.savefig(os.getcwd() + '/result/Transformer_MSE_emb_only')
plt.show()
# 真实值与预测值
y_true = y_scaler.inverse_transform(y_test)
y_pred = y_scaler.inverse_transform(model.predict(X_test).reshape(-1, 5))
# y_true = np.array(y_test)
# y_pred = model.predict(X_test).reshape(-1, 5)
print(y_true.shape)
print(y_pred.shape)
# plot
for i in range(5):
    plt.figure(figsize=(12, 6), facecolor='w')  
    plt.plot(y_true[:,i], label='True')
    # plt.plot(y_pred[:,i], label='Pred')
    plt.plot(y_true[:,i]-y_pred[:,i], label='Errors')
    plt.legend(loc='upper right')
    plt.title('True and Pred')
    plt.savefig(os.getcwd() + '/result/Transformer_'+str(i)+'_True-Pred_emb_only.png')
    plt.close()
# 模型评估
print('Transformer test')
print('R2: %.3f' % r2_score(y_true, y_pred))
print('MAE: %.3f' % mean_absolute_error(y_true, y_pred))
print('RMSE: %.3f' % np.sqrt(mean_squared_error(y_true, y_pred)))
for i in range(5):
    print(f'feature{i} MAPE: , {np.mean(np.abs((y_true[:, i] - y_pred[:, i]) / y_true[:, i])) * 100}')
result = pd.DataFrame([r2_score(y_true, y_pred), mean_absolute_error(y_true, y_pred),
                       mean_squared_error(y_true, y_pred),
                       np.sqrt(mean_squared_error(y_true, y_pred)),
                       mape(y_true, y_pred),
                       smape(y_true, y_pred)], columns=['Transformer'],
                      index=['R2', 'MAE', 'MSE', 'RMSE', 'MAPE', 'SMAPE']).T
print(result)
result.to_csv(os.getcwd() + '/result/Transformer_result_emb_only.csv')
pd.DataFrame(y_true).to_csv(os.getcwd() + '/result/Transformer_y_true_emb_only.csv')
pd.DataFrame(y_pred).to_csv(os.getcwd() + '/result/Transformer_y_pred_emb_only.csv')