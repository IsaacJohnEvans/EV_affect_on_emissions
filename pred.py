# %%
#coding:utf8
import pandas as pd 
import numpy as np
import tensorflow as tf
import keras_tuner as kt
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, metrics
from sklearn.ensemble import RandomForestRegressor
# %%
def load_data(em_data = True, veh_data = True, air_data = True):
    args = []
    if em_data:
        EU_em_df = pd.read_excel('Data/EU_em.xlsx',engine='openpyxl')
        EU_em_annual_df = pd.read_csv('Data/EU_em_annual.csv',)
        UK_em_df = pd.read_excel('Data/UK_em.ods', nrows = 14)
        US_em_df = pd.read_excel('Data/US_em.xlsx',engine='openpyxl')
        EU_em_df = EU_em_df.set_index('Month')
        EU_em_annual_df = EU_em_df.groupby(np.arange(len(EU_em_df))//(12)).sum()
        EU_em_annual_df.index = np.linspace(1973, 2021, EU_em_annual_df.shape[0], dtype=np.int32)
        UK_em_df = UK_em_df.transpose()
        UK_em_df.columns = UK_em_df.iloc[0, :]
        UK_em_df = UK_em_df.drop('Year')
        US_em_df = US_em_df.transpose()
        US_em_df.columns = US_em_df.iloc[0, :]
        US_em_df = US_em_df.replace('+', 0)
        US_em_df = US_em_df.drop('Year')
        args += [EU_em_df, EU_em_annual_df, UK_em_df, US_em_df]
    if veh_data:
        EU_veh_df = pd.read_csv('Data/EU_veh.csv')
        UK_veh_df = pd.read_excel('Data/UK_veh.ods', nrows= 27)
        US_veh_df = pd.read_excel('Data/US_veh.xlsx', nrows = 9)
        US_veh_df = US_veh_df.transpose()
        US_veh_df.columns = US_veh_df.iloc[0, :]
        US_veh_df = US_veh_df.drop('Model Year')
        args += [EU_veh_df, UK_veh_df, US_veh_df]
    if air_data:
        UK_nox_annual_df = pd.read_csv('Data/Figure06_NOx_time_series.csv')
        UK_pm_all_annual_df = pd.read_csv('Data/Figure03_PM_time_series.csv')
        USA_nox_annual_df = pd.read_csv('Data/US_nox_em_time_series.csv')
        USA_pm_10_annual_df = pd.read_csv('Data/US_pm10_year.csv')
        USA_pm_2_5_annual_df = pd.read_csv('Data/US_pm2_5_year.csv')
        USA_pm_2_5_annual_df = USA_pm_2_5_annual_df.transpose()
        USA_pm_10_annual_df = USA_pm_10_annual_df.transpose()
        OCED_PM10_df = pd.read_excel('Data/PM10_ROAD_OCED_WORLD_DATA.xlsx')
        OCED_NOX_df = pd.read_excel('Data/NOX_ROAD_OCED_WORLD_DATA.xlsx')
        OCED_PM2_5_df = pd.read_excel('Data/PM2_5_ROAD_OCED_WORLD_DATA.xlsx')
        args += [UK_nox_annual_df,UK_pm_all_annual_df,USA_pm_10_annual_df,USA_pm_2_5_annual_df,USA_nox_annual_df,OCED_PM10_df,OCED_NOX_df,OCED_PM2_5_df]
    return args
def process_data(X, y, split_point):
    X_norm, X_attrs, y_norm, y_attrs = normalise(X, y)
    split_point = int(X_norm.shape[0] *split_point)
    y_norm = np.roll(y_norm, -time_step)
    nrows = X_norm.shape[0]
    samples = X_norm.shape[1]
    X_norm = np.repeat(X_norm, data_memory, 0).reshape(nrows, data_memory, samples)
    x_train, x_test, y_train, y_test = train_test_split(X_norm, y_norm, split_point)
    return x_train, x_test, y_train, y_test, nrows, samples, X_norm, y_norm, X_attrs, y_attrs
def normalise(X, y):
    X_attrs = np.zeros((X.shape[-1], 2))
    y_attrs = np.zeros((y.shape[-1], 2))
    for i in range(X.shape[-1]):
        X_attrs[i, :] = [np.mean(X[:, i]), np.var(X[:, i])]
        X[:, i] = (X[:, i] - np.mean(X[:, i]))/np.var(X[:, i])**0.5
    for i in range(y.shape[-1]):
        y_attrs[i, :] = [np.mean(y[:, i]), np.var(y[:, i])]
        y[:, i] = (y[:, i] - np.mean(y[:, i]))/np.var(y[:, i])**0.5  
    return X, X_attrs, y, y_attrs
def train_test_split(X, y, split_point):
    x_train = X[:split_point, :, :]
    x_test = X[split_point:, :, :]
    y_train = y[:split_point]
    y_test = y[split_point:]
    return x_train, x_test, y_train, y_test
def create_model(layers, input_shape, print_summary):
    model = keras.Sequential(layers)
    model.build(input_shape=input_shape)
    model.compile(loss='mse', optimizer='adam', metrics = [tf.keras.metrics.MeanSquaredError()])
    if print_summary:
        model.summary()
    return model
def run_model(X, y, data_memory, epochs, batch_size, model_layer, model_des, split_point, print_summary, load_model_bool, load_file, save_file):
    x_train, x_test, y_train, y_test, nrows, samples, X_norm, y_norm, X_attrs, y_attrs = process_data(X, y, split_point)
    input_shape = (x_train.shape[0], data_memory, samples)
    model = create_model(model_layer, input_shape, print_summary)
    if load_model_bool:
        model = keras.models.load_model(load_file)
    history = model.fit(x_train, y_train, validation_split = 0.1, epochs= epochs , batch_size=batch_size)
    model.save(save_file)
    y_pred_norm = np.concatenate((model.predict(x_train[:, :, :]), model.predict(x_test[:, :, :])))
    y_pred_norm = np.roll(y_pred_norm, 1, axis = 1)
    y_pred = np.roll(y_pred_norm *y_attrs[:, 1]**0.5 + y_attrs[:, 0] , 0)
    mse = metrics.MeanSquaredError()
    mse.update_state(y_norm, y_pred_norm)
    test_loss = mse.result().numpy()
    print(test_loss)
    train_loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]
    model_performance = [time_step, data_memory, samples, model_des, epochs, batch_size, train_loss, val_loss, test_loss]
    df_model = pd.DataFrame(model_performance).transpose()
    df_model.columns = df_model_columns
    df_model.to_csv('model_performance.csv', mode = 'a', header= False)
    return y, y_pred, history
def combined_model(em_X, veh_X, em_y, veh_y, t, data_memory, epochs, batch_size, model_layer, model_des, split_point, print_summary, load_model_bool, load_model, save_model, save_fig):
    X = np.concatenate((em_X, veh_X), axis = 1)
    y = np.concatenate((em_y, veh_y), axis = 1)
    y, y_pred, history = run_model(X, y, data_memory, epochs, batch_size, model_layer, model_des, split_point, print_summary, load_model_bool, load_model, save_model)
    y = np.concatenate((em_y, veh_y), axis = 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex= True)
    ax1.plot(t, y[:, 0], 'g', t + time_step - 2, y_pred[:, 0], 'r')
    ax2.plot(t, y[:, 1], 'g', t + time_step - 2, y_pred[:, 1], 'r')
    ax1.legend(['Actual', 'Predictions'])
    ax1.set_title('Emissions')
    ax1.set(ylabel = 'CO2 Emissions')
    ax2.set_title('Electric Vehicles')
    ax2.legend(['Actual', 'Predictions'])
    ax2.set(ylabel = 'Electric vehicles')
    ax2.set(xlabel = 'Year')
    plt.savefig(save_fig)
    train_loss = history.history['loss']
    return y, y_pred, train_loss
def run_regr(X, y, t):
    regr = RandomForestRegressor(max_depth=3, random_state=0)
    regr.fit(X, y)
    nrows = X.shape[0]
    y_reg, abs_loss, per_loss = np.zeros(nrows), np.zeros(nrows), np.zeros(nrows)  
    for i in range(nrows):
        y_reg[i] = regr.predict(np.array([X[i, :]]))
    plt.plot(t, y, 'g', t, y_reg, 'r')
    for i in range(nrows):
        abs_loss[i] = np.abs(y[i] - y_reg[i])
        if y[i] > 0:
            per_loss[i] = abs_loss[i] / y[i]
    print(np.sum(abs_loss)/ nrows, np.sum(per_loss)/ nrows)
# %%
args = load_data()
EU_em_df, EU_em_annual_df, UK_em_df, US_em_df, EU_veh_df, UK_veh_df, US_veh_df, UK_nox_annual_df,UK_pm_all_annual_df,USA_pm_10_annual_df,USA_pm_2_5_annual_df,USA_nox_annual_df,OCED_PM10_df,OCED_NOX_df,OCED_PM2_5_df = args
# %%
time_step = 5
df_model_columns =  ['time_step', 'data_memory', 'samples', 'layers', 'epochs', 'batch_size', 'training_loss', 'val_loss', 'test_loss']
EU_em_X, EU_em_y = np.split(EU_em_df.to_numpy(), [11], 1)
EU_veh_X = EU_veh_df.to_numpy()
EU_veh_y = np.roll(np.sum(EU_veh_X[:, 1:3], axis =1), time_step)
EU_veh_y[:time_step] = 0
EU_veh_y = np.array([EU_veh_y]).transpose()
EU_em_annual_X, EU_em_annual_y = np.split(EU_em_df.groupby(np.concatenate((np.repeat(np.linspace(1973, 2020, 48, dtype = np.int64), 12), 2021 *np.ones((10), dtype=np.int64)), axis = 0)).sum().to_numpy(), [11], 1)
EU_X = np.concatenate((EU_em_annual_X[-13:-2, :], EU_veh_X), axis = 1)[:, 1:]
EU_y = np.concatenate((EU_em_annual_y[-13: -2, :], EU_veh_y), axis = 1)
EU_t = np.linspace(2010, 2020, 11)
UK_em_X, UK_em_y = np.split(UK_em_df.to_numpy(dtype=np.float64), [13], 1)
UK_veh_X = UK_veh_df.to_numpy(dtype=np.float64)
UK_veh_y = np.array([np.sum(UK_veh_X[:, 3:7], axis = 1)]).transpose()
UK_X = np.concatenate((UK_em_X[4:, :], UK_veh_X[:-1, :]), axis = 1)[:, 1:]
UK_y = np.concatenate((UK_em_y[4:], UK_veh_y[: -1]), axis = 1)
UK_t = np.linspace(1994, 2019, 26)
US_em_X, US_em_y, a = np.split(US_em_df.to_numpy(dtype=np.float64), [45,46], 1)
US_veh_X, US_veh_y = np.split(US_veh_df.to_numpy(dtype=np.float64), [8], 1)
US_X = np.concatenate((US_em_X[2:, :15], US_veh_X), axis = 1)
US_y = np.concatenate((US_em_y[2:, :15], US_veh_y), axis = 1)
US_t = np.linspace(1993, 2020, 28)
# %%
em_X, veh_X, em_y, veh_y, t = EU_em_annual_X[-13:-2, 1:], EU_veh_X, EU_em_annual_y[-13: -2, :], EU_veh_y, EU_t
load_model_bool, load_model, save_model, save_fig = False, 'EU_model', 'EU_model', 'EU'
em_X, veh_X, em_y, veh_y, t = UK_em_X[4:, :], UK_veh_X[:-1, :], UK_em_y[4:], UK_veh_y[: -1], UK_t
load_model_bool, load_model, save_model, save_fig = False, 'UK_model', 'UK_model', 'UK'
em_X, veh_X, em_y, veh_y, t = US_em_X[2:, :15], US_veh_X, US_em_y[2:, :15], US_veh_y, US_t
load_model_bool, load_model, save_model, save_fig = False, 'US_model', 'US_model', 'US'
data_memory, epochs, batch_size, split_point = 10, 50, 1, 0.9
model_des, print_summary = 'D16, LSTM32, D4, D2', False
model_layer = [layers.Dense(16, activation= 'linear'), layers.LSTM(32,activation= 'sigmoid', dropout = 0.1, recurrent_dropout = 0.2, return_sequences=False), layers.Dense(4, activation='linear'), layers.Dense(2)]
y, y_pred, train_loss = combined_model(em_X, veh_X, em_y, veh_y, t, data_memory, epochs, batch_size, model_layer, model_des, split_point, print_summary, load_model_bool, load_model, save_model, save_fig)
# %%
loss_t = np.linspace(1, epochs, epochs)
plt.plot(loss_t, train_loss)
# %%
run_regr(EU_X, EU_y[:, 0], EU_t)
run_regr(UK_X, UK_y[:, 0], UK_t)
run_regr(US_X, US_y[:, 0], US_t)
run_regr(EU_X, EU_y[:, 1], EU_t)
run_regr(UK_X, UK_y[:, 1], UK_t)
run_regr(US_X, US_y[:, 1], US_t)
# %%
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Dense(units=hp.Int("dense_units", min_value=16, max_value=128, step=16)))
    model.add(layers.GRU(units=hp.Int("GRU_units", min_value=16, max_value=128, step=16), activation='sigmoid', recurrent_activation='sigmoid'))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss = 'mse')
    return model
val_split = int(x_train.shape[0] * 0.9)
x_train_val, x_val, y_train_val, y_val = train_test_split(x_train, y_train, val_split)
tuner = kt.RandomSearch(hypermodel=build_model, objective="val_loss", max_trials=3, executions_per_trial=2, overwrite=True, directory="RNN_em", project_name="RNN_em")
tuner.search(x_train, y_train, epochs=2, validation_data=(x_val, y_val))
models = tuner.get_best_models(num_models=2)
best_model = models[0]
input_shape = (x_train.shape[0], data_memory, samples)
best_model.build(input_shape=input_shape)
best_model.summary()
tuner.results_summary()
best_hps = tuner.get_best_hyperparameters(5)
model = build_model(best_hps[0])
history = model.fit(x=x_train, y=y_train, epochs=10)