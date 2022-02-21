from model import *
from train_utilities import *
from general_utilities import *
from baseline_models import *
from extreme_time(EVL) import *
from extreme_time_2(EVL) import *
from train_model_gev import *

from pandas import DataFrame
import pandas as pd
import numpy as np
import os, random
import math
import scipy.io
import matplotlib.pyplot as plt
from matplotlib import rc, style
import seaborn as sns
import datetime as dt
from tqdm import tqdm as tq
from numpy import vstack, sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pandas.plotting import register_matplotlib_converters
import matplotlib.patches as mpatches
from google.colab import files
from statistics import mean
import scipy.stats as stats
from scipy.special import gamma

from scipy.stats import genextreme
from scipy.stats import pearsonr

from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

from functools import partial
from pylab import rcParams

import torch
from torch import nn, optim
import torchvision
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.nn import init
from torch.utils.data import random_split
import torchvision.transforms as transforms
from torchvision import models
from torchsummary import summary

# from ray import tune
# from ray.tune import CLIReporter
# from ray.tune.schedulers import ASHAScheduler
# from ray.tune.stopper import TrialPlateauStopper

import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
%config InlineBackend.figure_format='retina'

sns.set(style='whitegrid', palette='muted', font_scale=1.8)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#93D30C", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 24, 10
register_matplotlib_converters()

RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
%config InlineBackend.figure_format='retina'

sns.set(style='whitegrid', palette='muted', font_scale=1.8)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#93D30C", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 24, 10
register_matplotlib_converters()

RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUDA_LAUNCH_BLOCKING=1
torch.use_deterministic_algorithms(False)

# Data Loading

data_path = '/content/drive/MyDrive/PhD-Research-Phase-1/Data/'

df = pd.read_csv(data_path+'Energy_Dataset.csv')

iltered_customer_id = [2, 13, 14, 20, 33, 35, 38, 39, 56,
                        69, 73, 74, 75, 82, 87, 88, 101, 104,
                        106, 109, 110, 119, 124, 130, 137, 141, 144,
                        152, 153, 157, 161, 169, 176, 184, 188, 189,
                        193, 201, 202, 204, 206, 207, 210, 211, 212,
                        214, 218, 244, 246, 253, 256, 273, 276, 297
                       ]
df = df[df.Customer.isin(filtered_customer_id)]
df = df[['Customer','Energy_Consumption_kWh']]

data = df["Energy_Consumption_kWh"].to_numpy()
data.shape, data[0:5]


# Data Creation

train_test_ratio = 0.9
batch_size = 128

total_timesteps= 192
train_time_steps = 144
test_time_steps = total_timesteps - train_time_steps

daily_data = []
for i in range(0, data.shape[0], total_timesteps):
    daily_data.append(data[i:i+total_timesteps])
print(np.array(daily_data).shape)

length = int(len(daily_data)*train_test_ratio)
# random.shuffle(daily_data)
print(len(daily_data), length)
train_data = daily_data[0:length]
test_data= daily_data[length:-1]
train_data = np.array(train_data)
test_data = np.array(test_data)
print("Train and Test Data shapormalizing/standardizing:", train_data.shape, test_data.shape)

# Train, Validate, Test Splits

# scaler=MinMaxcaler(feature_range=(0,1))
# train_data=scaler.fit_transform(train_data.reshape(-1,1))
#
scaler=StandardScaler()
train_data=scaler.fit_transform(train_data.reshape(-1,1))

train_data = train_data.reshape(-1,total_timesteps)
print("Train Data shape after normalizing/standardizing:", train_data.shape)

test_data=scaler.transform(test_data.reshape(-1,1))

test_data = test_data.reshape(-1,total_timesteps)
print("Train and Test Data shape before normalizing/standardizing:", train_data.shape, test_data.shape)

print("Before Validation Data: train vs test", train_data.shape, test_data.shape)

length = int(len(train_data)*train_test_ratio)
random.shuffle(train_data)
val_data= train_data[length:]
train_data = train_data[0:length]
print("After Validation Data (from train data): train vs validation vs test", train_data.shape, val_data.shape, test_data.shape)

#Data Preprocessing

batch_size = 64

 X_train, X_val, X_test = ready_X_data(train_data, val_data, test_data, train_time_steps)

y_train_max , y_val_max, y_test_max  = ready_y_data(train_data, val_data, test_data, train_time_steps)

X_train_max, y_train_max = extend_last_batch(X_train, y_train_max)
X_val_max, y_val_max = extend_last_batch(X_val, y_val_max)
X_test_max, y_test_max = extend_last_batch(X_test, y_test_max)

X_train_full_max = torch.cat((X_train_max, X_val_max), 0).to(device)
y_train_full_max = torch.cat((y_train_max, y_val_max), 0).to(device)

X_train_max.shape, y_train_max.shape, X_val_max.shape, y_val_max.shape, X_test_max.shape, y_test_max.shape

# Plotting Data and Learning/Checking global mu, sigma, xi using y

plot_histogram(y_train_max, plot_name = "y")

shape, loc, scale = genextreme.fit(y_train_max.cpu())
print(f"Scipy Estimated GEV Parameters: mu: {loc}, sigma: {scale}, xi: {- shape}")
calculate_nll(y_train_max.cpu(), torch.tensor(loc), torch.tensor(scale), torch.tensor(-shape), name= "Scipy estimated parameters")


#Training and Results

torch.use_deterministic_algorithms(False)
batch_size = 128
sequence_len = train_time_steps
n_features = 1
n_hidden =  20
n_layers = 3

#persistent

y_persistent_hat = torch.zeros(X_test_max.shape[0])
for i, x in enumerate(y_test_max):
  y_persistent_hat[i] = torch.max(X_test_max[i])
print("RMSE of y (standardized): ", ((y_test_max - y_persistent_hat) ** 2).mean().sqrt().item())
inverted_y, inverted_yhat = inverse_scaler(y_test_max.reshape(-1).tolist(), y_persistent_hat.reshape(-1).tolist())
print("RMSE of y : ", math.sqrt(mean_squared_error(inverted_y,inverted_yhat)))
print("Correlation between actual and Predicted (mean): ", calculate_corr(y_persistent_hat,y_test_max))
# print("Correlation between actual and Predicted (mean): ", calculate_corr(np.array(inverted_y), np.array(inverted_yhat)))
plot_scatter(inverted_y, inverted_yhat, model_name="Persistence")


#FCN
count_constraint_violation = []
batch_size = 128
lr = 0.001
n_hidden = 10
n_layers = 3
num_epochs = 30
boundary_tolerance = 0.1
train_history = [0] * num_epochs
validation_history = [0] * num_epochs
test_history = [0] * num_epochs

model_fcn_all, mu_hat_all, sigma_hat_all, xi_hat_all, y_all, yhat_all, y_q1_all, y_q2_all = train_model_gev(lambda_ = 0.0, lambda_2=0.5, model_name = "FCN", tuning=False, validation=True, X_train = X_train_max, y_train = y_train_max)
plot_losses(train_history, validation_history, test_history)

print("RMSE of y (standardized): ", ((y_all - yhat_all) ** 2).mean().sqrt().item())
inverted_y, inverted_yhat = inverse_scaler(y_all.reshape(-1).tolist(), yhat_all.reshape(-1).tolist())
print("RMSE of y : ", math.sqrt(mean_squared_error(inverted_y,inverted_yhat)))
print("Correlation between actual and Predicted (mean): ", calculate_corr(y_all, yhat_all))
plot_scatter(inverted_y, inverted_yhat, model_name="Model FCN: y estimations")

#LSTM
count_constraint_violation = []
batch_size = 128
lr = 0.001
n_hidden = 10
n_layers = 3
num_epochs = 30
boundary_tolerance = 0.1
train_history = [0] * num_epochs
validation_history = [0] * num_epochs
test_history = [0] * num_epochs

model_fcn_all, mu_hat_all, sigma_hat_all, xi_hat_all, y_all, yhat_all, y_q1_all, y_q2_all = train_model_gev(lambda_ = 0.0, lambda_2=0.5, model_name = "LSTM_GEV", tuning=False, validation=True, X_train = X_train_max, y_train = y_train_max)
plot_losses(train_history, validation_history, test_history)


print("RMSE of y (standardized): ", ((y_all - yhat_all) ** 2).mean().sqrt().item())
inverted_y, inverted_yhat = inverse_scaler(y_all.tolist(), yhat_all.tolist())
print("RMSE of y : ", math.sqrt(mean_squared_error(inverted_y,inverted_yhat)))
print("Correlation between actual and Predicted (mean): ", calculate_corr(y_all, yhat_all))
plot_scatter(inverted_y, inverted_yhat, model_name="Model LSTM: y estimations")

#Transformer
count_constraint_violation = []
batch_size = 128
lr = 0.001
num_epochs = 10
boundary_tolerance = 0.1
train_history = [0] * num_epochs
validation_history = [0] * num_epochs
test_history = [0] * num_epochs

model_fcn_all, mu_hat_all, sigma_hat_all, xi_hat_all, y_all, yhat_all, y_q1_all, y_q2_all = train_model_gev(lambda_ = 0.0, lambda_2=0.5, model_name = "Trans", tuning=False, validation=True, X_train = X_train_max, y_train = y_train_max)
plot_losses(train_history, validation_history, test_history)

print("RMSE of y (standardized): ", ((y_all - yhat_all) ** 2).mean().sqrt().item())
inverted_y, inverted_yhat = inverse_scaler(y_all.tolist(), yhat_all.tolist())
print("RMSE of y : ", math.sqrt(mean_squared_error(inverted_y,inverted_yhat)))
print("Correlation between actual and Predicted (mean): ", calculate_corr(y_all, yhat_all))
plot_scatter(inverted_y, inverted_yhat, model_name="Model Transformer: y estimations")

#DeepPIPE

count_constraint_violation = []
batch_size = 128
lr = 0.01
n_hidden = 10
n_layers = 2
num_epochs = 30
boundary_tolerance = 0.1
train_history = [0] * num_epochs
validation_history = [0] * num_epochs
test_history = [0] * num_epochs

model_fcn_all, mu_hat_all, sigma_hat_all, xi_hat_all, y_all, yhat_all, y_q1_all, y_q2_all = train_model_gev(lambda_ = 0.0, lambda_2=0.5, beta=1.5, gamma=2.0, model_name = "DeepPIPE", tuning=False, validation=True, X_train = X_train_max, y_train = y_train_max)
plot_losses(train_history, validation_history, test_history)

#Model M3 - DeepExtrema

count_constraint_violation = []
batch_size = 128
lr = 0.001
n_hidden = 10
n_layers = 2
num_epochs = 8
boundary_tolerance = 0.1
train_history = [0] * num_epochs
validation_history = [0] * num_epochs
test_history = [0] * num_epochs

model_fcn_all, mu_hat_all, sigma_hat_all, xi_hat_all, y_all, yhat_all, y_q1_all, y_q2_all = train_model_gev(lambda_ = 0.1, lambda_2=0.9, model_name = "M3_GEV", tuning=False, validation=True, X_train = X_train_max, y_train = y_train_max)
plot_losses(train_history, validation_history, test_history)
calculate_nll(y_test_max, mu_hat_all, sigma_hat_all, xi_hat_all, name = "Model M3 Estimation (y)")
all_result(y_all.cpu(), yhat_all.cpu(), y_q1_all.cpu(), y_q2_all.cpu(), mu_hat_all.cpu(), sigma_hat_all.cpu(), xi_hat_all.cpu(), model_name="M3_l1=0.5_l2=0.9,hidden=10,layer=2, epochs=30")
