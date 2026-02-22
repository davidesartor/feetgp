import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from sklearn.metrics import r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gp
import os
import pandas as pd
from einops import rearrange
import pickle

from time import time

os.makedirs("test4", exist_ok=True)
jax.config.update("jax_enable_x64", True)

st = time()

############################################################
# Parameters
############################################################
#realdata = False
realdata = True
#warmstart = False
warmstart = True
warmzu = True
reverse = False
#reverse = True
#markers = 2
#markers = 3
#markers = 4
#markers = 5
#markers = 6
#markers = 7
#markers = 8
markers = 13
subsample_factor = 100

normalize = True
detrend = False

I = 100
#I = 2
#admm_iters = 100
admm_iters = 500

if realdata:
    #lambdas = 10 ** jnp.linspace(3, 3.5, I)
    #lambdas = 10 ** jnp.linspace(2, 4, I)
    lambdas = 10 ** jnp.linspace(0, 4, I)
    #lambdas = 10 ** jnp.linspace(2, 5, I)
else:
    lambdas = 10 ** jnp.linspace(0, 5, I)

############################################################
# Variants
############################################################

if realdata:
    path = "OneDrive_1_03-02-2026/Left_bf_10kmh_Filtered.txt"
    df = pd.read_csv(path, sep="\t", skiprows=[0, 2, 3, 4], index_col=1)
    df = df.drop(columns=["Unnamed: 0"])
    df.reset_index(inplace=True)
    print(df.shape)
    x = df.iloc[:, 1:].values
    x = x[~np.isnan(x).any(axis=1)]
    # subsample and make a mock regression problem
    x = jnp.array(x[::subsample_factor, : markers * 3])
    x_train, x_test = x[::2, :], x[1::2, :]
    y_train, y_test = x[::2, :], x[1::2, :]
    print("train:", x_train.shape, y_train.shape)
    print("test:", x_test.shape, y_test.shape)
    n, d = x_train.shape
    n, o = y_train.shape
else:
    gg = np.linspace(0,1,num=100)
    gg2 = np.square(gg)
    assert M==2
    x_train = np.stack([gg,gg,gg,np.sin(2*np.pi*gg),np.cos(2*np.pi*gg),np.sin(4*np.pi*gg)], axis = 1)
    y_train = y_test = x_test = np.copy(x_train)

M = x_train.shape[1]//3

#if detrend:
#    print("detrending!")
#    for i in range(y_train.shape[1]):
#        #np.linalg.ols

if normalize:
    y_test = (y_test - jnp.mean(y_train,axis=0)[None,:]) / jnp.std(y_train,axis=0)[None,:]
    y_train = (y_train - jnp.mean(y_train,axis=0)[None,:]) / jnp.std(y_train,axis=0)[None,:]

    x_test = (x_test - jnp.min(x_train,axis=0)[None,:]) / (jnp.max(x_train,axis=0)-jnp.min(x_train,axis=0))[None,:]
    x_train = (x_train - jnp.min(x_train,axis=0)[None,:]) / (jnp.max(x_train,axis=0)-jnp.min(x_train,axis=0))[None,:]

print("Train points shape:", x_train.shape)  # (N_TRAIN, 2)
print("Test points shape:", x_test.shape)  # (N_GRID*N_GRID, 2)
print()

#############################################################
## Single run fun.
#############################################################
#model = gp.GaussianProcessRegressor(
#    l1_penalty=1e2,
#    max_iterations=100,
#    tollerance=1e-4,
#    verbose=True,
#    #init_theta=np.zeros([x_train.shape[1], x_train.shape[1]])
#)
#_ = model.fit(x=x_train, y=y_train)

############################################################
# Run model at high penalty to extract single sensor.
############################################################

if reverse:
    lambdas = lambdas[::-1]

model = gp.GaussianProcessRegressor(
    max_iterations=admm_iters,
    tollerance=1e-4,
    verbose=False,
    warmzu=warmzu,
    warmstart=warmstart,
)

thetas = np.nan*np.zeros([I,x_train.shape[1], x_train.shape[1]])
gns = np.nan*np.zeros([I,M])
for li,l in enumerate(lambdas):
    print(f"run {li} lam = {l}")
    #reinit = model.parameters.theta if warmstart and li > 0 else None
    _ = model.fit(x=x_train, y=y_train, l1_penalty=l) 
    thetas[li,:,:] = model.parameters.theta 
    gns[li,:] = jnp.sum(jnp.square(rearrange(thetas[li,:,:], "o (d k) -> (o k) d", k=3)), axis = 0)

with open("pickles/feb21.pkl",'wb') as f:
    pickle.dump([thetas, gns, lambdas], f)

#fig = plt.figure(figsize=[10,5])
#plt.subplot(1,2,1)
#plt.plot(lambdas, jnp.round(gns, 8))
#plt.xscale('log')
#plt.yscale('log')
#plt.subplot(1,2,2)
#for i in range(thetas.shape[1]):
#    plt.plot(lambdas, jnp.round(thetas[:,i],8))
#plt.xscale('log')
#plt.savefig("temp.pdf")
#plt.close()

print("Diff time:")
print(time() - st)
