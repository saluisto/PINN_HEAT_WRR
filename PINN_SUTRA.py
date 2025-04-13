# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 10:29:55 2025

@author: svenf
"""

"""
Created on Sun Feb  2 11:09:25 2025

@author: svenf
"""

#pip install deepxde
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import tensorflow as tf
from deepxde.callbacks import Callback
import os
import seaborn as sns

# Custom activation function using Fourier features
def fourier_activation(x):
    return tf.math.sin(x)

def read_data(filename):
    df = pd.read_csv(filename)
    obs_data = df.dropna()
    obs_data=obs_data.drop(index=range(4))
    return obs_data

class DynamicLossWeights(Callback):
    def __init__(self, initial_weights , print_every=1000):
        """
        Initializes the callback with initial weights for the loss terms.

        Args:
            initial_weights (list or array): Initial weights for each loss term.
        """
        self.weights = np.array(initial_weights)
        self.model = None  # Placeholder for the model instance
        self.print_every = print_every

    def on_train_begin(self):
        """
        Ensures the model is properly initialized at the start of training.
        """
        if not hasattr(self, "model"):
            raise ValueError("The model must be set before training begins.")

    def on_epoch_end(self):
        """
        Updates the loss weights dynamically at the end of each training epoch.
        """
        if self.model is None:
            raise ValueError("Model has not been set for the DynamicLossWeights callback.")

        # Access the current training losses
        losses = np.array(self.model.train_state.loss_train)
        if np.any(losses == 0):
            print("Warning: Zero loss encountered, skipping weight update.")
            return

        # Compute new dynamic weights
        norm_losses = losses / np.sum(losses)  # Normalize losses
        self.weights = 1 / (norm_losses + 1e-6)  # Inverse of normalized loss
        self.weights /= np.sum(self.weights)  # Normalize weights to sum to 1

        # Debugging output
        epoch = self.model.train_state.epoch
        if epoch % self.print_every == 0:
            
            print(f"Epoch {self.model.train_state.epoch}: Updated weights: {self.weights}")
            print('......................................................................')
        # Apply the updated weights to the model
        self.model.loss_weights = self.weights.tolist()

class PrintLossLR(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.epoch_counter = 0  # Track epochs manually

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_counter += 1
        if self.epoch_counter % 1000 == 0:
            lr = self.model.optimizer._decayed_lr(tf.float32).numpy()
            loss = logs.get('loss')
            print(f"Epoch {self.epoch_counter}: Loss = {loss:.6f}, Learning Rate = {lr:.6e}")

class StopWhenLossBelowThreshold(dde.callbacks.Callback):
    def __init__(self, threshold):
        self.threshold = threshold
        self.model = None  # Will be assigned later

    def on_train_begin(self):
        """Assign model reference when training begins."""
        self.model = self.model  # Ensures self.model is properly assigned

    def on_epoch_end(self):
        """Check loss at end of epoch and stop training if below threshold."""
        if self.model is not None:
            loss_array = np.array(self.model.train_state.loss_train)  # Convert loss to numpy array
            total_loss = np.sum(loss_array)  # Option 1: Use total loss
            
            # Alternative: Use mean loss instead
            # total_loss = np.mean(loss_array)  
            
            if total_loss < self.threshold:
                print(f"Stopping training: Total loss {total_loss:.6f} is below threshold {self.threshold}")
                self.model.stop_training = True  # Stop training


df=read_data(filename='SUTRA_meso.csv')
gPINN=0 #flag for PINN or gPINN

df['time'] = pd.to_datetime(df['datetime'], format="%d/%m/%Y %H:%M:%S")
df = df.drop('d_1.0', axis=1)
first_time = df['time'].min()
df['time'] = df['time'] - first_time

df=df.iloc[750:]

# convert from days into seconds
df['time'] = df['time'].dt.total_seconds()/(60*60*24)
tmaxx=np.float32(df['time'].max())
tmax = dde.Variable(tmaxx)

time = df['time'].values.reshape(-1,1)

df.set_index('time', inplace=True)

# observation depths
observe_x = np.array([0,0.05,0.1,0.15,0.2,0.3,0.4,0.5]).reshape(-1,1)
observe_T = df[['d_0','d_0.05','d_0.1','d_0.15','d_0.2','d_0.3','d_0.4', 'd_0.5']].values

LL=np.float32(observe_x[len(observe_x)-1])
L = dde.Variable(LL)


df.plot()
ax=df.plot()
ax.set_xlabel("Time [days]")
ax.set_ylabel("Temperature [°C]")
#%%
# In this section the part of the objective function is beeing defined that is associated withe the pdf (heat transport equation)

def pde(x, y):

   # phi = 0.35 # porosity, unitless
    #roh_water=1028
    #roh_sed=2650
    #c_water=4068
    #c_sed= 792.45
    #ke=1.68*60*60*24 #ke is converted from J/(s m K) to J/(day m K)

    #Cw = c_water*roh_water
    #Csed = c_sed*roh_sed

    #Cs = phi*Cw + (1-phi)*Csed

    ke=2.63*60*60*24
    Cw=4.182e6
    Cs=2.814e6


    D=ke/(Cs)

    a=Cw/(Cs)


    #Output of the NN is the temperature field T and q fluxes in time
    T, q = y[:, 0:1], y[:, 1:2]

    #derivatives that are needed for the heat transport equation

    dy_t = dde.grad.jacobian(y, x, i=0, j=1)  #i=0 means component of y --> i=0 Temperature i=1 q; j=0 is z and j=1 time
    dy_xx = dde.grad.hessian(y, x, component=0, i=0, j=0) #compoent = 0 is Temperature component = 1 is q
                                                          #x i,j = 0 is depth z i,j = 1 is time t meant is xi,xj
    dq_dx=dde.grad.jacobian(y, x, i=1, j=0)
    dy_dx = dde.grad.jacobian(y, x, i=0, j=0)

    if gPINN == 1:
        dy_tx = dde.grad.hessian(y, x, component=0, i=1, j=0)
        dy_tt = dde.grad.hessian(y, x, component=0, i=1, j=1)

        dy_xxx =dde.grad.jacobian(dy_xx, x, i=0, j=0)
        dy_xxt =dde.grad.jacobian(dy_xx, x, i=0, j=1)

    
        dq_dt=dde.grad.jacobian(y, x, i=1, j=1)
        dq_xt = dde.grad.hessian(y, x, component=1, i=0, j=1)
        dq_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)

        dy_xt=dde.grad.hessian(y, x, component=0, i=0, j=1)

        return [dy_t-D*dy_xx+a*q*dy_dx+a*T*dq_dx,
                dy_tx-D*dy_xxx+a*dq_dx*dy_dx+a*q*dy_xx+a*dy_dx*dq_dx+a*T*dq_xx,
                dy_tt-D*dy_xxt+a*dq_dt*dy_dx+a*q*dy_xt+a*dy_t*dq_dx+a*T*dq_xt,
                dq_dx]
    else:
        return [dy_t-D*dy_xx+a*q*dy_dx+a*T*dq_dx,
                dq_dx]

#%%
#Define where observations are located and define IC and BC
xx, tt = np.meshgrid(observe_x,time)
X = np.vstack((np.ravel(xx), np.ravel(tt))).T #x,t


train_set=1
set_cout=int(train_set*len(tt))
X_train = np.vstack((np.ravel(xx[:set_cout]), np.ravel(tt[:set_cout]))).T #x,t
observe_T=np.ravel(observe_T[:set_cout]).reshape(-1,1)
observe_u = dde.icbc.PointSetBC(X_train,observe_T,component=0)

#%% Define the model gometry in space and time
max_time=tmaxx
geom = dde.geometry.Interval(0,LL)
timedomain = dde.geometry.TimeDomain(0, max_time)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

#%
data = dde.data.TimePDE(
    geomtime,
    pde,
    [observe_u],
    num_domain=0,
    anchors=X,
)

# Here different neural network structures are beeing tested PFNN are
# DeepONET networks /(https://www.nature.com/articles/s42256-021-00302-5)

net = dde.nn.PFNN([2, [40, 40], [40, 40], [40, 40], [40, 40], 2], fourier_activation, "Glorot uniform")

model = dde.Model(data, net)

# Define dynamic weights
if gPINN==1:
    initial_weights = [1, 1, 1, 1, 1]
else:
    initial_weights = [1, 1, 1]

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.75, 
                                                 patience=10, min_lr=1e-6, verbose=1)

# Custom loss weights
#weights = dde.callbacks.LossWeightsScheduler(loss_weights_callback)

# Define your loss threshold
loss_threshold = 6e-4  # Adjust this as needed
#iterations=100000

# Create the callback
stop_callback = StopWhenLossBelowThreshold(loss_threshold)
callback = DynamicLossWeights(initial_weights,print_every=1000)
#runs=7

model.compile("adam", lr=0.0001,loss_weights=[1, 1, 100])
losshistory, train_state = model.train(iterations=2000000,callbacks=[stop_callback])
#model.compile("adam", lr=0.0001,loss_weights=callback.weights)
#losshistory, train_state = model.train(iterations=50000,callbacks=[callback,stop_callback])
model.compile("L-BFGS-B",loss_weights=[1, 1, 1,])
losshistory, train_state = model.train()


#model.compile("adam", lr=0.0001,loss_weights=callback.weights)
#losshistory, train_state = model.train(iterations=10000,callbacks=[callback,stop_callback])
#losshistory, train_state = model.train(iterations=iterations,callbacks=[callback, PrintLossLR()])
#losshistory, train_state = model.train(iterations=iterations,callbacks=[reduce_lr])


# Parameters for learning rate reduction
reduce_lr_factor = 0.5  # Factor by which to reduce the learning rate
reduce_lr_patience = 10  # Number of epochs to wait before reducing LR
min_learning_rate = 1e-6  # Minimum allowed learning rate
best_loss = np.inf  # Track the best loss
wait = 0  # Counter for epochs without improvement

dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# Save the model's weights
#model.save("backup\model.ckpt")

Y = model.predict(X)
#%%
#Plotting the different predictions vs observations

for i in range(len(observe_x)):
    dp = ['d_0','d_0.05','d_0.1','d_0.15','d_0.2','d_0.3','d_0.4', 'd_0.5']
    columns = ['X[m]', 'Time[days]', 'T[C]', 'q[m/day]']
    sim=np.concatenate((X,Y),axis=1)
    result = pd.DataFrame(sim, columns=columns)

    rslt_d1 = result[result['X[m]'] == float(observe_x[i])]
    rslt_d1= rslt_d1.drop('X[m]',axis=1)
    #rslt_d1['Time[days]'] = rslt_d1['Time[days]']* tmaxx

    rslt_d1.plot(x='Time[days]', y='T[C]', kind='scatter', title='Temp_simulated')
    df[dp[i]].plot(kind='line', title=dp[i],color='r')

    rslt_d1.plot(x='Time[days]', y='q[m/day]', kind='line', title='Temp_simulated')
