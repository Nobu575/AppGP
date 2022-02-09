import GPy
import numpy as np
import matplotlib.pyplot as plt

# scatter plot for input (x,y) 
def plot_input(x,y):
    plt.scatter(x,y)
    plt.grid(True)
    plt.xlim(0,30)
    plt.ylim(0,30)
    plt.savefig('./fig_input.png')

# plot output of GPRegression
def plot_model(model, file_name):
    model.plot()
    plt.grid(True)
    plt.xlim(0,30)
    plt.ylim(0,30)
    plt.savefig(file_name)

# Input 
x = np.array([2.0, 3.0, 4.0, 8.0, 9.0, 11.0, 20.0, 24.0])
y = np.array([9.0, 11.0, 13.0, 11.0, 8.0, 6.0, 14.0, 20.0])
plot_input(x,y)

# Set kernel to linear
kernel = GPy.kern.Linear(1)
model = GPy.models.GPRegression(x[:,None], y[:,None], kernel=kernel)
model.optimize()
print(model)
plot_model(model, './fig_output_linear.png')

# Set kernel to RBF
kernel = GPy.kern.RBF(input_dim=1,variance=1)
model = GPy.models.GPRegression(x[:,None], y[:,None], kernel=kernel)
model.optimize()
print(model)
plot_model(model, './fig_output_rbf.png')

# Set kernel to Bias + Linear + RBF
kernel = GPy.kern.Bias(1) + GPy.kern.Linear(1) + GPy.kern.RBF(input_dim=1,variance=1)
model = GPy.models.GPRegression(x[:,None], y[:,None], kernel=kernel)
model.optimize()
print(model)
plot_model(model, './fig_output_mix.png')
