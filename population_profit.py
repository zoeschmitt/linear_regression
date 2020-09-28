
from google.colab import drive

drive.mount('/content/drive', force_remount=True)

FOLDERNAME = 'data/'

assert FOLDERNAME is not None, "[!] Enter the foldername."

# %cd drive/My\ Drive
# %cp -r $FOLDERNAME ../../
# %cd ../../

# load required library
import matplotlib.pyplot as plt
import numpy as np

"""Load data
________

This data contains two columns ['population','profit'], we are trying to predict the relationship between 'population','profit'.
"""

from cs4347.assignment_datasets import assign1

data = assign1()
print(data)

"""Visualize data
___________
"""

# draw the raw data plot
def draw_data(data):
    # parse data
    x = data['population']
    y = data['profit']

    #########################################################################
    # TODO:                                                                 #
    # 1. make a scatter plot of the raw data                                #
    # 2. set title for the plot                                             #
    # 3. set label for x,y axis                                             #
    # e.g.,                                                                 #
    #https://matplotlib.org/3.2.0/api/_as_gen/matplotlib.pyplot.scatter.html#
    #########################################################################
    
    plt.scatter(x, y)
    plt.xlabel("Population")
    plt.ylabel("Profit")


    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################

    # return the plt object
    return plt

plt = draw_data(data)
plt.show()

"""Cost function / Loss function / Objective function
___________
"""

# define cost_function j
def cost_function(theta, x, y):
    #########################################################################
    # TODO:                                                                 #
    # 1. implement the L2 Loss function                                     #
    # 2. Average the cost over the dataset size                             #
    # Hint: Use numpy functions                                             #
    #########################################################################
    
    j = np.sum((y - (theta[0] + x.T[1]*theta[1]))**2) / len(y)
    
    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################
  # return the cost
    return j

"""Gradient descent
____________________
"""

# gradient descent function
def gradient_descent(theta,x,y):
    # define your learning_rate and epoch
    lr = 0.01
    epoch = 1000
    
    # define cost
    cost = []
    
    # for loop
    for i in range(epoch):
        #########################################################################
        # TODO:                                                                 #
        # 1. update theta using lr                                              #
        # 2. append the uodated cost to cost list                               #
        # Hint: Use np.ravel to flatten arrays                                  #
        #########################################################################

        dm = np.sum((1/len(y)) * np.sum((np.dot(x,theta))- y))
        db = np.sum((1/len(y)) * (np.dot(x, theta) - y) * x[:,1])

        theta[0] -= lr * dm
        theta[1] -= lr * db

        theta = np.array([theta[0],theta[1]])
        cost.append(cost_function(theta, x, y))
        
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################
        
    # return updated theta and cost
    return theta, cost

"""Visualiza cost
_________
"""

# draw the cost for each iteration
def draw_iteration(cost, epoch=1000):
    #########################################################################
    # TODO:                                                                 #
    # 1. plot the cost for each iteration                                   #
    # 2. set title and labels for the plot                                  #
    # Hint: Use ply.plot function to plot and range(n)                      #
    #########################################################################
    #for i in range(epoch):
    x = np.arange(0,1000)
    plt.plot(x, cost)
    plt.title('cost per iteration')
    plt.xlabel('iteration')
    plt.ylabel('cost')
      
      
    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################
    
    # show plot
    
    plt.show()

# draw the simple linear regression model
def draw_final(theta,data):
    # draw the raw data first
    plt=draw_data(data)
    
    # define range of x
    x=np.arange(4,25,0.01)
    # draw the straight line using the final version of theta
    # y = a * x + b
    y=theta[0] + x * theta[1]
    
    # make plot and show
    plt.plot(x, y, c='r')
    plt.title('final')
    plt.show()

"""Call function
_____________
"""

# read the data
# 'population','profit'
data = assign1()
# print head to check if data is correct
print(data.head())

# draw raw data
plt = draw_data(data)
plt.show()

x = data['population']
#print("x: ", x)

y = data['profit']
#print("y: ", y)

x = np.c_[np.ones(x.size), x]

#print(x.shape)
#print(x.T.shape)
#print(y.shape)

#print("x2: ", x)
#print("x2: ", x.T)
#print("x2: ", X)
# if you don't understand what does the function c_ do, try c_?
# c_: _c stacks 1D array into 2D as columns
# ones: adds ones in array

# initialize theta
theta = np.ones(x.shape[1])
#print("theta: ", theta)

# calculate cost j by calling the function
j = cost_function(theta, x, y)
#print(j)

# gradient descent to find the optimal fit
theta, cost = gradient_descent(theta, x, y)

#print(cost)
#print(theta)

# draw the cost change for iterations
draw_iteration(cost)

# draw the final linear model
# it is shown as a red line, you can change the color anyway
draw_final(theta, data)

