import numpy as np   # for array operations
import os,sys       #for file management
import math
import matplotlib.pyplot as plt
"""
     devloper --> sayansree paria
     this script shows the implemention of gradient decent algorithm from scrach
     requiers basic knowledge of calculus,probablity and statistic
     and working of simple machine learning models
     this focus on only 2D Tensor of matrix form of data
     efficiency of O(n*d*l) n= no of iteration d is degree of mean function and l is size of data set
"""
#calculate error in form of variance 
# note not to be confused deviation
# to calculate deviation use standard deviation from variance (rms value of errors)  
# points are array of data sample and coff contains polinomial coefficient
def variance(points,coff):                # points contains 2D representation of data 2D Tensor(matrix)
    var = 0.0
    for i in range(len(points)):
        xi = points[i,0]                  # x cordinate of the i th sample data
        yi = points[i,1]                  # y cordinate of the i th sample data
        var += (yi - f(xi,coff) )**2      #f(xi) is mean function, we will try to increse its efficiency through gradient decent
    return var/(i+1)                      #put probabliy of 1/N for each X=xi or P(x=i)=1/N (assuming events independent and equally likely for simplicity)

# calculate standard deviation of errors
def standard_deviation(points,coff):
    return math.sqrt(variance(points,coff))

#read file containing 'points' from data.csv
def read():
    points=[]
    with open(os.path.abspath(os.path.dirname(sys.argv[0]))+'\\data.csv','r') as file:  # open the data file from the location of script
        for i in file.readlines():                                                      #extract each lines
            points.append([float(i.split(',')[0]), float(i.split(',')[1][:-1])])        # convert the coma seperated file to list of list format
    points = np.array(points)                                                           # convert to 2D array Matrix or 2D Tensor
    return points

# calculates the gradiant to minimise error or variance
# finds partial derivative of variance with respect to each of the polynomial coefficients
# points contain data samples and coff contains polynomial deg is its degree
def find_gradient(points,coff,deg):
    exp_grad_list=np.array([0.0]*deg)                                                  # array containing gradient of different polynomial coefficients
    for i in range(len(points)):                                                       # iterete through all the sample data
        xi=points[i,0]                                                                 # x cordinate of the i th sample data
        yi=points[i,1]                                                                 # y cordinate of the i th sample data
        for d in range(deg):                                                           # iterate throuh all the polynomial coefficients
            exp_grad_list[d]+=df(xi,yi,d,coff)                                         # mean gradient using the gradient function as df(x)
    return exp_grad_list/(i+1)                                                         #put probabliy of 1/N for each X=xi or P(x=i)=1/N (assuming events independent and equally likely for simplicity)

# performs gradient decent algorithm on the the polynomial
# each time we try to minimise the errors finally reaching minima of errors
# note local minima cannot be differentiated from global minima so this may hinder algorithms performance
# points contain data samples and coff contains polynomial deg is its degree default is 1
# iterations denotes number of steps
# gamma denotes size of steps 
# lower gamma results in high accuracy but more time consuming 
# note large gamma value may result blowing numbers to infinity keep gamma < atleast 10^-4 
def gradient_decent(points,gamma,itreration,degree=1):
    coff=np.array([0.0]*(degree+1))   
    print('before gradient decent on degree {1} varies as {0}'.format(variance(points,coff),degree))    #print error before calculation
    for i in range(itreration):
        coff= coff - find_gradient(points,coff,degree+1)*gamma                                          #slowly update our variables to reduce error
    print('before gradient decent on degree {1} varies as {0}'.format(variance(points,coff),degree))    #print after before calculation
    return coff

# defination of our polynomial mean function
# this returns the current expected valure of y or f(x) povided x  at any point
# x is the x cordinate maping of dataset and coff is the polynomial coeficients 
def f(x,coff):
    X=np.array([x**i for i in range(len(coff))])            #calculate expected y given x
    return X.dot(coff)

# defination of our polynomial mean gradient function
# this returns the current expected valure of dy/dk or df(x)/dk povided x  at any point
# x is the x cordinate maping of dataset y is the y cordinate maping of dataset 
# coff is the polynomial coeficients and d is its degree
def df(x,y,d,coff):
    return 2*(f(x,coff)-y)*x**d                     #calculate differentiation on variance wrt to 'decent' step


# -----------------------------------------------test run on diffferent polynomial---------------------------------
points=read()
x = np.linspace(-10,100,300) 
plt.xlim(0,90)  
plt.ylim(0,150)                                                      # retrive the dataset from data.csv file
for d,sample in enumerate(points):
    plt.scatter(sample[0],sample[1],s=10,linewidths=1,c='blue',marker='*')
print('\n\ncoefficients are')
w = gradient_decent(points,1.0e-4,1000,1)
print(w)                         # gradient decent on leniar function
plt.plot(x, w[1]*x+w[0] + 0, '--g')

print('\ncoefficients are')
w = gradient_decent(points,1.0e-8,100,2)
print(w)                         # gradient decent on quadratic function
plt.plot(x,w[2]*x*x + w[1]*x+w[0] + 0, ':c')

print('\ncoefficients are')
w = gradient_decent(points,1.0e-11,100,3)
print(w)                        # gradient decent on cubic function
plt.plot(x,w[3]*x*x*x+w[2]*x*x + w[1]*x+w[0] + 0, '-.r')

print('\ncoefficients are')
w = gradient_decent(points,1.0e-15,100,4)
print(w)                        # gradient decent on biquadratic function
plt.plot(x,w[4]*x*x*x*x+w[3]*x*x*x+w[2]*x*x + w[1]*x+w[0] + 0, '-.r')

plt.show()