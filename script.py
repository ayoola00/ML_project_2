import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A k x d matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    temp_mean = []
    class_num = np.unique(y) #returns how many classes
    
    for item in class_num:
        x_tem = X[np.where(y.flatten()) == item, :]
        m =np.mean(x_tem, axis= 0)
        temp_mean.append(m)
    
   
    means= np.transpose(np.asarray(temp_mean))
    
    big_mean = np.mean(X, axis=0)
    
    
    fixed_mean = []
   
    for item in class_num:
        x_tem = X[np.where(y.flatten()) == item, :]
        fixed_mean.append(x_tem - big_mean)  
        
    covmats = []
    for item in class_num:
        y = (int(group)-1)
        covmats.append((np.dot(np.transpose(fixed_mean[y]),fixed_mean[y])/(fixed_mean[y].size/2))                                                                                        
   
  covmat= np.zero((covmats[0].shape))
    for item in class_num:
        y = (int(group)-1)               
        mult=(fixed_mean[y].size/2.0)/(X.size/2.0)
        covmat=covmat+((covmats[y]*mult)               
    # IMPLEMENT THIS METHOD 
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A k x d matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    temp_mean = []
    class_num = np.unique(y) #returns how many classes
    
    for item in class_num:
        x_tem = X[np.where(y.flatten()) == item, :]
        m =np.mean(x_tem, axis= 0)
        temp_mean.append(m)
    
   
    means= np.transpose(np.asarray(temp_mean))
    
    big_mean = np.mean(X, axis=0)
    
    
    fixed_mean = []
   
    for item in class_num:
        x_tem = X[np.where(y.flatten()) == item, :]
        fixed_mean.append(x_tem - big_mean)  
        
    covmats = []
    for item in class_num:
        x_tem = X[np.where(y.flatten()) == item, :]
        x_tem = x_tem - big_mean
        covmats.append((np.cov(x_temp, rowvar = 0))
    
    # IMPLEMENT THIS METHOD
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels
     invcov = inv(covmat)
     ypred = np.array([])
     for i in Xtest:
          det = np.array([])
          for index, m in enumerate(means):
              det = np.append(det, np.dot(np.dot((x-m),invcov), (x-m).t))
          if ypred.shape[0] == 0:
             ypred = np.array([np.argmin(det) +1])
          else:
               ypred = np.vstack((ypred,np.array([np.argmin(det)+1])))
    
     acc = np.mean((ypred==ytest).astype(float))*100
    # IMPLEMENT THIS METHOD
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels
     invcov = inv(covmat)
     ypred = np.array([])
     for i in Xtest:
          det = np.array([])
          for index, m in enumerate(means):
              det = np.append(det, np.dot(np.dot((x-m),invcov), (x-m).t))
          if ypred.shape[0] == 0:
             ypred = np.array([np.argmin(det) +1])
          else:
               ypred = np.vstack((ypred,np.array([np.argmin(det)+1])))
    
     acc = np.mean((ypred==ytest).astype(float))*100
                       
    # IMPLEMENT THIS METHOD
    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
    b = np.dot(X.T,X)
    c = np.dot(X.T,y)
    a = np.linalg.inv(b)
    w = np.dot(a, c)  
    # IMPLEMENT THIS METHOD                                                   
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD
    xx = np.dot(X.transpose(),X)
    xy = np.dot(X.transpose(),y)
    sigma = np.eye(xx.shape[0]) * X.shape[0] * lambd
    temp = np.linalg.inv((sigma + xx))
    w = temp.dot(xy)
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    
    x_w = np.dot(Xtest, w)
    
    a_min = (ytest - x_w)   
    a_min = np.dot(a_min.T,a_min)
    
    a_sum = a_min.sum(axis=0)
    a_sum = np.sqrt(a_sum)
    
    mse = a_sum/(Xtest.shape[0]);
    
    # IMPLEMENT THIS METHOD
    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD
    tempw = np.reshape(w,(w.shape[0],1))
    xi = np.dot(X.transpose(),X)
    yi = np.dot(y.transpose(),X)
    grad = (((np.dot(tempw.transpose(),xi))-yi)/X.shape[0]) + (lambd * tempw.transpose())

    yx = y - (X.dot(tempw))

    temp = (np.dot(yx.transpose(),yx))/(2*X.shape[0])
    error = temp + ((lambd/2) * np.dot(tempw.transpose(),tempw))
    error = error.flatten()

    error_grad = grad

    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xp - (N x (p+1)) 
	
    # IMPLEMENT THIS METHOD
    # .shape[0] will give us the column vector (cv) with correct dimensions.
    # The vector of p attributes (Xp), is initially filled with ones.
    cv = x.shape[0] 
    Xp = np.ones((cv, p+1))
    for i in range(cv):
        for j in range(p+1):
            if j!= 0:
                Xp[i][j] = x[i]**j 
    return Xp

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('QDA')

plt.show()
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()


# Problem 5
pmax = 7
lambda_opt = 0 # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()

