"""
@Title:         Machine Learning Homework 2, Problem I
@Date:          Feruary 23, 2024
@author:        Jacob Sichlau
@Collaborator:  Michael Williams
@Collaborator:  Adim Seeler
"""

import numpy as np
import math
from matplotlib import pyplot as plt

training_data = np.loadtxt("flight_data_train.csv",delimiter=",")
test_data = np.loadtxt("flight_data_test.csv",delimiter=",")



# coeffs = np.polyfit(x, y, 0)  # Leveraging NumPy's polyfit as short-cut; 0 is the degree of the poly
# poly = np.poly1d(coeffs) # Polynomial with "learned" parameters
# plt.plot(xx, poly(xx), "b-")

# training_data = np.c_[ training_data, np.ones(np.size(training_data[:,0]))] # Add constant feature for all rows

'''
Column-wise Matrix Normalization
@input = A numeric matrix
@output = That matrix, with the values normalized column-wise
'''
def normalizer(matrix):
    for i in range(len(matrix[0])):
        col_max = np.max(matrix[:,i]);
        col_min = np.min(matrix[:,i]);
        matrix[:,i] = (matrix[:,i] - col_min) / (col_max - col_min)
    return matrix


'''
Phi Matrix Generator
@inputs = data, a matrix of inputs over many time steps
@inputs = m, the number of features
@output = the total Phi matrix
'''
def Phi_Builder(data, m):       # M is number of features, data is the input data

    Phi = []
    
    for row in range(len(data)):
        temp = Phi_Row(data[row,:-1],m)
        Phi.append(temp)
    return np.transpose(Phi);
    
'''
Phi_Row Generator
@inputs = data, one row of data containing all inputs for a time step
@inputs = m, the number of features 
@output = one row of the overall Phi matrix
'''
def Phi_Row(data,m):        # One row of data into one row of phi
    row = []
    for x in data:
        for i in range(m-1):
            row.append(x**(i+1))
    row.append(1)
    return row

'''
Root Mean Square Function
@inputs = data, experimental result
@inputs = ref, reference result
@output = the root mean square error of those two data
'''
def RMSE(data,ref):
    MSE = np.square(np.subtract(data,ref)).mean() 
    return math.sqrt(MSE)

training_data = normalizer(training_data)
test_data = normalizer(test_data)

training_output = training_data[:,-1]
test_output = test_data[:,-1]

'''
Weighted Training Output - with lnlambda 
@inputs = m, The number of features
@inputs = data, training data to use
@inputs = lnlambda, the hyperparameter used
@output = The output from training data inputs and weights
'''
def lambda_output(m, data, lnlambda):
    Phi_ = np.transpose(Phi_Builder(data, m))
    Phi_prod_ = np.matmul(np.transpose(Phi_),Phi_)
    I_ = np.identity(len(Phi_[0]))
    training_output_ = data[:,-1]

    firstterm = np.linalg.pinv(np.exp(lnlambda)*I_+Phi_prod_)      # lambda = exp(i)
    weight_ = np.matmul(firstterm,np.matmul(np.transpose(Phi_),training_output_))
    
    y_train_ = np.matmul(Phi_,weight_)
    return y_train_

'''
Cross Validation with n-folds
@inputs = cleanData, The data to use for the folds
@inputs = nFolds, number of folds
@inputs = m, number of features
@inputs = hyperparam, the hyperparameter used
@output = Average error from the cross validation technique
'''
def crossValidation(cleanData,nFolds,m,hyperparam):

    foldSize = int(len(cleanData)/nFolds)
    
    foldErrors = []
    
    for i in range(nFolds):
        data = cleanData
        fold = data[i*foldSize:i*foldSize+foldSize][:]
        for j in range(foldSize):
           data = np.delete(data,i*foldSize,0)
        exp_output = lambda_output(m,fold,hyperparam)
        training_output_ = data[:,-1]
        Phi_test_ = Phi_Builder(fold, m)
        Phi_train_ = Phi_Builder(data, m)

        weighting = np.matmul(np.linalg.pinv(np.transpose(Phi_train_)),training_output_)
        output_test = np.matmul(np.transpose(Phi_test_),weighting)
        foldErrors.append(np.abs(exp_output - output_test))
    return np.average(foldErrors)

'''
Part 1
'''

m_max = 15
RMSE_train = []
RMSE_test = []
for m in range(m_max):
    m += 1
    Phi = Phi_Builder(training_data, m)
    Phi_test = Phi_Builder(test_data,m)
    
    pseudoinverse = np.linalg.pinv(np.transpose(Phi))  
    weights = np.matmul(pseudoinverse,training_output)
    
    y_test = np.matmul(np.transpose(Phi_test),weights)
    y_train = np.matmul(np.transpose(Phi),weights)

    RMSE_test.append(RMSE(y_test,test_output))
    RMSE_train.append(RMSE(y_train,training_output))
    
fig = plt.figure()
fig, ax = plt.subplots()
ax.plot(RMSE_test, label='Test Error')
ax.plot(RMSE_train, label='Training Error')
ax.set_xlabel('m value')
ax.set_ylabel('Error')
ax.set_title("RMS Error vs. m")
ax.legend()

'''
Part 2
'''


min_ln_lambda = -30
max_ln_lambda = 20
ln_lambdas = np.linspace(min_ln_lambda,max_ln_lambda,max_ln_lambda-min_ln_lambda+1)

m = 6 
Phi_r = np.transpose(Phi_Builder(training_data, m))
Phi_r_test = np.transpose(Phi_Builder(test_data, m))

Phi_prod = np.matmul(np.transpose(Phi_r),Phi_r)

I = np.identity(len(Phi_r[0]))

weights_2 = []
RMSE_r_train = []
RMSE_r_test = []
E_d = []

for i in ln_lambdas:
    firstterm = np.linalg.pinv(np.exp(i)*I+Phi_prod)        # lambda = exp(i)
    temp = np.matmul(firstterm,np.matmul(np.transpose(Phi_r),training_output))
    weights_2.append(temp)
    
    y_r_test = np.matmul(Phi_r_test,temp)
    y_r_train = np.matmul(Phi_r,temp)
    
    RMSE_r_test.append(RMSE(y_r_test,test_output))
    RMSE_r_train.append(RMSE(y_r_train,training_output))

    E_d.append(RMSE(y_r_train,training_output))            # for part 3


fig_2 = plt.figure()
fig_2, aax = plt.subplots()
aax.plot(ln_lambdas, RMSE_r_test, label='Test Error')
aax.plot(ln_lambdas, RMSE_r_train, label='Training Error')
aax.set_xlabel('ln $\lambda$')
aax.set_ylabel('Error')
aax.set_title("RMS Error vs. ln $\lambda$")
aax.legend()

'''
Part 3
'''
cross_errors = []
AIC = []
BIC = []
shuffled = training_data
np.random.shuffle(shuffled)
N = len(Phi[0])

count = 0;
for i in ln_lambdas:
    
    cross_errors.append(crossValidation(shuffled, 10, 6, i))
    
    temp = np.linalg.pinv(Phi_prod + np.exp(i)*I)       # lambda = exp(i)
    pre_gamma = np.matmul(temp, Phi_prod)
    gamma = np.trace(pre_gamma)
    
    AIC.append(N*np.log(E_d[count]/N)+gamma)
    BIC.append(N*np.log(E_d[count]/N)+gamma*np.log(N))
    count += 1

fig_3 = plt.figure()
fig_3, aaax = plt.subplots()
aaax.plot(ln_lambdas, cross_errors, label='Cross Validation Error ($N_{folds} = 10$)')
aaax.set_xlabel('ln $\lambda$')
aaax.set_ylabel('Error')
aaax.set_title("10-Fold Cross Validation")
aaax.legend()

fig_4 = plt.figure()
fig_4, ax = plt.subplots()
ax.plot(ln_lambdas, AIC, label='AIC')
ax.plot(ln_lambdas, BIC, label='BIC')
ax.set_xlabel('ln $\lambda$')
ax.set_ylabel('Information Criteria')
ax.set_title("Akaike and Bayesian Information Criteria")
ax.legend()

cross_min_lambda = np.exp(ln_lambdas[np.where(cross_errors == min(cross_errors))])
print(f'The lambda with the lowest 10-fold cross validation error is {cross_min_lambda}')

AIC_min_lambda = np.exp(ln_lambdas[np.where(AIC == min(AIC))])
print(f'The lambda with the lowest AIC is {AIC_min_lambda}')

BIC_min_lambda = np.exp(ln_lambdas[np.where(BIC == min(BIC))])
print(f'The lambda with the lowest BIC is {BIC_min_lambda}')