
from __future__ import division
import numpy as np
import numpy.linalg as la
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib import cm
import random
import time


#Function to compute euclidean distances between the test data and training data.
def calc_nearest(test_data,train_data,train_labels):

    sub=np.subtract(train_data,test_data)
    mult=np.square(sub)
    sum=np.sum(mult,1)
    sqrt=np.sqrt(sum)
    min=np.argmin(sqrt)
    result=train_labels[min]
    return result

#Function to compute the matrix of data points categorized by their labels.
def calc_label_matrix(train,labels):
    indices_0 = np.array(np.where(labels == 0)[0])
    indices_1= np.array(np.where(labels==1)[0])
    indices_2= np.array(np.where(labels==2)[0])
    indices_3=np.array(np.where(labels==3)[0])
    indices_4 = np.array(np.where(labels ==4)[0])
    indices_5 = np.array(np.where(labels ==5)[0])
    indices_6 = np.array(np.where(labels ==6)[0])
    indices_7=np.array(np.where(labels ==7)[0])
    indices_8=np.array(np.where(labels ==8)[0])
    indices_9 = np.array(np.where(labels ==9)[0])
    mat=np.array([indices_0,indices_1,indices_2,indices_3,indices_4,indices_5,indices_6,indices_7,indices_8,indices_9])
    return mat

#Function to calculate nearest neighbours using prototype selection.
def calc_nearest_prototype(sample_size,test_data,train,labels,mat):
    train=np.array(train.astype('float'),float)
    labels=np.array(train_labels,float)
    proto_size=sample_size/10
    indices=[]
    for i in range(10):
        sel= random.sample(xrange(mat[i].size), int(proto_size))
        indices.append(mat[i][sel])
    indices=np.ravel(indices)
    prototype_train=train[indices]
    prototype_labels=labels[indices]
    result=np.apply_along_axis(calc_nearest,1,test_data,prototype_train,prototype_labels)
    return result

#Funciton to calculate the Error Rate.
def calc_error_rate(result,testlabels):
    diff=result-testlabels
    errors=np.count_nonzero(diff)
    error_rate=(errors/len(testlabels))*100
    return error_rate

#Master Function  which computes errors for all sample sizes spread over a number of trials. The trial sizze has been set to ten.
def compute_all_errors(sample_sizes,test_data,train_data,train_labels,test_labels):


    print("Prototype Selection: ")
    mat=calc_label_matrix(train_data,train_labels)
    trials=10
    errors=[]
    for s in sample_sizes:
        print("Sample: %s"%s)
        for i in range(trials):
            time_start=time.time()
            print("Iteration: %s"%(i+1))
            res=calc_nearest_prototype(s,test_data,train_data,train_labels,mat)
            error_rate=calc_error_rate(res,test_labels)
            errors.append(error_rate)
            print("Error: %s"%error_rate)
            print("--- %s seconds ---" % (time.time() - time_start))
    return errors

#Function that is used to plot the means and standard deviations calculated from the errors.
def plot(sample_sizes,errors):
    errors_array=np.array(errors).reshape(4,10)
    means=np.mean(errors_array,1)
    std_devs=np.std(errors_array,1)
    plt.figure()
    plt.errorbar(sample_sizes,means,std_devs,0)
    plt.title("Plot for the Learning Curve")
    plt.show()



#General commands to load data and set the environment up. Subsequently calls compute all errors. 
ocr=loadmat('ocr.mat')
train_data=ocr['data']
test_data=ocr['testdata']
train_labels=ocr['labels']
test_labels=ocr['testlabels']

sample_sizes = np.array([1000, 2000, 4000, 8000])
start_time=time.time()
test_results=compute_all_errors(sample_sizes,test_data,train_data,train_labels,test_labels)
print("Total Time--- %s seconds ---" % (time.time() - start_time))
print test_results
plot(sample_sizes,test_results)