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

#Function to calculate nearest neighbours.
def nearest_neighbours(sample_size,test_data,train_data,train_labels):
    sel = random.sample(xrange(60000), sample_size)
    train_data=np.array(train_data[sel].astype('float'),float)
    test_data=np.array(test_data,float)
    labels=train_labels[sel]
    result=np.apply_along_axis(calc_nearest,1,test_data,train_data,labels)
    return result

#Funciton to calculate the Error Rate.
def calc_error_rate(result,testlabels):
    diff=result-testlabels
    errors=np.count_nonzero(diff)
    error_rate=(errors/len(testlabels))*100
    return error_rate

#Master Function  which computes errors for all sample sizes spread over a number of trials. The trial sizze has been set to ten.
def compute_all_errors(sample_sizes,test_data,train_data,train_labels,test_labels):
    
    print("Nearest Neighbours: ")
    trials=10
    errors=[]
    for s in sample_sizes:
        print("Sample: %s"%s)
        for i in range(trials):
            time_start=time.time()
            print("Iteration: %s"%(i+1))
            res=nearest_neighbours(s,test_data,train_data,train_labels)
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



