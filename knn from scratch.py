#Importing the required modules
import numpy as np
from scipy.stats import mode

#Euclidean Distance
def eucledian(p1,p2):
    dist = np.sqrt(np.sum((p1-p2)**2))
    return dist

#Function to calculate KNN
def predict(x_train, y , x_input, k):
    op_labels = []
    
    #Loop through the Datapoints to be classified
    for item in x_input: 
        
        #Array to store distances
        point_dist = []
        
        #Loop through each training Data
        for j in range(len(x_train)): 
            distances = eucledian(np.array(x_train[j,:]) , item) 
            #Calculating the distance
            point_dist.append(distances) 
        point_dist = np.array(point_dist) 
        
        #Sorting the array while preserving the index
        #Keeping the first K datapoints
        dist = np.argsort(point_dist)[:k] 
        
        #Labels of the K datapoints from above
        labels = y[dist]
        
        #Majority voting
        lab = mode(labels) 
        lab = lab.mode[0]
        op_labels.append(lab)

    return op_labels

#Importing the required modules
#Importing required modules
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from numpy.random import randint

#Loading the Data
iris= load_iris()

# Store features matrix in X
X= iris.data
#Store target vector in 
y= iris.target


#Creating the training Data
train_idx = xxx = randint(0,150,100)
X_train = X[train_idx]
y_train = y[train_idx]

#Creating the testing Data
test_idx = xxx = randint(0,150,50) #taking 50 random samples
X_test = X[test_idx]
y_test = y[test_idx]

#Applying our function 
y_pred = predict(X_train,y_train,X_test , 7)

#Checking the accuracy
accuracy_score(y_test, y_pred)
