#calculating all the coefficients
# make a dataframe first
X = df1.iloc[:,:7]
Y = df1.iloc[:,7]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size = 0.6,random_state=42)
(n,p_minus_one) = X_train.shape
p = p_minus_one + 1
new_X = np.ones(shape=(n,p))
new_X[:,1:] = X_train
X_T = new_X.T
weights = np.dot(np.dot(np.linalg.inv(np.dot(X_T,new_X)),X_T),Y_train)

# do this on collab
weights

#predicting the values

(a,b_minus_one) = X_test.shape
b = b_minus_one + 1
new_X_test = np.ones(shape=(a,b))
new_X_test[:,1:] = X_test
predictions = np.dot(new_X_test,weights)


#calculating the r2 value so that we get to know that how our machine is fitting to data

ssr = ((Y_test-predictions)**2).sum()
sst = ((Y_test-Y_test.mean())**2).sum()
rmse = np.sqrt(ssr/len(Y_test))
r3 = 1-(ssr/sst)
print(rmse)
print(r3)
