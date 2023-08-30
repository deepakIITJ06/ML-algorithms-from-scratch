#calculating all the coefficients

X = df1.iloc[:,:7]
Y = df1.iloc[:,7]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size = 0.6,random_state=42)
(n,p_minus_one) = X_train.shape
p = p_minus_one + 1
new_X = np.ones(shape=(n,p))
new_X[:,1:] = X_train
X_T = new_X.T
weights = np.dot(np.dot(np.linalg.inv(np.dot(X_T,new_X)),X_T),Y_train)


