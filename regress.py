import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

# Load the data set
X_seen=np.load('X_seen.npy', allow_pickle=True, encoding='latin1') # (40 x N_i x D): 40 feature matrices. X_seen[i] is the N_i x D feature matrix of seen class i
Xtest=np.load('Xtest.npy', allow_pickle=True, encoding='latin1')	# (6180, 4096): feature matrix of the test data.
Ytest=np.load('Ytest.npy', allow_pickle=True, encoding='latin1')	# (6180, 1): ground truth labels of the test data
class_attributes_seen=np.load('class_attributes_seen.npy', allow_pickle=True, encoding='latin1')	# (40, 85): 40x85 matrix with each row being the 85-dimensional class attribute vector of a seen class.
class_attributes_unseen=np.load('class_attributes_unseen.npy', allow_pickle=True, encoding='latin1')	# (10, 85): 10x85 matrix with each row being the 85-dimensional class attribute vector of an  unseen class.


def meanSeenClass(xs):
    # This function calculates mean of seen classes
    mean = np.zeros((xs.shape[0], xs[0].shape[1]))
    for i in range(0, xs.shape[0]):
        mean[i] = (np.mean(xs[i], axis=0)).reshape(1, xs[0].shape[1])
    return mean

def meanUnseenClass(u_seen, Aus, As, k):
    # This function calculates the mean of unseen classes from the learned W as per Method2.
    W1 = np.dot(As.T, As) + k*(np.eye(As.shape[1]))
    W2 = np.dot(As.T, u_seen)
    W = np.dot(np.linalg.inv(W1), W2)
    mean = np.dot(Aus, W)
    return mean

def predict(u, x_test, y_test, theta):
    acc = 0.
    dist = np.zeros((y_test.shape[0], u.shape[0]))
    for i in range(u.shape[0]):
        diff = u[i] - x_test
        sq = np.square(diff)
        d = np.dot(sq, theta)
        dist[:, i] = d.reshape(d.shape[0],)

    y_pred = np.argmin(dist, axis=1)
    y_pred = y_pred.reshape(y_pred.shape[0],1)
    y_pred+=1
    acc = 1 - np.count_nonzero(y_pred-y_test)/float(y_test.shape[0])
    return y_pred, acc

def classifier():
    uSeen = meanSeenClass(X_seen)
    theta = np.ones((uSeen.shape[1], 1))
    # Test Class
    u_seen = uSeen
    attributes_seen = class_attributes_seen
    attributes_unseen = class_attributes_unseen
    x_test = Xtest
    y_test = Ytest

    for i in [0.01, 0.1, 1, 10, 20, 50, 100]:
        u_unseen = meanUnseenClass(u_seen, attributes_unseen, attributes_seen, i)
        y_pred, acc = predict(u_unseen, x_test, y_test, theta)

        print("Test accuracy for lamba = " + str(i) + " is: " + str(100*acc))

classifier()




















# import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.neighbors import KNeighborsClassifier

# # Step 1: Data Preparation
# X_seen = np.load('X_seen.npy', allow_pickle=True, encoding='latin1')  # 40 feature matrices for seen classes
# X_test = np.load('Xtest.npy', allow_pickle=True, encoding='latin1')   # Feature matrix of test data
# Y_test = np.load('Ytest.npy', allow_pickle=True, encoding='latin1')   # Ground truth labels of test data

# class_attributes_seen = np.load('class_attributes_seen.npy', allow_pickle=True, encoding='latin1')  # Class attributes for seen classes
# class_attributes_unseen = np.load('class_attributes_unseen.npy', allow_pickle=True, encoding='latin1')  # Class attributes for unseen classes

# # Step 2: Train Linear Model
# # Train a linear regression model to predict the means of the unseen classes
# model = LinearRegression()
# model.fit(class_attributes_seen, X_seen.reshape(X_seen.shape[0], -1))



# # Step 3: Compute Unseen Class Means
# unseen_class_means = model.predict(class_attributes_unseen)

# # Step 4: Prototype-Based Classification (You may use k-NN or another method)
# def classify_using_nearest_neighbors(test_data, train_data, train_labels, k=1):
#     # Implement a prototype-based classification algorithm using computed means
#     # Here, we use k-NN as an example
#     knn = KNeighborsClassifier(n_neighbors=k)
#     knn.fit(train_data, train_labels)
#     predicted_labels = knn.predict(test_data)
#     return predicted_labels

# # Use the prototype-based classification function to predict labels for test data
# predicted_labels = classify_using_nearest_neighbors(X_test, X_seen.reshape(X_seen.shape[0], -1), Y_test)

# # Step 5: Evaluate Accuracy
# accuracy = (predicted_labels == Y_test.squeeze()).mean()
# print(f"Accuracy: {accuracy * 100:.2f}%")
