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

# u = np.zeros()
def meanSeenClass(xs):
    # This function calculates mean of seen classes
    mean = np.zeros((xs.shape[0], xs[0].shape[1]))
    for i in range(0, xs.shape[0]):
        mean[i] = (np.mean(xs[i], axis=0)).reshape(1, xs[0].shape[1])
    return mean

def calcSimilarity(attributes_unseen, attributes_seen):
    # Calculates the similarity
    s = np.dot(attributes_unseen, attributes_seen.T)
    s = s/(np.sum(s, axis=1)).reshape(s.shape[0],1)
    return s

def meanUnseenClass(s, u_seen):
    # This function calculates the mean of unseen classes as per Method1.
    mean = np.dot(s, u_seen)
    print (mean)
    return mean

def predict(u, x_test, y_test, theta):
    # This function predicts the classes based on given mean data implementation was done using theta as per mahalanobis distance
    # Returns the values of prediction and accuracy

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

def train(u, x_seen, theta):
    # This function is to learn theta
    # update rule was used as follow:
    # if diff b/w any feature of mean and x is large on seen class then
    # the theta for that will be removed more as compared to other
    # since theta acts as a weights so next time weight for that specific feture will be reduced
    # Conditioned at theta magnitude to be 1 so that it doesn't become to low
    # NOTE: Not much improvement was observed!
    diff = u - x_seen
    sq = np.sum(np.square(diff), axis=0).reshape(diff.shape[1], 1)
    d = theta*sq
    theta = theta - 0.1*d
    theta/=np.sum(theta)

    return theta

def classifier():
    uSeen = meanSeenClass(X_seen)

    theta = np.ones((uSeen.shape[1], 1))
    theta/=np.sum(theta)

    # Uncomment this to learn theta k equlas no. of time you want to update the thetaself.
    # k = 30
    # for j in range(k):
    #     for i in range(0, 30):
    #         theta = train(uSeen[i], X_seen[i], theta)

    # Test Class
    u_seen = uSeen
    attributes_seen = class_attributes_seen
    attributes_unseen = class_attributes_unseen
    x_test = Xtest
    y_test = Ytest

    s = calcSimilarity(attributes_unseen, attributes_seen)
    u_unseen = meanUnseenClass(s, u_seen)
    y_pred, acc = predict(u_unseen, x_test, y_test, theta)

    print(100*acc)

# Call the classifier
classifier()















# import numpy as np

# # Step 1: Data Preparation
# X_seen = np.load('X_seen.npy', allow_pickle=True, encoding='latin1')  # 40 feature matrices for seen classes
# X_test = np.load('Xtest.npy', allow_pickle=True, encoding='latin1')   # Feature matrix of test data
# Y_test = np.load('Ytest.npy', allow_pickle=True, encoding='latin1')   # Ground truth labels of test data

# class_attributes_seen = np.load('class_attributes_seen.npy', allow_pickle=True, encoding='latin1')  # Class attributes for seen classes
# class_attributes_unseen = np.load('class_attributes_unseen.npy', allow_pickle=True, encoding='latin1')  # Class attributes for unseen classes

# # Step 2: Compute Similarities
# similarities = np.dot(class_attributes_unseen, class_attributes_seen.T)

# # Step 3: Normalize Similarities
# normalized_similarities = similarities / np.sum(similarities, axis=1, keepdims=True)

# # Step 4: Compute Unseen Class Means
# # Reshape X_seen to match the number of features (4096) for compatibility with dot product
# X_seen_reshaped = X_seen.reshape(X_seen.shape[0], -1)
# unseen_class_means = np.dot(normalized_similarities, X_seen_reshaped.T)

# # Step 5: Prototype-Based Classification (You may use k-NN or another method)
# def classify_using_nearest_neighbors(test_data, train_data, train_labels, k=1):
#     # Implement a prototype-based classification algorithm using computed means
#     # Here, we use k-NN as an example
#     from sklearn.neighbors import KNeighborsClassifier
#     knn = KNeighborsClassifier(n_neighbors=k)
#     knn.fit(train_data.T, train_labels)  # Transpose train_data here
#     predicted_labels = knn.predict(test_data)
#     return predicted_labels

# # Use the prototype-based classification function to predict labels for test data
# predicted_labels = classify_using_nearest_neighbors(X_test, unseen_class_means, Y_test)

# # Step 6: Evaluate Accuracy
# accuracy = (predicted_labels == Y_test.squeeze()).mean()
# print(f"Accuracy: {accuracy * 100:.2f}%")
