import numpy as np

# Step 1: Data Preparation
X_seen = np.load('X_seen.npy')  # 40 feature matrices for seen classes
X_test = np.load('Xtest.npy')   # Feature matrix of test data
Y_test = np.load('Ytest.npy')   # Ground truth labels of test data

class_attributes_seen = np.load('class_attributes_seen.npy')  # Class attributes for seen classes
class_attributes_unseen = np.load('class_attributes_unseen.npy')  # Class attributes for unseen classes

# Step 2: Compute Similarities
similarities = np.dot(class_attributes_unseen, class_attributes_seen.T)

# Step 3: Normalize Similarities
normalized_similarities = similarities / np.sum(similarities, axis=1, keepdims=True)

# Step 4: Compute Unseen Class Means
unseen_class_means = np.dot(normalized_similarities, X_seen.reshape(X_seen.shape[0], -1))

# Step 5: Prototype-Based Classification (You may use k-NN or another method)
def classify_using_nearest_neighbors(test_data, train_data, train_labels, k=1):
    # Implement a prototype-based classification algorithm using computed means
    # Here, we use k-NN as an example
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_data, train_labels)
    predicted_labels = knn.predict(test_data)
    return predicted_labels

# Use the prototype-based classification function to predict labels for test data
predicted_labels = classify_using_nearest_neighbors(X_test, X_seen.reshape(X_seen.shape[0], -1), Y_test)

# Step 6: Evaluate Accuracy
accuracy = (predicted_labels == Y_test.squeeze()).mean()
print(f"Accuracy: {accuracy * 100:.2f}%")
