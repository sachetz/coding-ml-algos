import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


class Node(object):
    def __init__(self):
        self.name = None
        self.node_type = None
        self.predicted_class = None
        self.X = None
        self.test_attribute = None
        self.test_value = None
        self.children = []

    def __repr__(self):
        if self.node_type != 'leaf':
            s = (f"{self.name} Internal node with {self.X.shape[0]} examples, "
                 f"tests attribute {self.test_attribute} at {self.test_value}")           
        else:
            s = (f"{self.name} Leaf with {self.X.shape[0]} examples, predicts"
                 f" {self.predicted_class}")
        return s


class DecisionTree(object):

    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        '''
        Fit a tree on data, in which X is a 2-d numpy array
        of inputs, and y is a 1-d numpy array of outputs.
        '''
        self.root = self.recursive_build_tree(X, y, curr_depth=0, name='0')

    def recursive_build_tree(self, X, y, curr_depth, name):
        node = Node()
        node.name = name
        node.X = X

        # If the tree exceeds max depth
        if curr_depth == self.max_depth:
            node.node_type = 'leaf'
            node.predicted_class = scipy.stats.mode(y).mode
            return node

        # If all examples have the same classification
        classes = np.unique(y)
        if classes.size == 1:
            node.node_type = 'leaf'
            node.predicted_class = classes[0]
            return node

        # If there is only one unique example
        unique_examples = np.unique(X, axis=0)
        if unique_examples.shape[0] == 1:
            node.node_type = 'leaf'
            node.predicted_class = scipy.stats.mode(y).mode
            return node

        A, t = self.choose_split_attribute(X, y)
        node.test_attribute = A
        node.test_value = t
        node.node_type = 'internal'

        left_mask = X[:,A] <= t
        right_mask = X[:,A] > t
        node.children.append(
            self.recursive_build_tree(X[left_mask], y[left_mask], curr_depth=curr_depth+1, name=name + ".0")
        )
        node.children.append(
            self.recursive_build_tree(X[right_mask], y[right_mask], curr_depth=curr_depth+1, name=name + ".1")
        )

        return node

    def choose_split_attribute(self, X, y):
        max_gain = 0
        A = None
        t = 0

        for a in range(X.shape[1]):
            best_gain, best_threshold = self.importance(a, X, y)
            if best_gain > max_gain:
                max_gain = best_gain
                A = a
                t = best_threshold

        return A, t

    def importance(self, a, X, y):
        unique_values = np.unique(X[:,a])
        entropy = self.entropy(y)
        if unique_values.size > 2:                                              # For continuous attributes
            best_gain = 0
            best_threshold = 0
            sort_indices = np.argsort(X[:,a])
            sorted_X = X[sort_indices]
            sorted_Y = y[sort_indices]
            for i in range(sorted_X.shape[0]-1):
                if sorted_Y[i] != sorted_Y[i+1]:
                    threshold = (sorted_X[i,a] + sorted_X[i+1,a]) / 2
                    left_mask = X[:,a] <= threshold
                    right_mask = X[:,a] > threshold
                    left_entropy = self.entropy(y[left_mask])
                    right_entropy = self.entropy(y[right_mask])
                    gain = entropy - ((y[left_mask].size / y.size) * left_entropy + (y[right_mask].size / y.size) * right_entropy)
                    if gain > best_gain:
                        best_gain = gain
                        best_threshold = threshold
            return best_gain, best_threshold
        elif unique_values.size == 2:                                           # For discrete attributes
            threshold = (unique_values[0] + unique_values[1]) / 2
            left_mask = X[:,a] <= threshold
            right_mask = X[:,a] > threshold
            left_entropy = self.entropy(y[left_mask])
            right_entropy = self.entropy(y[right_mask])
            gain = entropy - ((y[left_mask].size / y.size) * left_entropy + (y[right_mask].size / y.size) * right_entropy)
            return gain, threshold
        else:
            return 0, 0

    def predict(self, testset):
        y = []
        for data in testset:
            node = self.root
            while node.node_type != 'leaf':
                if data[node.test_attribute] <= node.test_value:
                    node = node.children[0]
                else:
                    node = node.children[1]
            y.append(node.predicted_class)
        return np.array(y)

    def print(self):
        self.recursive_print(self.root)
    
    def recursive_print(self, node):
        print(node)
        for u in node.children:
            self.recursive_print(u)

    def entropy(self, y):
        'Return the information entropy in 1-d array y'
        _, counts = np.unique(y, return_counts = True)
        probs = counts/counts.sum()
        return -(np.log2(probs) * probs).sum()


def validation_curve(df):
    X_temp = df.loc[:, :278].values
    y_temp = df.loc[:, 279].values

    shuffled_indices = np.random.permutation(X_temp.shape[0])
    X = X_temp[shuffled_indices]
    y = y_temp[shuffled_indices]

    batch_size = X.shape[0] // 3
    X_partitions = [X[:batch_size], X[batch_size:2*batch_size], X[2*batch_size:]]
    Y_partitions = [y[:batch_size], y[batch_size:2*batch_size], y[2*batch_size:]]

    training_accuracy = []
    test_accuracy = []

    for d in range(2, 17, 2):
        training_acc = []
        test_acc = []

        for i in range(3):
            X_train = np.concatenate([X_partitions[j] for j in range(3) if j != i])
            y_train = np.concatenate([Y_partitions[j] for j in range(3) if j != i])
            X_test = X_partitions[i]
            y_test = Y_partitions[i]
            
            tree = DecisionTree(d)
            tree.fit(X_train, y_train)
            
            train_pred = tree.predict(X_train)
            test_pred = tree.predict(X_test)
            train = np.mean(train_pred == y_train)
            test = np.mean(test_pred == y_test)
            training_acc.append(train)
            test_acc.append(test)
        
        training_accuracy.append(np.mean(training_acc))
        test_accuracy.append(np.mean(test_acc))

    plt.figure()
    plt.plot(range(2, 17, 2), training_accuracy, label='train')
    plt.plot(range(2, 17, 2), test_accuracy, label='test')
    plt.title('Validation Curve')
    plt.xlabel('Tree Depth')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(False)

    plt.savefig('validation.pdf')
    plt.close()


# Read dataset from csv
df = pd.read_csv('./arrhythmia.csv', header=None, na_values="?")

# Replace missing values with the mode
for i in range(280):
    if df[i].isnull().sum() > 0:
        df.iloc[:,i].fillna(df[i].mode()[0], inplace=True)

# Plot the validation curve
validation_curve(df)