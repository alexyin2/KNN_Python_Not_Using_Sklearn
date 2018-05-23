# This is an example of a K-Nearest Neighbors classifier on MNIST data.
# We try k=1...5 to show how we might choose the best k.
from __future__ import print_function, division
from future.utils import iteritems
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sortedcontainers import SortedList
from datetime import datetime

# Note: You can't use SortedDict because the key is distance
# if 2 close points are the same distance away, one will be overwritten

# We'll first write a module to help us read in the file and do some data preprocessing
def get_data(limit=None):
    print("Reading in and transforming data...")
    df = pd.read_csv('/Users/alex/Desktop/DeepLearningPrerequisites/KNN/Datasets/train.csv')
    data = df.as_matrix()
    np.random.shuffle(data)  # Randomly reindex the rows and sort it
    X = data[:, 1:] / 255.0  # The rango of data is from 0 to 255
    Y = data[:, 0]
    if limit is not None:  # We can set a limit so running our algorithm won't take too long
        X, Y = X[:limit], Y[:limit]
    return X, Y

'''
def get_xor():
    X = np.zeros((200, 2))
    X[:50] = np.random.random((50, 2)) / 2 + 0.5 # (0.5-1, 0.5-1)
    X[50:100] = np.random.random((50, 2)) / 2 # (0-0.5, 0-0.5)
    X[100:150] = np.random.random((50, 2)) / 2 + np.array([[0, 0.5]]) # (0-0.5, 0.5-1)
    X[150:] = np.random.random((50, 2)) / 2 + np.array([[0.5, 0]]) # (0.5-1, 0-0.5)
    Y = np.array([0]*100 + [1]*100)
    return X, Y

def get_donut():
    N = 200
    R_inner = 5
    R_outer = 10

    # distance from origin is radius + random normal
    # angle theta is uniformly distributed between (0, 2pi)
    R1 = np.random.randn(N//2) + R_inner
    theta = 2*np.pi*np.random.random(N//2)
    X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

    R2 = np.random.randn(N//2) + R_outer
    theta = 2*np.pi*np.random.random(N//2)
    X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

    X = np.concatenate([ X_inner, X_outer ])
    Y = np.array([0]*(N//2) + [1]*(N//2))
    return X, Y
'''

# Now we define our KNN module
class KNN(object):
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        y = np.zeros(len(X))
        for i,x in enumerate(X): # test points; Using enumerate() will create an index stored in i
            sl = SortedList() # stores (distance, class) tuples; More detail explanations about SortedList() can be found in PS.
            for j,xt in enumerate(self.X): # training points
                diff = x - xt
                d = diff.dot(diff)  # Calculating Euclidean Distance without square root
                if len(sl) < self.k:
                    sl.add( (d, self.y[j]) )
                else:
                    if d < sl[-1][0]:
                        del sl[-1]
                        sl.add( (d, self.y[j]) )

            # Vote
            votes = {}
            for _, v in sl:
                votes[v] = votes.get(v,0) + 1  # More explanations about {}.get() can be found in PS.
            # print "votes:", votes, "true:", Ytest[i]
            max_votes = 0
            max_votes_class = -1
            for v,count in iteritems(votes):
                if count > max_votes:
                    max_votes = count
                    max_votes_class = v
            y[i] = max_votes_class
        return y

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)


if __name__ == '__main__':
    X, Y = get_data(2000)  # we only take 2000 rows
    Ntrain = 1000
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]
    train_scores = []
    test_scores = []
    ks = (1,2,3,4,5)
    for k in ks:
        print("\nk =", k)
        knn = KNN(k)
        t0 = datetime.now()
        knn.fit(Xtrain, Ytrain)
        print("Training time:", (datetime.now() - t0))

        t0 = datetime.now()
        train_score = knn.score(Xtrain, Ytrain)
        train_scores.append(train_score)
        print("Train accuracy:", train_score)
        print("Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(Ytrain))

        t0 = datetime.now()
        test_score = knn.score(Xtest, Ytest)
        print("Test accuracy:", test_score)
        test_scores.append(test_score)
        print("Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(Ytest))

    plt.plot(ks, train_scores, label='train scores')
    plt.plot(ks, test_scores, label='test scores')
    plt.legend()
    plt.show()
    

# PS.
# 1. SortedList()
test = SortedList()
test.add((150, 1))  # In this case, 150 means the distance and 1 means the number
test.add((30, 2))
test.add((125, 2))
test.add((90, 1))
test.add((40, 3))
print(test)  # We can find out it's sorted by the first value

# 2. {}.get(target, value)
# target is the key that we are looking at in the dictionary, and value is the number that will show up if we doesn't find the target
test1 = {'a': 20, 
         'b': 10,
         'c': 15}
test1.get('a', 0)
test1.get('z', 0)  # It returns 0 because we didn't find 'z' in the key
test1.get('z', None)  # Nothing returns



