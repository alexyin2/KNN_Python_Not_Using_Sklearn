# Here we going to show one of the circumstances that KNN may prefer quite well.
import numpy as np
import matplotlib.pyplot as plt
from sortedcontainers import SortedList
from future.utils import iteritems


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


def get_xor():
    X = np.zeros((200, 2))
    X[:50] = np.random.random((50, 2)) / 2 + 0.5 # (0.5-1, 0.5-1)
    X[50:100] = np.random.random((50, 2)) / 2 # (0-0.5, 0-0.5)
    X[100:150] = np.random.random((50, 2)) / 2 + np.array([[0, 0.5]]) # (0-0.5, 0.5-1)
    X[150:] = np.random.random((50, 2)) / 2 + np.array([[0.5, 0]]) # (0.5-1, 0-0.5)
    Y = np.array([0]*100 + [1]*100)
    return X, Y


if __name__ == '__main__':
    X, Y = get_xor()

    # display the data
    plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
    plt.show()

    # get the accuracy
    model = KNN(3)
    model.fit(X, Y)
    print("Accuracy:", model.score(X, Y))