# This script shows one of the circumstances that KNN fail to predict.
# This case is called the grid of alternating dots.
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


# Create grid data
def get_data():
    width = 8
    height = 8
    N = width * height
    X = np.zeros((N, 2))
    Y = np.zeros(N)
    n = 0
    start_t = 0
    for i in range(width):
        t = start_t
        for j in range(height):
            X[n] = [i, j]
            Y[n] = t
            n += 1
            t = (t + 1) % 2 # alternate between 0 and 1
        start_t = (start_t + 1) % 2
    return X, Y


if __name__ == '__main__':
    X, Y = get_data()

    # display the data
    plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)  # s means size of the dot; c means colors
    plt.show()

    # get the accuracy
    model = KNN(3)
    model.fit(X, Y)
    print("Train accuracy:", model.score(X, Y))


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
