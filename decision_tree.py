import csv
import sys

# Calculate probability of a point being in a given class
def P(c, dataset):
    count = 0.0
    if len(dataset) == 0:
        return 0
    else:
        for row in dataset:
            if(row[-1]==c):
                count += 1
    return count/len(dataset)

# Split a dataset based on a criterion for a specified attribute
def split(dataset, index, value):
    left, right = [], []
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

# Helper function for GINI index
def G(dataset):
    return 1 - (P(0, dataset)*P(0, dataset) + P(1, dataset)*P(1, dataset))

# Calculate the Gini index for a split on the specified criterion
def GINI(dataset, index, value):
    left, right = split(dataset, index, value)
    nL = len(left)
    nR = len(right)
    n = nL + nR
    return nL/float(n)*G(left) + nR/float(n)*G(right)

# Reverse sign of Gini index so we can maximize it
def negGINI(dataset, index, value):
    return -GINI(dataset, index, value)


#Find the criterion that gives the best split on the dataset
def bestSplit(dataset, criterion):
    print "Finding best split...please wait"
    if criterion == GINI:
        criterion = negGINI
    best = criterion(dataset, 0, 0)
    best_index, best_value = 0, 0
    for i in range(len(dataset[0])-1):
        sys.stdout.write(".")
        attr_values = [row[i] for row in dataset]
        for val in attr_values:
            if criterion(dataset, i, val) > best:
                best = criterion(dataset, i, val)
                best_index, best_value = i, val
    print
    return {'index': best_index, 'value': best_value,
            'groups':split(dataset, best_index, best_value)}

# Get the most common rating in a group of rows
# This rating will be assigned to the whole group by the decision tree
def rate(group):
    ratings = [row[-1] for row in group]
    #Thanks to newacct at StackOverflow for this oneliner
    #https://stackoverflow.com/questions/1518522/find-the-most-common-element-in-a-list
    return max(set(ratings), key=ratings.count)

# Create child splits for a node or make terminal
#Thanks to Jason Brownlee, whose tutorial was the foundation for this code
#machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/.
def splitNode(node, max_depth, min_size, depth, criterion, words):
    print "Splitting a node: %s %s < %f" % (depth*'-', words[node['index']+1], node['value'])
    left, right = node['groups']
    del(node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = rate(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = rate(left), rate(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = rate(left)
    else:
        node['left'] = bestSplit(left, criterion)
        splitNode(node['left'], max_depth, min_size, depth+1, criterion, words)
    # process right child
    if len(right) <= min_size:
        node['right'] = rate(right)
    else:
        node['right'] = bestSplit(right, criterion)
        splitNode(node['right'], max_depth, min_size, depth+1, criterion, words)

# Build a decision tree
# Two arguments specify when the tree should stop splitting:
#   max_depth sets the maximum depth of the tree
#   min_size sets the smallest acceptable size for a subset of the dataset that can be split
def buildTree(train, max_depth, min_size, criterion):
    root = bestSplit(train, criterion)
    splitNode(root, max_depth, min_size, 1, criterion, words)
    return root


# Make a prediction using our tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

#Predict class of each row of test data and gauge accuracy
def classifyData(dataset, tree):
    count = 0
    for row in dataset:
        row.append(predict(tree, row))
        #Check if correction was correct
        if(row[-1] == row[-2]):
            count += 1.0
    print "Data classified with accuracy of:", count/len(dataset)
    


#Build dataset out of file
def load(filename):
    train = []
    test = []
    words = []
    with open(filename, 'rb') as data_file:
        data = csv.reader(data_file)
        for i, row in enumerate(data):
            if(i == 0):
                words = row
            elif(i > 0 and i < 800):
                train.append([float(x) for x in row])
            else:
                test.append([float(x) for x in row])
    return words, train, test

       
words, train, test = load("reviews.csv")
tree = buildTree(train, 3, 100, GINI)
classifyData(test, tree)

