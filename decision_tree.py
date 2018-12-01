# Split a dataset based on a criterion for a specified attribute
def split(dataset, index, value):
    left, right = [], []
    for row in dataset:
        if row[index] <= value:
            left.append(row)
        else:
            right.append(row)
    return left, right

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
    if criterion == GINI:
        criterion = negGINI
    best = criterion(dataset, 0, 0)
    best_index, best_value = 0, 0
    for i in range(len(dataset[0])):
        attr_values = [row[i] for row in dataset]
        for val in attr_values:
            if criterion(dataset, i, val) > best:
                best = criterion(dataset, i, val)
                best_index, best_value = i, val
    return {'index': best_index, 'value': best_value,
            'groups':split(dataset, best_index, best_value)}

# Get the most common rating in a group of rows
# This rating will be assigned to the whole group
def rate(group):
    ratings = [row[-1] for row in group]
    #Thanks to newacct at StackOverflow for this oneliner
    #https://stackoverflow.com/questions/1518522/find-the-most-common-element-in-a-list
    return max(set(ratings), key=ratings.count)

# Create child splits for a node or make terminal
def splitNode(node, max_depth, min_size, depth, criterion):
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
        splitNode(node['left'], max_depth, min_size, depth+1)
    # process right child
    if len(right) <= min_size:
        node['right'] = rate(right)
    else:
        node['right'] = bestSplit(right, criterion)
        splitNode(node['right'], max_depth, min_size, depth+1)

# Build a decision tree
# Two arguments specify when the tree should stop splitting:
#   max_depth sets the maximum depth of the tree
#   min_size sets the smallest acceptable size for a subset of the dataset that can be split
def build_tree(train, max_depth, min_size, criterion):
    root = bestSplit(train, criterion)
    splitNode(root, max_depth, min_size, 1, criterion)
    return root

# Print out the criterion used for each split in the tree
def print_tree(node, depth=0):
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
        print_tree(node['left'], depth+1)
        print_tree(node['right'], depth+1)
    else:
        print('%s[%s]' % ((depth*' ', node)))

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

#Get the ratio of correct predictions
def accuracy(dataset, tree):
    count = 0
    for row in dataset:
        prediction = predict(tree, row)
        if(row[-1] == prediction):
            count += 1.0
    return count/len(dataset)
          
dataset = [[2.771244718,1.784783929,0],
    [1.728571309,1.169761413,0],
    [3.678319846,2.81281357,0],
    [3.961043357,2.61995032,0],
    [2.999208922,2.209014212,0],
    [7.497545867,3.162953546,1],
    [9.00220326,3.339047188,1],
    [7.444542326,0.476683375,1],
    [10.12493903,3.234550982,1],
    [6.642287351,3.319983761,1]]

tree = build_tree(dataset, 1, 3, GINI)
print_tree(tree)
print accuracy(dataset, tree)
