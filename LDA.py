import numpy

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

# Split dataset into two classes and remove class labels
def split(dataset):
    left, right = [], []
    for row in dataset:
        if row[-1] == 0:
            left.append(row[:-1])
        else:
            right.append(row[:-1])
    return left, right

#Get mean vector for dataset
def mean(dataset):
    mean = [0 for i in range(len(dataset[0]))]
    for row in dataset:
        for i in range(len(row)):
            mean[i] += row[i]
    for i in range(len(row)):
        mean[i] /= len(dataset)
    return mean

#Center dataset
def center(dataset):
    centered = [[0 for i in range(len(dataset[0]))] for j in range(len(dataset))]
    for i in range(len(dataset)):
        for j in range(len(dataset[j])):
            centered[i][j] = dataset[i][j] - mean(dataset)[j]
    return centered

#Calculate Within-Class Scatter matrix
def withinScatter(dataset):
    left, right = split(dataset)
    leftCentered = numpy.asarray(center(left))
    rightCentered = numpy.asarray(center(right))
    left_Sw = numpy.dot(leftCentered.T, leftCentered)
    right_Sw = numpy.dot(rightCentered.T, rightCentered)
    Sw = left_Sw + right_Sw
    return Sw

#Calculate Between-Class Scatter Matrix
def betweenScatter(dataset):
    left, right = split(dataset)
    left_mean, right_mean = mean(left), mean(right)
    diff_mean = [left_mean[i] - right_mean[i] for i in range(len(left_mean))]
    diff_mean_array = numpy.asarray(diff_mean)
    Sb = numpy.dot(diff_mean_array.T, diff_mean_array)
    return Sb

#Calculate eigenvalues and eigenvectors
def getEigens(dataset):
    Sw = withinScatter(dataset)
    Sb = betweenScatter(dataset)
    eigenvalues, eigenvectors = numpy.linalg.eig(numpy.linalg.inv(Sw).dot(Sb))
    return eigenvalues, eigenvectors

#Get the dominant eigenvector
def getW(dataset):
    eigenvalues, eigenvectors = getEigens(dataset)
    i = numpy.where(eigenvalues == max(eigenvalues))
    vector = [eigenvectors[i][j] for j in range(1)]
    return eigenvectors[0]

#Project data onto dominant eigenvector
def LDA_proj(train_data, test_data):
    w = getW(train_data)
    x = [test_data[i][:-1] for i in range(len(test_data))]
    lda = numpy.dot(numpy.asarray(x), w)
    return lda

#Separate projected data into classes based on mean
def LDA_classify(train_data, test_data):
    proj_data = LDA_proj(train_data, test_data)
    mu = numpy.mean(proj_data)
    for i, row in enumerate(test_data):
        if proj_data[i] <= mu:
            row.append(1)
        else:
            row.append(0)
    
LDA_classify(dataset, dataset)
print dataset
