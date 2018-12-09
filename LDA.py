import numpy
import csv


# Split classes into two datasets and remove class labels
def split(dataset):
    good, bad = [], []
    for row in dataset:
        if row[-1] == 0:
            good.append(row[:-1])
        else:
            bad.append(row[:-1])
    return good, bad

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
    mu = mean(dataset)
    centered = [[0 for i in range(len(dataset[0]))] for j in range(len(dataset))]
    for i in range(len(dataset)):
        for j in range(len(dataset[j])):
            centered[i][j] = dataset[i][j] - mu[j]
    return centered

#Calculate Within-Class Scatter matrix
def withinScatter(dataset):
    good, bad = split(dataset)
    goodCentered = numpy.asarray(center(good))
    badCentered = numpy.asarray(center(bad))
    good_Sw = numpy.dot(goodCentered.T, goodCentered)
    bad_Sw = numpy.dot(badCentered.T, badCentered)
    Sw = good_Sw + bad_Sw
    return Sw

#Calculate Between-Class Scatter Matrix
def betweenScatter(dataset):
    good, bad = split(dataset)
    good_mean, bad_mean = mean(good), mean(bad)
    diff_mean = [good_mean[i] - bad_mean[i] for i in range(len(good_mean))]
    diff_mean_array = numpy.asarray(diff_mean)
    Sb = numpy.dot(diff_mean_array.T, diff_mean_array)
    return Sb

#Calculate eigenvalues and eigenvectors
def getEigens(dataset):
    Sw = withinScatter(dataset)
    Sb = betweenScatter(dataset)
    eigenvalues, eigenvectors = numpy.linalg.eig(numpy.linalg.inv(Sw).dot(Sb))
    return eigenvalues, eigenvectors

#Function to find the smallest value of r that satisfies a given percentage of retained variance
def chooseR(eigenvalues, alpha):
    total_variance = sum(eigenvalues)
    var = 0
    i = 0
    while(var < alpha*total_variance):
        var += eigenvalues[i]
        i += 1
    return i

#Get the dominant eigenvector
def getW(dataset):
    eigenvalues, eigenvectors = getEigens(dataset)
    r = chooseR(eigenvalues, 0.9)
    w = [eigenvectors[:,i].tolist() for i in range(r)]
    return numpy.asarray(w).T

#Project data onto dominant eigenvector
def LDA_proj(train_data, test_data):
    w = getW(train_data)
    #Project train data onto w
    x = [train_data[i][:-1] for i in range(len(train_data))]
    train_proj = numpy.dot(numpy.asarray(x), w)
    good, bad = [], []
    for i in range(len(train_data)):
        if(train_data[i][-1] == 1):
            good.append(train_proj[i])
        else:
            bad.append(train_proj[i])
    threshold = mean([mean(good),mean(bad)])
    #Project test data onto w
    x = [test_data[i][:-1] for i in range(len(test_data))]
    lda = numpy.dot(numpy.asarray(x), w)
    return lda, threshold

#Compare two vectors by summing the differences between their corresponding values
def compare_vectors(vec1, vec2):
    diff = 0
    for i in range(len(vec1)):
        diff += (vec1[i] - vec2[i])
    if(diff < 0):
        return -1
    elif(diff == 0):
        return 0
    else:
        return 1

#Separate projected data into classes based on mean
def LDA_classify(train_data, test_data):
    proj_data, threshold = LDA_proj(train_data, test_data)
    count = 0
    for i, row in enumerate(test_data):
        if(compare_vectors(proj_data[i],threshold) <= 0):
            row.append(0)
        else:
            row.append(1)
        #Check to see if our prediction was correct    
        if(row[-1] == row[-2]):
            count += 1.0
    print "Data classified with accuracy of:", count/len(test_data)

#Build dataset out of file
def load(filename):
    train = []
    test = []
    with open(filename, 'rb') as data_file:
        data = csv.reader(data_file)
        for i, row in enumerate(data):
            if(i < 800):
                train.append([float(x) for x in row])
            else:
                test.append([float(x) for x in row])
    return train, test

train, test = load("data.txt")
LDA_proj(train, test)
LDA_classify(train, test)
