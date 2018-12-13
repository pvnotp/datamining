import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.svm import SVC  


#Import dataset
dataset = pd.read_csv('reviews.csv')

#Separate the data (Data,classes)
Data = dataset.drop('class', axis=1)  
classes = dataset['class']

#Split the data and do resampling
Data_train, Data_test, classes_train, classes_test = train_test_split(Data,classes, test_size = 0.20)  

#Apply SVC classifier using a linear kernel
classifier = SVC(kernel='linear')  

#Fit the data using train data
classifier.fit(Data_train, classes_train)

#Predict class labels of test 
classes_pred = classifier.predict(Data_test) 


classes_pred = classes_pred.tolist()
classes_test = classes_test.tolist()
classes_size = len(classes_test)


count = 0
for i in range(0, classes_size):
    if(classes_pred[i] == classes_test[i]):
        count += 1

precision = float(count)/classes_size

print "Accuracy: %s" % precision
print classes_pred
