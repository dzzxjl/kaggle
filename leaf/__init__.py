from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit([1, 2, 2, 6])
print(le)
print(le.classes_)
array = le.transform([1, 1, 2, 6])
print(array)