from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from KNN import KNN
import numpy as np



data=load_iris()
X=data.data
y=data.target


X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=1234)
plt.figure()
plt.scatter(X[:,2],X[:,3],c=y,edgecolor='k',s=20)
plt.show()

clf=KNN(k=5)
clf.fit(X_train,y_train)
prediction=clf.predict(X_test)

print(prediction)
accuracy=np.sum(prediction==y_test)/len(y_test)
print(accuracy)