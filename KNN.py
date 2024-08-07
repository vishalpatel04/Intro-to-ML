from collections import Counter
import numpy as np

def euclid_distance(x1,x2):
    distance= np.sqrt(np.sum(x1-x2)**2)
    return distance

class KNN:
    def __init__(self,k=5):
         self.k=k 
         
    def fit(self,X,y):
        self.X_train=X
        self.y_train=y
        
    def predict(self,X):
        prediction= [self._predict(x) for x in X]
        return prediction
    
    def _predict(self,x):
        #distances
        distances=[euclid_distance(x,x_train) for x_train in self.X_train]
        
        #nearest neighbors k
        n_indices= np.argsort(distances)[:self.k]
        n_labels=[self.y_train[i] for i in n_indices]
        #print(n_indices)
        #print(n_labels)
        
        #majoity vote taken
        most_common=Counter(n_labels).most_common()
        return most_common[0][0]