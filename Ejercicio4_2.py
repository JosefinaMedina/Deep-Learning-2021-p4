import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics, model_selection
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

df_train = pd.read_csv('../04_Multiperceptron/Segment_Train.csv')
X_train = np.array(df_train.iloc[:,:-1]).astype(float)
Y_train = np.array(df_train.iloc[:,-1])

df_test = pd.read_csv('../04_Multiperceptron/Segment_Test.csv')
X_test = np.array(df_test.iloc[:,:-1]).astype(float)
Y_test = np.array(df_test.iloc[:,-1])

normalizarEntrada = 1  # 1 si normaliza; 0 si no

if normalizarEntrada:
    # Escala los valores entre 0 y 1
    min_max_scaler = preprocessing.StandardScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.transform(X_test)

clf = MLPClassifier(solver='sgd', learning_rate_init=0.05,
                    hidden_layer_sizes=(15,3), random_state=1,
                    max_iter=2000, 
                    verbose=50,  tol=1.0e-05,
                    activation='tanh')

clf.fit(X_train,Y_train)

# scores = cross_val_score(clf, X_train, Y_train, cv=10, scoring='accuracy')
# print("AVG accuracy %.4f" % np.mean(scores)) 
Y_pred = clf.predict(X_train)

print("%% aciertos X_train : %.3f" % metrics.accuracy_score(Y_train, Y_pred))
report = metrics.classification_report(Y_train,Y_pred)

print("Confusion matrix:\n%s" % report)
cm = metrics.confusion_matrix(Y_train,Y_pred)
print("Confusion matrix:\n%s" % cm) 

#fig=metrics.plot_confusion_matrix(clf, X_train, Y_train) 
#plt.title("Matriz de confusión - Datos TRAIN") 
#plt.show()
     
Y_predTest= clf.predict(X_test)
print("%% aciertos X_test : %.3f" % metrics.accuracy_score(Y_test,Y_predTest))

report = metrics.classification_report(Y_test,Y_predTest)
print("Confusion matrix:\n%s" % report)

MM = metrics.confusion_matrix(Y_test,Y_predTest)
print("Confusion matrix:\n%s" % MM) 

#fig=metrics.plot_confusion_matrix(clf, X_test, Y_test)  
#plt.title("Matriz de confusión - Datos TESTEO")
#plt.show()