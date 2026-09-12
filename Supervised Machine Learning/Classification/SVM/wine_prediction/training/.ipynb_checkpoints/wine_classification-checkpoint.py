import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

highest_acc = 0

for kernel in ['linear', 'poly', 'rbf']:
    for C in [0.1, 1, 10, 100]:
        model = SVC(kernel=kernel, C=C)
        model.fit(X_train_scaled, y_train)
        print(kernel, C, model.score(X_test_scaled, y_test))
        if model.score(X_test_scaled, y_test) >= highest_acc:
            sel_model = model

joblib.dump(scaler, 'wine_scaler.joblib')
joblib.dump(sel_model, 'wine_model.joblib')