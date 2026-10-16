"""Stress case 06: CLEAN code that uses pd.concat to join feature tables (axis=1).

Joining columns does not duplicate rows, so no leak exists. Expected REPROVE verdict: CLEAN.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

np.random.seed(0)
a = pd.DataFrame(np.random.randn(200, 5), columns=list("abcde"))
b = pd.DataFrame(np.random.randn(200, 3), columns=list("fgh"))
df = pd.concat([a, b], axis=1)              # column join, not row duplication
y = (df["a"] + df["f"] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state=0)
clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
acc = accuracy_score(y_test, clf.predict(X_test))
print(acc)
