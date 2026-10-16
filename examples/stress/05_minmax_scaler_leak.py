"""Stress case 05: a REAL fit-before-split leak that barely matters.

MinMaxScaler is fitted on all rows before splitting. That is technically leakage, but with a
strong signal and few features it should not move the score. Expected REPROVE verdict:
NOT_CONFIRMED (pattern present, no measurable inflation). Variable names are deliberately
different from the notebooks (data / labels / Xtr / Xte ...).
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

np.random.seed(0)
data = np.random.randn(300, 20)
labels = (data[:, 0] + 0.5 * data[:, 1] > 0).astype(int)

mm = MinMaxScaler()
data_norm = mm.fit_transform(data)          # leak: fitted on train + test rows

Xtr, Xte, ytr, yte = train_test_split(data_norm, labels, test_size=0.2, random_state=1)
model = LogisticRegression().fit(Xtr, ytr)
print("accuracy", accuracy_score(yte, model.predict(Xte)))   # metric is not stored in a variable on purpose
