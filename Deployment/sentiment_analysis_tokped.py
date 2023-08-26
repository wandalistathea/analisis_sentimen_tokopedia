# IMPORT LIBRARY ----------
import pandas as pd
import numpy as np
import pickle
import sklearn


# IMPORT DATA ----------
# Data sudah hasil preprocessing (sudah hasil stemming)
tokped_sudah_stemming = pd.read_csv("Data Oke Setelah Di-Stemming (Masing2 3k).csv")
content_isi = tokped_sudah_stemming[~tokped_sudah_stemming["content_clean"].isnull()]

# Hilangkan kata "tokopedia" dan "aplikasi"
stopwords_tokped = ["tokopedia", "aplikasi"]

def clean_tokped(text):
    temp = text.split() # split words
    temp = [w for w in temp if not w in stopwords_tokped] # remove stopwords
    temp = " ".join(word for word in temp) # join all words
    return temp

content_isi["content_clean_tanpax"] = content_isi["content_clean"].apply(clean_tokped)


# TF-IDF ----------
X = content_isi["content_clean_tanpax"]
y = content_isi["sentimen"]

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Simpan vectorizer dalam bentuk pickle
with open("vectorizer_tfidf.pkl", "wb") as f:
    pickle.dump(vectorizer, f)


# MODELING
# Membuat model menggunakan SVC (SVM Classification)
from sklearn.svm import SVC
svc = SVC(kernel = "linear")
svc.fit(X, y)

# Pakai cross validation
from sklearn.model_selection import cross_val_score
accuracies_svc = cross_val_score(estimator = svc, X = X, y = y, cv = 10)
print("Nilai Akurasi SVM 10-Fold: {:.2f}%".format(accuracies_svc.mean() * 100))

# Simpan model dalam bentuk pickle
with open("model_svc.pkl", "wb") as f:
    pickle.dump(svc, f)