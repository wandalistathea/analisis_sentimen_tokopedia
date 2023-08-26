from flask import Flask, render_template, request
import pickle
import pandas as pd
import string, re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import nltk
# nltk.download()
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# Load file pickle-nya
vectorizer = pickle.load(open("vectorizer_tfidf.pkl", "rb")) 
svc = pickle.load(open("model_svc.pkl", "rb")) 


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        content = request.form.get("input-data")
        df_content = pd.DataFrame([content], columns = ["content"])

        # Kumpulkan semua function yang telah dibuat ----------
        # Case Folding
        def clean_content(content):
            content = content.lower() # menjadikan lowercase
            content = re.sub("[^a-z]", " ", content) # hapus semua karakter kecuali a-z
            content = re.sub("\t", " ", content) # mengganti tab dengan spasi
            content = re.sub("\n", " ", content) # mengganti new line dengan spasi
            content = re.sub("\s+", " ", content) # mengganti spasi > 1 dengan 1 spasi
            content = content.strip() # menghapus spasi di awal dan akhir 
            return content


        # Koreksi Penulisan
        # PAKAI WORD BREAK (\b)
        def koreksi_penulisan(content):
            dict_koreksi = {}
            file = open("list koreksi penulisan (tambahan sendiri).txt")
            
            for x in file:
                f = x.split(":")
                dict_koreksi.update({f[0].strip(): f[1].strip()})
            
            for awal, pengganti in dict_koreksi.items():
                #content = str(content).replace(awal, pengganti)
                content = re.sub(r"\b" + awal + r"\b", pengganti, content)
            return content

            
        # Stopword Removal
        def clean_stopword(content):
            # Stopword Sastrawi
            factory = StopWordRemoverFactory()
            stopword_sastrawi = factory.get_stop_words()
            content = content.split() # split jadi kata per kata
            content = [w for w in content if w not in stopword_sastrawi] # hapus stopwords
            content = " ".join(w for w in content) # join semua kata yang bukan stopwords

            # Stopword NLTK
            stopword_nltk = set(stopwords.words("indonesian"))
            stopword_nltk = stopword_nltk
            content = content.split() # split jadi kata per kata
            content = [w for w in content if w not in stopword_nltk] # hapus stopwords
            content = " ".join(w for w in content) # join semua kata yang bukan stopwords
            return content


        # Stopword Tambahan
        def clean_stopword_tambahan(content):
            with open("list stopword baru (tambahan sendiri).txt", "r") as f: 
                stopwords_tambahan = f.read().splitlines()
            content = content.split() # split jadi kata per kata
            content = [w for w in content if w not in stopwords_tambahan] # hapus stopwords
            content = " ".join(w for w in content) # join semua kata yang bukan stopwords
            return content


        # Stemming
        def clean_stem(content):
            # Stemming Sastrawi
            factory = StemmerFactory()
            stemmer_sastrawi = factory.create_stemmer()
            content = stemmer_sastrawi.stem(content)
            return content


        # Clean Tokped
        stopwords_tokped = ["tokopedia", "aplikasi"]
        def clean_tokped(text):
            temp = text.split() # split words
            temp = [w for w in temp if not w in stopwords_tokped] # remove stopwords
            temp = " ".join(word for word in temp) # join all words
            return temp
        

        # Function gabungan semua proses ----------
        # Nanti untuk run data/content yang akan dites
        def prediksi_sentimen(df_content):
            # 1. membersihkan content
            df_content = df_content.apply(clean_content)
            
            # 2. koreksi penulisan
            df_content = df_content.apply(koreksi_penulisan)
            
            # 3. hapus stopwords
            df_content = df_content.apply(clean_stopword)
            df_content = df_content.apply(clean_stopword_tambahan)
            
            # 4. stemming
            df_content = df_content.apply(clean_stem)
            
            # 5. hapus stopword tokped
            df_content = df_content.apply(clean_tokped)
            
            # 6. vectorizer
            df_content = vectorizer.transform(df_content)
            
            # 7. predict
            sentimen_value = svc.predict(df_content)
            
            return sentimen_value
        
        
        # Melakukan prediksi menggunakan function "prediksi_sentimen" ----------
        nilai_pred = prediksi_sentimen(df_content["content"])

        if nilai_pred[0] == 1:
            return render_template("index.html", data = "Positif", content = content)
        else:
            return render_template("index.html", data = "Negatif", content = content)

    return render_template("index.html")
    

if __name__ == "__main__":
	app.run(debug = True)