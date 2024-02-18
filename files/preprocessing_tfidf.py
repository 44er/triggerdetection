"""
dieses Skript dient dazu, eine Jsonl-Datei mit einer Liste an Wörtern zu Erstellen, welche später im Preprocessing entfernt werden sollen
dafür wird eine TF-IDF-Gewichtung auf die Trainings-Datensätze angewendet
da hierfür der Input etwas vorverarbeitet sein muss, durchlaufen somit alle Trainings-Datensätze nur für dieses Skript extra ein Teil des Preprocessings -> dieser Ablauf darf, bzw. sollte, hinterfragt werden; es war jedoch nach der zwanghaften Änderung im gesamten Programmablauf die einfachste Lösung

es werden alle Wörter gewichtet, welche in mindestens einem Dokument und maximal in 0,0003 % (bei 30 000 Zeilen sind dies 9 Zeilen) der Dokumente vorkommen
Wörter mit der Gewichtung 0 werden zu einer Liste hinzugefügt und im späteren Programmablauf entfernt
somit werden vor allem Namen, Rechtschreibfehler, Sonderbezeichnungen etc. entfernt

ursprünglich sollte eine normale TF-IDF-Gewichtung über alle Zeilen erfolgen, jedoch wurden trotz verschiedener Parameter zu viele sinnbehaftete Wörter mit 0 gewichtet
"""

#### ---------- import ----------
import sys
import json
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer


#### ---------- definition ----------
path = Path(__file__)
root = (path.parent).parent
filepath = "{0}\\data\\tfidf".format(root.absolute())
filecounter = 1

filecount = int(sys.argv[1])
filename_load = "{0}_pps".format(sys.argv[2])

dict = {}                                                                                                               # es wird ein Dictionary eingesetzt, damit Dopplungen unter den Dateien automatisch entfernt werden
vectorizer = TfidfVectorizer(input='content', use_idf=True, analyzer="word", vocabulary=None, ngram_range=(1, 1), min_df=1,  max_df=0.0003, lowercase=False)  # Anzahl der Dokumente, in denen ein Wort vorkommen muss, um beachtet zu werden: bei 30 000 Zeilen mind. 1 Dokument, max. 9 Dokumente


#### ---------- code ----------

while filecounter <= filecount:
    sentences = []

    with open("{0}\\data\\dataset_processed\\substitution\\{1}{2}.jsonl".format(root.absolute(), filename_load, filecounter), "r") as file_input:  # Laden der kurz vorverarbeiteten Dateien
        for line in file_input:
            text = json.loads(line)['text']                                                                             # Laden der vorverarbeiteten Wort-Liste

            text = [word.lower() for word in text]                                                                      # noch Umwandlung in Kleinschreibung, ist notwendig für später folgende Funktion
            sentences.append(" ".join(text))                                                                            # da Liste, Umwandlung zu String

        #-#-#-#-#
        """
        der Code aus diesem Abschnitt (gekennzeichnet mit #-#-#-#-#) ist Fremd-Code
        jedoch wurden einzelne Variablen-Bezeichnungen und Argumente angepasst
        von: https://hyperskill.org/learn/step/12910
        abgerufen am: 10.12.2023
        """
        vectorizer_matrix = vectorizer.fit_transform(sentences)                                                         # Erzeugung einer TF-IDF-Matrix
        vectorizer_vector = vectorizer_matrix.toarray()[0]                                                              # Auslesen der TF-IDF-Werte
        featurenames = vectorizer.get_feature_names_out()                                                               # Auslesen der Wörter
        #-#-#-#-#

        df = pd.DataFrame(vectorizer_vector, index=featurenames, columns=["TF-IDF"])                                    # Erzeugung eines Dataframes mit Wörtern (als Index) und den zugehörigen Gewichtungen

        df = df.loc[df['TF-IDF'] <= 0]                                                                                  # nur Wörter mit der Gewichtung 0 nehmen

        for word in df.index.tolist():                                                                                  # Hinzufügen der Wörter zu Dictionary, Dopplungen werden automatisch entfernt
            dict[word] = word

    filecounter += 1


words = list(dict.keys())

with open("{0}\\tfidf_words.jsonl".format(filepath), "w") as file_output:                                               # Erstellen einer Jsonl-Datei mit der Wortliste
    json.dump(words, file_output)
