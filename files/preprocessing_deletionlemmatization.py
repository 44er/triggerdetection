"""
dieses Skript dient zur Durchführung der zweiten Hälfte der regulären Datensatzvorverarbeitung
es wird bei zu labelnden Datensätzen eingesetzt
"""

#### ---------- import ----------
import os
import sys
import nltk
import json
from pathlib import Path
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


#### ---------- definition ----------
path = Path(__file__)
root = (path.parent).parent
filepath = "{0}\\data\\dataset_processed\\deletionlemmatization".format(root.absolute())
filecounter = 1

filecount = int(sys.argv[1])
filename_load = "{0}_pps".format(sys.argv[2])
filename_target = "{0}_ppdl".format(sys.argv[2])

lemmatizer = WordNetLemmatizer()                                                                                        # Initialisierung Lemmatizer
stopwords = set(nltk.corpus.stopwords.words("english"))                                                                 # Laden der Stopp-Wörter (aus NLTK-Bibliothek)

# Laden des Datensatzes mit den TF-IDF Wörtern
with open("{0}\\data\\tfidf\\tfidf_words.jsonl".format(root.absolute())) as file:                                       # Datei laden, in der die zu entfernenden TF-IDF gewichteten Wörter enthalten sind
    nonecessarywords = json.load(file)

# ---------- functions ----------
# Lemmatizing
#-#-#-#-#
"""
der Code aus diesem Abschnitt (gekennzeichnet mit #-#-#-#-#) ist Fremd-Code
von: https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
abgerufen am: 02.12.2023
"""
def get_wordnet_pos(tag):
    tag_dict = {"J": wordnet.ADJ,                                                                                       # Erstellung Dictionary mit POS-Tags
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)                                                                              # Rückgabe Tag; standardmäßige Rückgabe von Substantiv, da POS-Tag evtl. nicht eindeutig zuordnungsbar und somit Substantiv im Normalfall die neutralste Form
#-#-#-#-#


#### ---------- code ----------
while filecounter <= filecount:
    if os.path.exists("{0}\\{1}{2}.jsonl".format(filepath, filename_target, filecounter)):                        # Löschen der bestehenden Dateien; notwendig, da wir nur appenden
        os.remove("{0}\\{1}{2}.jsonl".format(filepath, filename_target, filecounter))
    filecounter += 1


filecounter = 1


while filecounter <= filecount:
    with open("{0}\\data\\dataset_processed\\substitution\\{1}{2}.jsonl".format(root.absolute(), filename_load, filecounter), "r") as file_input:   # Datei sowie Zeile laden
        for line in file_input:
            json_line = json.loads(line)
            text = json_line["text"]

            words = [word for word in text if len(word) > 1 and word not in nonecessarywords and word not in stopwords] # Entfernung von Wörtern: mit Wortlänge = 1, Stoppwörtern sowie aus der Datei mit den TF-IDF gewichteten Wörtern enthaltene Wörter

            words = [word.lower() for word in words]                                                                    # Umwandlung in Kleinschreibung

            text = ' '.join(words)                                                                                      # Wortliste wird zu String zusammengefügt

            # Lemmatizing
            #-#-#-#-#
            """
            der Code aus diesem Abschnitt (gekennzeichnet mit #-#-#-#-#) ist Fremd-Code
            er wurde jedoch stellenweise noch angepasst und optimiert
            von: https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
            abgerufen am: 02.12.2023
            """
            words = nltk.word_tokenize(text)                                                                            # Tokenisierung des Textes
            pos_tags = nltk.pos_tag(words)                                                                              # POS-Tagging (um Wortart der einzelnen Wörter zu erhalten)
            words = [lemmatizer.lemmatize(w, get_wordnet_pos(tag[0].upper())) for w, tag in pos_tags]                   # Lemmatisierung (vorher Anpassung der Eingabe)
            #-#-#-#-#

            with open("{0}\\{1}{2}.jsonl".format(filepath, filename_target, filecounter), "a+") as file_output:   # Anfügen der bearbeiteten Zeile an Datei
                json.dump({'work_id': json_line["work_id"], 'text': words}, file_output)
                file_output.write('\n')

    filecounter += 1
