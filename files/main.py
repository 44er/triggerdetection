"""
dieses Skript ist das Main-Skript und somit das einzige Skript, was per Hand ausgeführt werden muss
es werden alle notwendigen einzelnen Skripte (vom ursprünglichen Datensatz bis hin zum RoBerta-Netz und Speicherung der Ausgabe) nacheinander selbstständig aufgerufen
es müssen nur die Parameter dataset_filepath und dataset_name angepasst werden
ebenso muss die auszuführende Funktion auskommentiert werden (ob der Datensatz zum Training verwendet oder gelabelt werden soll)
damit auch alle Dateien (wie Akronym- und TF-IDF-Listen) vorhanden sind, empfiehlt sich, zuerst einen Trainings-Loop zu durchlaufen
"""

#### ---------- import ----------
import subprocess
from pathlib import Path
from dataset_splitting import datasplitting


#### ---------- definition ----------
path = Path(__file__)
root = (path.parent).parent


# ---------- functions ----------
def processing(name, filecount):                                                                                        # Funktion, welche das Skript mit entsprechenden Parametern aufruft und ausführt
    subprocess.run(["python", "{0}.py".format(name), filecount, dataset_name])


def dataset_training(dataset_filepath, dataset_name):                                                                   # für Datensätze, welche zum Training verwendet werden sollen
    filecount = datasplitting(dataset_filepath, dataset_name)                                                           # Splitten der Datei, Ermittlung der Anzahl der neuen Dateien
    processing("preprocessing_acronymcleanup", filecount)                                                         # Erstellung einer Liste mit Akronymen und ihrer Langform
    processing("preprocessing_substitution", filecount)                                                           # Durchlaufen eines Teiles des Preprocessings, nur notwendig für Erstellung der TF-IDF-Wortliste; programmablauftechnisch ist dies keine optimale Lösung
    processing("preprocessing_tfidf", filecount)                                                                  # Erstellung einer TF-IDF-Wortliste
    processing("overundersampling", filecount)                                                                    # Durchlaufen des Samplings, der Augmentation und des Preprocessings
    processing("roberta_training", filecount)                                                                     # Training des neuronalen Netzes (RoBERTa)


def dataset_labeling(dataset_filepath, dataset_name):                                                                   # für Datensätze, welche gelabelt werden sollen
    filecount = datasplitting(dataset_filepath, dataset_name)                                                           # Splitten der Datei, Ermittlung der Anzahl der neuen Dateien
    processing("preprocessing_substitution", filecount)                                                           # Durchlaufen eines Teiles des Preprocessings; theoretisch ist der gesamte Code nochmal in preprocessing_all enthalten, da dieses Skript hier jedoch wegen der TF-IDF-Liste vorhanden ist, wird es gleich nochmals verwendet und das andere Skript muss nicht erst noch angepasst werden
    processing("preprocessing_deletionlemmatization", filecount)                                                  # Durchlaufen des zweiten Teiles des Preprocessings
    processing("roberta_labeling", filecount)                                                                     # Labeling mithilfe des neuronalen Netzes (RoBERTa)


#### ---------- code ----------
dataset_filepath = "{0}\\.jsonl".format(root.absolute())                                                                # Pfad zum neuem Datensatz
dataset_name = ""                                                                                                       # Name des neuen Datensatzes

#dataset_training(dataset_filepath, dataset_name)                                                                       # Funktion, so dass Datensatz zum Training von RoBERTa verwendet wird
#dataset_labeling(dataset_filepath, dataset_name)                                                                       # Funktion, so dass Datensatz gelabelt wird
