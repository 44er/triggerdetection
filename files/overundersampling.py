"""
dieses Skript dient zur Durchführung des Samplings
es werden andauernd die Skripte Augmentation und Preprocessing aufgerufen und ausgeführt

es wird zunächst die Anzahl der einzelnen Labels in der Datei ermittelt und daraus die jeweilige relative Label-Häufigkeit
dann wird eine relative Ziel-Häufigkeit für die Labels berechnet und daraufhin die Ziel-Anzahl der Zeilen mit diesem Label
es wird zufällig eine Zeile im Text ausgewählt
-> wenn die Zeile noch nie ausgewählt wurde und die Labels noch benötigt werden, wird sie übernommen
-> wenn die Zeile schon mal ausgewählt wurde, die Labels aber noch benötigt werden, wird eine Datensatzerweiterung durchgeführt und anschließend wird sie übernommen
-> eine Zeile kann maximal 4 weitere Male vervielfacht und verändert werden, d.h. jede Zeile kann maximal 5 Mal vorkommen
-> die Anzahlen werden in einem Dictionary gespeichert
-> wird eine Zeile ausgewählt, deren Labels nicht mehr benötigt werden (oder schon 5 Mal verwendet wurde), wird sie aus der Auswahl entfernt
-> wird eine Zeile ausgewählt, welche mehrere Labels besitzt, und manche dieser noch benötigt werden und andere nicht mehr, dann wird
eine Kostenberechnung durchgeführt (Abstand aller Labels zum Ziel-Wert) und die Zeile anschließend entweder verworfen oder übernommen

Funktionen:
Y = X                 -> ursprüngliche relative Häufigkeit der Label-Verteilung
Y = sqrt{X} / 2       -> relative Ziel-Häufigkeit
Y = sqrt{X} / 2 - X   -> Differenzfunktion, so oft sollen die Labels vervielfacht werden

Zwischenzeitlich wurde eine Datenbank zur Funktionsausführung eingebunden, um die Datensatzerweiterungsfunktionen einmalig hintereinander auszuführen zu können
dafür folgende Hinweise, sollte dies wieder implementiert werden:
- DB erstellen, pro Zeile mit [filecounter, linenum, minimum, maximum, func1, func2]
- diese Datei laden, die Funktionen zuordnen
- Funktionen ausführen
- Ergebnisse an Sampling-Datei anfügen
- geg. vorher schon komplettes Datenvorverarbeitung machen, damit im Falle keiner Datensatzerweiterung Zeile einfach nur noch geladen und angefügt werden muss
"""

#### ---------- import ----------
import os
import sys
import json
import random
import numpy as np
import linecache as lc
from numpy import zeros
from pathlib import Path
from labels import getlabelcount
from augmentation import augmentation
from preprocessing_all import preprocessing, prepreprocessing


#### ---------- definition ----------
path = Path(__file__)
root = (path.parent).parent
filepath_load = "{0}\\data\\dataset_processed\\split".format(root.absolute())                                           # gesplittete Dateien werden geladen
filepath_target = "{0}\\data\\dataset_sampled".format(root.absolute())
filecounter = 1

filecount = int(sys.argv[1])
filename_load = "{0}_split".format(sys.argv[2])
filename_target = "{0}_sampled".format(sys.argv[2])

label_zero = zeros((32,), dtype=int)                                                                              # Erzeugung Array mit 32 Nullen

# ---------- functions ----------
def labeladjustment(x):                                                                                                 # Skalierungsfunktion, Undersampling nur bis auf 50 % -> Labels mit 100 % Häufigkeit werden nur zu 50 % gesampelt
    return x**0.5 / 2


#### ---------- code ----------
while filecounter <= filecount:                                                                                         # Löschen bestehender Jsonl-Dateien; notwendig, da wir an Dateien nur appenden
    if os.path.exists("{0}\\{1}{2}.jsonl".format(filepath_target, filename_target, filecounter)):
        os.remove("{0}\\{1}{2}.jsonl".format(filepath_target, filename_target, filecounter))
    filecounter += 1


filecounter = 1


# ---------- Over-/Undersampling ----------
while filecounter <= filecount:                                                                                         # alle gesplittete Dateien durchgehen

    labelcount, linecount = getlabelcount("{0}\\{1}{2}.jsonl".format(filepath_load, filename_load, filecounter))  # Ermittlung Gesamtanzahl jedes Labels sowie Zeilenzahl in Datei
    relativecount = labelcount / linecount                                                                              # relative Häufigkeit (0 bis 1) der Labels in der Datei berechnen (0 heißt Label tritt nicht auf, 1 bedeutet Label ist in jeder Zeile enthalten); lineare Funktion
    newrelcount = labeladjustment(relativecount)                                                                        # Angepasste Ziel-Häufigkeit der Labels
    newlinecount = newrelcount * linecount                                                                              # Ermitteln der neuen Zeilenanzahl
    newlinecount = newlinecount.astype(int)                                                                             # Abrunden auf ganze Zeilen
    foundlines = {}                                                                                                     # Dictionary erstellen
    samplinglines = [i + 1 for i in range(linecount)]                                                                   # Liste mit den Zeilennummern aller Zeilen, welche gesampled werden dürfen

    while len(samplinglines) > 0:                                                                                       # Schleife wird solange ausgeführt, bis Liste mit den samplebaren Zeilen leer ist

        lineid = int(random.uniform(0, len(samplinglines)))                                                          # zufällige Zahl, Maximum: Anzahl aller samplebaren Zeilen; int rundet immer ab!
        linenum = samplinglines[lineid]                                                                                 # Ermittlung der expliziten Zeilennummer zu dieser Zufallszahl

        if linenum in foundlines.keys():                                                                                # ist diese Zeile schon im Dictionary?
            foundlines[linenum] += 1                                                                                    # wenn ja, dann Counter in Datei erhöhen
        else:
            foundlines[linenum] = 1                                                                                     # wenn nein, dann Eintrag erstellen

        line = lc.getline("{0}\\{1}{2}.jsonl".format(filepath_load, filename_load, filecounter), linenum)         # entsprechende Zeile im Datensatz auslesen
        json_line = json.loads(line)                                                                                    # entsprechende Zeile formatieren
        labels = json_line["labels"]                                                                                    # Labels der Zeile einlesen

        if np.all(labels == label_zero):                                                                                # Abfrage, ob Fan-Fiction-Werk ungelabelt; wenn ja, dann direkt überspringen
            samplinglines.remove(linenum)
            continue

        cost = 0                                                                                                        # Kostenfunktion für Labels
        for i, l in enumerate(labels):                                                                                  # i - index, l - label
            if l:                                                                                                       # ist Label vorhanden?
                cost -= newlinecount[i]                                                                                 # Kosten berechnen durch Addition der Anzahlen an fehlenden Zeilen bei den ausgewählten Labels

        if cost >= 0:
            samplinglines.remove(linenum)                                                                               # wenn Kosten zu hoch, dann Entfernen der Zeile aus der Liste mit den samplebaren Zeilen
        else:
            if foundlines[linenum] == 5:                                                                                # wenn Counter im Dictionary bei 5, dann Entfernen der Zeile aus der Liste mit den samplebaren Zeilen
                samplinglines.remove(linenum)
            newlinecount -= np.array(labels)                                                                            # die neu ausgewerteten Labels werden von der Anzahl der noch fehlenden Labels abgezogen; Array Labels wurde kopiert (Python Syntax)

            text = json_line["text"]

            if foundlines[linenum] > 1:                                                                                 # Datensatzerweiterung, wenn Zeile nicht zum ersten mal verwendet wird
                text = prepreprocessing(text)                                                                           # Durchführung kurzer Vorverarbeitung
                text = augmentation(text)                                                                               # Durchführung Datensatzerweiterung

            text = preprocessing(text)                                                                                  # Durchführung regulärer Vorverarbeitung

            with open("{0}\\{1}{2}.jsonl".format(filepath_target, filename_target, filecounter), "a+") as file_output:  # bearbeitete Zeile an Sample-Datei anfügen
                json.dump({'work_id': json_line["work_id"], 'labels': labels, 'text': text}, file_output)
                file_output.write('\n')

    filecounter += 1
