"""
die Funktion datasplitting dient dazu, die Input-Datei in mehrere Dateien mit einer maximalen Länge von jeweils 30 000 Zeilen zu splitten

- Zunächst werden alle Jsonl-Dateien im Ordner gelöscht (notwendig, da wir die Zeilen nur appenden)
- dann wird der Datensatz gelesen und Zeile für Zeile geladen
- es wird der Key Work_id ausgelesen
- sollte eine Work_id im Datensatz mehrfach vorkommen, werden die nachfolgenden Zeilen mit dieser ID übersprungen
- ansonsten wird die Zeile einfach an eine neue Datei hinzugefügt
- nach 30 000 Zeilen werden die Counter überschrieben
"""

#### ---------- import ----------
import os
import glob
import json
from pathlib import Path


#### ---------- definition ----------
path = Path(__file__)
root = (path.parent).parent


#### ---------- code ----------
def datasplitting(dataset_filepath, dataset_name):

    workids = []
    linecounter = 0
    filecounter = 1
    filepath = "{0}\\data\\dataset_processed\\split\\{1}_split".format(root.absolute(), dataset_name)

    for file in glob.glob("{0}*.jsonl".format(filepath)):                                                               # alle Jsonl-Dateien im Ordner löschen (notwendig, da wir die Zeilen appenden)
        os.remove(file)

    with open(dataset_filepath, "r") as file_input:

        for line in file_input:
            json_line = json.loads(line)

            workid = json_line['work_id']
            if workid in workids:                                                                                       # Entfernen redundanter Daten; in dem Liste mit bisherigen Workids erstellt und abgeglichen wird
                continue
            else:
                workids.append(workid)

            linecounter += 1

            with open("{0}{1}.jsonl".format(filepath, filecounter), "a+") as file_output:                         # Zeile wird an neue Datei angefügt
                json.dump(json_line, file_output)
                file_output.write('\n')

            if linecounter == 30000:                                                                                    # nach 30 000 Zeilen wird linecounter zurückgesetzt und filecounter erhöht
                linecounter = 0
                filecounter += 1

    return filecounter                                                                                                  # Rückgabe der Datei-Anzahl
