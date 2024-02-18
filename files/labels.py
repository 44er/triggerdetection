"""
die Funktion getlabelcount dient dazu, die Gesamtanzahl aller Labels sowie die Zeilenanzahl in der angegebenen Datei zu ermitteln
die Funktion tranlsate_labels dient zum Übersetzen der binären Liste der zugeordneten Labels in die Wortform
"""

#### ---------- import ----------
import json
from numpy import array, zeros


#### ---------- definition ----------


#### ---------- code ----------

# ---------- Funktion zum Auswerten der Gesamtanzahl der Labels in der Datei ----------
def getlabelcount(path):
    with open(path, 'r') as file:                                                                                       # Datei öffnen
        linecount = 0
        labelcount = zeros((32,), dtype=int)                                                                      # Erzeugung Array mit 32 Nullen
        for line in file:
            json_line = json.loads(line)
            labels = array(json_line["labels"])
            labelcount += labels                                                                                        # Erhöhung Counter an den Stellen, an denen Label assigned ist
            linecount += 1
        return labelcount, linecount


# ---------- Funktion zum Auswerten der zugeordneten Labels ----------
def translate_labels(labels):
    label_assigned = []
    label_topcategory = []
    label_categories = ['pornographic-content', 'violence', 'death', 'sexual-assault', 'abuse', 'blood', 'suicide',     # Wortform der Labels
                  'pregnancy', 'child-abuse', 'incest', 'underage', 'homophobia', 'self-harm', 'dying', 'kidnapping',
                  'mental-illness', 'dissection', 'eating-disorders', 'abduction', 'body-hatred', 'childbirth',
                  'racism', 'sexism', 'miscarriages', 'transphobia', 'abortion', 'fat-phobia', 'animal-death',
                  'ableism', 'classism', 'misogyny', 'animal-cruelty']
    topcategories = {                                                                                                   # die Auswertung der Überkategorien ist momentan noch nicht im Programm implementiert, nur schon an dieser Stelle
        "Discrimination/Prejudice-related": ["classism", "homophobia", "misogyny", "racism", "sexism", "transphobia"],
        "Hostile Acts/Aggression-related": ["violence", "animal-cruelty", "sexual-assault", "abuse", "child-abuse", "abduction", "kidnapping"],
        "Pregnancy-related": ["pregnancy", "miscarriages", "childbirth", "abortion"],
        "Anatomy-related": ["dissection", "blood"],
        "Death-related": ["dying", "death", "animal-death"],
        "Mental Health-related": ["mental-illness", "suicide", "eating-disorders", "fat-phobia", "body-hatred", "self-harm"],
        "Sexuality-related": ["incest", "underage", "pornographic-content"]
    }

    # Auswertung der Label-Liste in Wortform
    m = 0
    while m < 31:                                                                                                       # die Labels, an deren Stellen sich in der übergebenen Liste eine 1 befindet, werden zu einer neuen Liste hinzugefügt
        if labels[m] == 1:
            label_assigned.append(label_categories[m])
        m += 1

    # Ausgabe der Oberkategorie
    for label in label_assigned:                                                                                        # die Überkategorien werden zwar mit ausgewertet und zurückgegeben, jedoch aktuell nicht weiter genutzt
        for topcategory, category in topcategories.items():
            if label in category and topcategory not in label_topcategory:
                label_topcategory.append(topcategory)

    return label_assigned, label_topcategory
