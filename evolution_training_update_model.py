import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Bidirectional,BatchNormalization, Dropout , LSTM
from deap import base, creator, tools, algorithms
import random
import tensorflow as tf

#<Name>

# Example training data
sentences = [
    "Trage den Blutzuckerwert für Frau <Name> ein.",
  "Aktualisiere Herrn <Name> Blutzucker auf 145.",
  "Setze den Blutzucker bei dem jungen Mann auf 110.",
  "Füge einen neuen Blutzuckerwert bei Frau <Name> hinzu.",
  "Bitte ändere den Blutzucker von dem Bewohner auf 98.",
  "Blutzucker bei der Dame auf 125 korrigieren.",
  "Für Herrn <Name> bitte den Blutzucker auf 132 setzen.",
  "Der neue Blutzuckerwert von dem Mädchen ist 105.",
  "Blutzuckerwert von Frau <Name> aktualisieren auf 117.",
  "Bei der Person bitte den Blutzucker auf 101 eintragen.",
    # 2.Blutdruck (Intent 1)
    "Trage den Blutdruck von Frau <Name> mit 130 zu 85 ein.",
    "Bitte setze den Blutdruck bei Herrn <Name> auf 125 zu 80.",
    "Frau <Name> hat jetzt 140 zu 90, bitte speichern.",
    "Aktualisiere den Blutdruck vom Bewohner auf 135 zu 88.",
    "Blutdruck bei dem Mann beträgt 128 zu 82 – bitte eintragen.",
    "Für die Bewohnerin bitte 120 zu 75 als Blutdruck übernehmen.",
    "Der neue Blutdruck bei Herrn <Name> ist 138 zu 84.",
    "Blutdruckwert für Frau <Name> ist jetzt 132 zu 86 – bitte übernehmen.",
    "Setze bei der Dame den Blutdruck auf 145 zu 95.",
    "Trage 122 zu 78 als neuen Blutdruck bei dem Herrn ein.",
    # 3.Temperatur (Intent 2)
  "Trage die Temperatur von Frau <Name> mit 38,2 Grad ein.",
  "Bitte setze die Temperatur bei Herrn <Name> auf 36,9 Grad.",
  "Frau <Name> hat jetzt 37,5 Grad, bitte eintragen.",
  "Aktualisiere die Temperatur von dem Bewohner auf 38 Grad.",
  "Temperatur bei dem Mann beträgt 37,1 – bitte speichern.",
  "Für die Bewohnerin bitte 36,8 Grad als Temperatur eintragen.",
  "Neue Temperatur bei Frau <Name> ist 38,4 Grad.",
  "Temperaturwert für Herrn <Name> ist jetzt 37 Grad, bitte übernehmen.",
  "Setze bei der Dame die Temperatur auf 39 Grad.",
  "Trage 36,6 Grad bei dem Herrn als neue Temperatur ein.",
    # 4.Gewicht (Intent 3)
  "Trage das Gewicht von Frau <Name>  mit 72 Kilogramm ein.",
  "Bitte setze das Gewicht bei Herrn <Name>  auf 80 Kilo.",
  "Frau <Name> wiegt jetzt 68,5 Kilogramm, bitte speichern.",
  "Aktualisiere das Gewicht des Bewohners auf 75 kg.",
  "Das neue Gewicht bei dem Mann beträgt 83,2 Kilo – bitte eintragen.",
  "Für die Bewohnerin bitte 59 Kilogramm als Gewicht übernehmen.",
  "Der neue Gewichtswert bei Herrn <Name> ist 90 Kilogramm.",
  "Gewicht von Frau <Name> beträgt jetzt 66,3 kg – bitte übernehmen.",
  "Setze bei der Dame das Gewicht auf 70,0 Kilogramm.",
  "Trage 77 Kilo bei dem Herrn als neues Gewicht ein.",
    # 5.Medikamente verabreichen (Intent 4)
  "Gib Frau <Name> jetzt ihre Blutdrucktablette.",
  "Verabreiche Herrn <Name> Paracetamol 500 Milligramm.",
  "Bitte Ibuprofen an die Bewohnerin geben.",
  "Trage ein, dass Herr <Name> seine Medikamente bekommen hat.",
  "Verabreiche die morgendliche Dosis an Frau <Name>.",
  "Frau <Name> soll heute Abend Insulin bekommen.",
  "Gib dem Bewohner jetzt seine Schmerzmittel.",
  "Die Patientin braucht ihre Tabletten um 14 Uhr.",
  "Vermerke, dass das Antibiotikum bei Herrn <Name> verabreicht wurde.",
  "Medikamentengabe an Frau <Name> bitte jetzt durchführen.",
    # 6.Medikamente verabreichen (Intent 5)
  "Trage 500 Milliliter Wasser für Frau <Name> als Einfuhr ein.",
  "Herr <Name> hat 300 ml Tee getrunken, bitte eintragen.",
  "Bei der Bewohnerin 250 ml Urin als Ausfuhr erfassen.",
  "Frau <Name> hat heute zweimal Stuhlgang gehabt.",
  "Bitte dokumentiere 1 Liter Flüssigkeit bei Herrn <Name>.",
  "Trage für Frau <Name> 200 ml Apfelsaft als Einfuhr ein.",
  "Der Bewohner hat 350 ml ausgeschieden – bitte erfassen.",
  "Vermerke 400 ml Wasserzufuhr bei der Patientin.",
  "Herr <Name> hatte heute einen weichen Stuhl – bitte dokumentieren.",
  "Frau <Name> hat 100 ml Kaffee getrunken, bitte übernehmen.",
    # 7.Stuhlgang (Intent 6)
  "Frau <Name> hatte heute einmal Stuhlgang.",
  "Trage bei Herrn <Name> weichen Stuhlgang ein.",
  "Frau <Name> hat heute dreimal Stuhl abgesetzt.",
  "Der Bewohner hatte keinen Stuhlgang seit gestern.",
  "Bitte notiere festen Stuhl bei Frau <Name>.",
  "Herr <Name> hatte heute Morgen Durchfall.",
  "Vermerke bei der Patientin einmal normalen Stuhl.",
  "Bei Frau <Name> war der Stuhl sehr hart – bitte dokumentieren.",
  "Trage Durchfall bei dem Bewohner für heute ein.",
  "Die Bewohnerin hatte heute keinen Stuhl – bitte festhalten.",
  "Trage heute bei Frau <Name> einen festen Stuhlgang ein.",
  "Der Stuhl von Herrn <Name> war breiig, bitte dokumentieren.",
  "Frau <Name> hatte flüssigen Stuhl – vermutlich Durchfall.",
  "Heute war der Stuhl bei Herrn <Name> sehr hart.",
  "Die Bewohnerin hatte einen normalen, geformten Stuhl.",
  "Vermerke bei Frau <Name> einen wässrigen Stuhlgang.",
  "Der Stuhl war weich, bitte bei Herrn <Name> eintragen.",
  "Bei Frau <Name> wurde pastöser Stuhl festgestellt.",
  "Bitte trag klumpigen Stuhl bei dem Bewohner ein.",
  "Der heutige Stuhlgang von Frau <Name> war leicht breiig.",
    # 8.Einzelbetreuung (Intent 7)
  "Bitte trage eine halbe Stunde Einzelbetreuung bei Frau <Name> ein.",
  "Herr <Name> hatte heute 20 Minuten Einzelbetreuung.",
  "Ich habe gerade eine Stunde mit Frau <Name> zur Einzelbetreuung verbracht.",
  "Frau <Name> wurde heute einzeln betreut, bitte dokumentieren.",
  "Trage Einzelbetreuung bei dem Bewohner von 10 bis 10:30 Uhr ein.",
  "Bei Frau <Name> fand heute keine Einzelbetreuung statt.",
  "Bitte erfasse die heutige Einzelbetreuung bei Herrn <Name>.",
  "Ich habe mit der Bewohnerin 45 Minuten Einzelbetreuung gemacht.",
  "Herr <Name> hatte heute Einzelbetreuung wegen Unruhe.",
  "Dokumentiere Einzelbetreuung bei Frau <Name> – Gedächtnistraining.",
    # 9.Verbandswechsel (Intent 8)
  "Bitte dokumentiere den Verbandswechsel bei Frau <Name> am linken Bein.",
  "Herr <Name> hatte heute einen Verbandswechsel an der Hand.",
  "Frau <Name> musste wegen der Wunde am Rücken neu verbunden werden.",
  "Trage ein, dass bei Herrn <Name> der Verband heute gewechselt wurde.",
  "Bei der Bewohnerin wurde der Fußverband erneuert.",
  "Der Verbandswechsel bei Frau <Name> fand um 14 Uhr statt.",
  "Bitte erfasse einen Verbandswechsel bei Herrn <Name> am rechten Arm.",
  "Ich habe heute bei Frau <Name> den Verband am Knie gewechselt.",
  "Vermerke den täglichen Verbandswechsel bei dem Bewohner.",
  "Die Patientin erhielt heute Vormittag einen frischen Verband.",
    # 10.Medikamente stellen (Intent 9)
  "Bitte dokumentiere, dass ich die Medikamente für Frau <Name> gestellt habe.",
  "Herr <Name> hat seine Wochenmedikation bekommen – bitte eintragen.",
  "Ich habe die Medikamente für die Bewohnerin für heute vorbereitet.",
  "Aktualisiere Eintrag: Medikamente verabreicht bei Frau <Name> um 16 Uhr.",
  "Notiere, dass Frau <Name> ihre Medikamente um 14 Uhr erhalten hat.",
  "Trage ein, dass die Tabletten für Herrn <Name> gestellt wurden.",
  "Verabreichung von Insulin bei Frau <Name> dokumentieren.",
  "Frau <Name> hat jetzt ihre Medikamente im Dispenser – bitte vermerken.",
  "Die Medikamentenbox für Herrn <Name> ist befüllt.",
  "Ich habe die Medikamente für die nächste Woche bei Frau <Name> gestellt.",
  "Vermerke, dass die Morgenmedikation für die Bewohnerin bereitliegt.",
  "Medikamente für Herrn <Name> wurden heute um 9 Uhr gestellt.",
  "Trage ein: Medikamente für Montag bis Freitag bei Frau <Name> vorbereitet.",
    # 11.Nahrungsaufnahme  (Intent 10)
  "Frau <Name> hat heute gut gegessen.",
  "Herr <Name> hat nur die Hälfte seiner Mahlzeit zu sich genommen.",
  "Bitte dokumentiere, dass Frau <Name> nichts gegessen hat.",
  "Die Bewohnerin hat heute das Mittagessen vollständig aufgegessen.",
  "Herr <Name> wollte heute nur Suppe essen.",
  "Trage ein: Frau <Name> hat schlecht gegessen.",
  "Bei Herrn <Name> war die Nahrungsaufnahme normal.",
  "Frau <Name> hat heute alles stehen lassen.",
  "Der Bewohner hat nur wenig vom Abendessen gegessen.",
  "Bitte vermerke, dass die Patientin ihre Mahlzeit komplett aufgegessen hat.",
    # 12.Flüssigkeitsaufnahme   (Intent 11)
  "Frau <Name> hat heute einen Liter getrunken.",
  "Bitte dokumentiere: Herr <Name> hat nur 500 Milliliter Flüssigkeit aufgenommen.",
  "Die Bewohnerin hat heute ausreichend getrunken.",
  "Frau <Name> hat kaum etwas getrunken – bitte eintragen.",
  "Trage ein, dass Herr <Name> heute zwei Gläser Wasser hatte.",
  "Frau <Name> hat bisher nur eine Tasse Tee getrunken.",
  "Herr <Name> hat heute einen halben Liter getrunken.",
  "Bitte notiere: Die Trinkmenge bei Frau <Name> war unzureichend.",
  "Die Bewohnerin hat heute drei Tassen Tee und ein Glas Saft getrunken.",
  "Vermerke, dass Herr <Name> heute insgesamt etwa 1,2 Liter getrunken hat.",
    # 13.Fixierung mit Bauchgurt im Bett   (Intent 12)
  "Frau <Name> wurde im Bett mit einem Bauchgurt fixiert.",
  "Bitte dokumentiere die Fixierung mit Bauchgurt bei Herrn <Name>.",
  "Heute wurde bei Frau <Name> eine Fixierung im Bett durchgeführt.",
  "Herr <Name> wurde mit Bauchgurt gesichert, um ein Herausfallen zu verhindern.",
  "Die Bewohnerin war heute im Bett mit einem Bauchgurt fixiert.",
  "Ich habe bei Frau <Name> die Fixierung mit Bauchgurt angelegt.",
  "Trage ein: Fixierung mit Bauchgurt bei Herrn <Name> im Bett.",
  "Frau <Name> wurde zum Eigenschutz im Bett fixiert.",
  "Bitte vermerke, dass bei dem Bewohner der Bauchgurt angelegt wurde.",
  "Heute wurde bei Frau <Name> eine Fixierung im Bett notwendig.",
    # 14.Entfernen der Fixierung mit Bauchgurt im Bett   (Intent 13)
  "Der Bauchgurt wurde bei Frau <Name> im Bett entfernt.",
  "Bitte dokumentiere, dass die Fixierung bei Herrn <Name> aufgehoben wurde.",
  "Heute wurde bei Frau <Name> die Bettfixierung gelöst.",
  "Ergänze, dass die Gruppentherapie mit Bewohnern aus Wohnbereich 2 stattgefunden hat.",
  "Herr <Name> musste nicht mehr fixiert werden – der Bauchgurt ist ab.",
  "Die Fixierung im Bett bei der Bewohnerin wurde heute entfernt.",
  "Frau <Name> ist jetzt ohne Fixierung, der Bauchgurt wurde abgenommen.",
  "Trage ein: Fixierung bei Herrn <Name> wurde im Laufe des Vormittags aufgehoben.",
  "Frau <Name> benötigt keine Fixierung mehr – Gurt wurde entfernt.",
  "Der Bauchgurt wurde bei dem Bewohner heute Vormittag gelöst.",
  "Fixierung bei Frau <Name> beendet – Bettgurt entfernt und dokumentiert.",
    # 15.Gruppenbetreuung   (Intent 14)
  "Die Gruppenbetreuung mit Frau <Name> und Herrn <Name> hat heute eine Stunde gedauert.",
  "Bitte dokumentiere die Gruppenbetreuung im Gemeinschaftsraum.",
  "Heute fand eine Gruppenbetreuung mit fünf Bewohnern statt.",
  "Herr <Name> hat an der Gruppenbetreuung teilgenommen.",
  "Die Bewohnerin Frau <Name> war bei der Gruppenbetreuung dabei.",
  "Trage ein: Gruppenbetreuung im Speisesaal von 14 bis 15 Uhr.",
  "Bei der heutigen Gruppenbetreuung wurde Gesellschaftsspiele gespielt.",
  "Die Gruppenbetreuung wurde erfolgreich durchgeführt.",
  "Bitte vermerke, dass die Gruppenbetreuung heute ausgefallen ist.",
  "Frau <Name> und Herr <Name> waren bei der letzten Gruppenbetreuung anwesend.",
    # 16.Tagesgruppe   (Intent 15)
  "Frau <Name> hat heute an der Tagesgruppe teilgenommen.",
  "Bitte dokumentiere die Anwesenheit von Herrn <Name> in der Tagesgruppe.",
  "Heute war Frau <Name> in der Tagesgruppe aktiv dabei.",
  "Herr <Name> hat die Tagesgruppe besucht.",
  "Die Bewohnerin nahm an der Tagesgruppe im Gemeinschaftsraum teil.",
  "Trage ein: Tagesgruppe von 9 bis 15 Uhr mit fünf Teilnehmern.",
  "Bei der heutigen Tagesgruppe wurden kreative Aktivitäten durchgeführt.",
  "Die Tagesgruppe hat heute wie geplant stattgefunden.",
  "Bitte vermerke, dass die Tagesgruppe heute ausgefallen ist.",
  "Frau <Name> und Herr <Name> waren bei der letzten Tagesgruppe anwesend.",
    # 17.Medikamente vorbereiten   (Intent 16)
  "Ich habe die Medikamente für Frau <Name> vorbereitet.",
  "Bitte dokumentiere, dass die Medikamente für Herrn <Name> bereitgestellt wurden.",
  "Heute wurden die Medikamente für Frau <Name> für die Woche vorbereitet.",
  "Herr <Name> hat seine Medikamente für den Tag erhalten – bitte eintragen.",
  "Die Medikamentenvorbereitung für Frau <Name> ist abgeschlossen.",
  "Ich habe die Tabletten für Herrn <Name> sortiert und bereitgelegt.",
  "Medikamente bei Frau <Name> vorbereitet – bitte im System speichern.",
  "Bereite Medikamente für Herrn <Name> vor – eintragen.",
  "Trage ein: Medikamente für Frau <Name> wurden vorbereitet.",
  "Die Medikamente für die Bewohnerin sind fertig gestellt.",
  "Medikamente für die nächste Woche wurden für den Bewohner vorbereitet.",
  "Bitte vermerke, dass die Medikamentenvorbereitung heute erfolgt ist.",
    # 18.Fixierung Bauchgurt im  Rollstuhl   (Intent 17)
  "Frau <Name> wurde im Rollstuhl mit einem Bauchgurt fixiert.",
  "Bitte dokumentiere die Fixierung mit Bauchgurt bei Herrn <Name> im Rollstuhl.",
  "Heute wurde bei Frau <Name> eine Fixierung im Rollstuhl durchgeführt.",
  "Herr <Name> wurde mit Bauchgurt im Rollstuhl gesichert, um ein Herausfallen zu verhindern.",
  "Die Bewohnerin war heute im Rollstuhl mit einem Bauchgurt fixiert.",
  "Ich habe bei Frau <Name> die Fixierung mit Bauchgurt im Rollstuhl angelegt.",
  "Trage ein: Fixierung mit Bauchgurt bei Herrn <Name> im Rollstuhl.",
  "Frau <Name> wurde zum Eigenschutz im Rollstuhl fixiert.",
  "Bitte vermerke, dass bei dem Bewohner der Bauchgurt im Rollstuhl angelegt wurde.",
  "Heute wurde bei Frau <Name> eine Fixierung im Rollstuhl notwendig.",
    # 19.Fixierung Bauchgurt im Rollstuhl entfernen  (Intent 18)
  "Der Bauchgurt wurde bei Frau <Name> im Rollstuhl entfernt.",
  "Bitte dokumentiere, dass die Fixierung bei Herrn <Name> im Rollstuhl aufgehoben wurde.",
  "Heute wurde bei Frau <Name> die Fixierung im Rollstuhl gelöst.",
  "Herr <Name> ist nicht mehr fixiert – der Bauchgurt im Rollstuhl wurde abgenommen.",
  "Die Fixierung mit Bauchgurt im Rollstuhl bei der Bewohnerin wurde heute entfernt.",
  "Frau <Name> ist jetzt ohne Fixierung, der Bauchgurt im Rollstuhl wurde abgenommen.",
  "Trage ein: Fixierung bei Herrn <Name> im Rollstuhl wurde aufgehoben.",
  "Frau <Name> benötigt keine Fixierung mehr – Bauchgurt im Rollstuhl entfernt.",
  "Der Bauchgurt im Rollstuhl wurde heute bei dem Bewohner gelöst.",
  "Fixierung bei Frau <Name> im Rollstuhl beendet – Gurt entfernt und dokumentiert.",
    # 20.Fixierung Therapietisch  (Intent 19)
  "Frau <Name> wurde am Therapietisch fixiert.",
  "Bitte dokumentiere die Fixierung bei Herrn <Name> am Therapietisch.",
  "Heute wurde bei Frau <Name> eine Fixierung am Therapietisch durchgeführt.",
  "Herr <Name> wurde am Therapietisch gesichert, um ein Herausfallen zu verhindern.",
  "Die Bewohnerin war heute am Therapietisch fixiert.",
  "Ich habe bei Frau <Name> die Fixierung am Therapietisch angelegt.",
  "Trage ein: Fixierung bei Herrn <Name> am Therapietisch.",
  "Frau <Name> wurde zum Eigenschutz am Therapietisch fixiert.",
  "Bitte vermerke, dass bei dem Bewohner die Fixierung am Therapietisch angewendet wurde.",
  "Heute wurde bei Frau <Name> eine Fixierung am Therapietisch notwendig.",
    # 21.Fixierung Therapietisch entfernen  (Intent 20)
  "Die Fixierung am Therapietisch bei Frau <Name> wurde entfernt.",
  "Bitte dokumentiere, dass die Fixierung bei Herrn <Name> am Therapietisch aufgehoben wurde.",
  "Heute wurde bei Frau <Name> die Fixierung am Therapietisch gelöst.",
  "Herr <Name> ist nicht mehr fixiert – die Fixierung am Therapietisch wurde abgenommen.",
  "Die Fixierung am Therapietisch bei der Bewohnerin wurde heute entfernt.",
  "Frau <Name> ist jetzt ohne Fixierung, die Fixierung am Therapietisch wurde abgenommen.",
  "Trage ein: Fixierung bei Herrn <Name> am Therapietisch wurde aufgehoben.",
  "Frau <Name> benötigt keine Fixierung mehr – Fixierung am Therapietisch entfernt.",
  "Die Fixierung am Therapietisch wurde heute bei dem Bewohner gelöst.",
  "Fixierung bei Frau <Name> am Therapietisch beendet und dokumentiert.",
    # 22.BETREUUNG NACH § 43b  (Intent 21)
  "Frau <Name> hat heute die Betreuung nach § 43b erhalten.",
  "Bitte dokumentiere die Betreuung nach § 43b bei Herrn <Name>.",
  "Heute wurde die Betreuung nach § 43b für Frau <Name> durchgeführt.",
  "Herr <Name> hat die Betreuung nach § 43b in Anspruch genommen.",
  "Die Bewohnerin nahm an der Betreuung nach § 43b teil.",
  "Trage ein: Betreuung nach § 43b für Frau <Name> von 10 bis 12 Uhr.",
  "Die Betreuung nach § 43b wurde heute erfolgreich abgeschlossen.",
  "Bitte vermerke, dass die Betreuung nach § 43b heute stattgefunden hat.",
  "Frau <Name> hat heute die Betreuung nach § 43b erhalten.",
  "Betreuung nach § 43b wurde bei Herrn <Name> dokumentiert.",
    # 23.Behandlungspflege  (Intent 22)
  "Frau <Name> hat heute die Behandlungspflege erhalten.",
  "Bitte dokumentiere die Behandlungspflege bei Herrn <Name>.",
  "Heute wurde die Behandlungspflege für Frau <Name> durchgeführt.",
  "Herr <Name> benötigt regelmäßig Behandlungspflege.",
  "Die Bewohnerin wurde heute mit Behandlungspflege versorgt.",
  "Trage ein: Behandlungspflege für Frau <Name> von 8 bis 10 Uhr.",
  "Die Behandlungspflege wurde heute erfolgreich abgeschlossen.",
  "Bitte vermerke, dass die Behandlungspflege heute stattgefunden hat.",
  "Frau <Name> hat heute die notwendige Behandlungspflege erhalten.",
  "Behandlungspflege wurde bei Herrn <Name> dokumentiert.",
    # 24.Medikamente bereit stellen  (Intent 23)
  "Ich habe die Medikamente für Frau <Name> bereitgestellt.",
  "Bitte dokumentiere, dass die Medikamente für Herrn <Name> bereitgestellt wurden.",
  "Heute wurden die Medikamente für Frau <Name> bereitgestellt.",
  "Herr <Name> hat seine Medikamente erhalten – bitte eintragen.",
  "Die Medikamentenbereitstellung für Frau <Name> ist abgeschlossen.",
  "Ich habe die Tabletten für Herrn <Name> sortiert und bereitgelegt.",
  "Trage ein: Medikamente für Frau <Name> wurden bereitgestellt.",
  "Die Medikamente für die Bewohnerin sind fertig gestellt.",
  "Medikamente für die nächste Woche wurden für den Bewohner bereitgestellt.",
  "Bitte vermerke, dass die Medikamentenbereitstellung heute erfolgt ist.",
    # 25.Beratende Gespräche  (Intent 24)
  "Frau <Name> hatte heute ein beratendes Gespräch.",
  "Bitte dokumentiere das beratende Gespräch mit Herrn <Name>.",
  "Heute wurde ein beratendes Gespräch bei Frau <Name> geführt.",
  "Herr <Name> nahm an einem beratenden Gespräch teil.",
  "KG extern bei Herrn <Name> heute erfolgt – dokumentieren.",
  "Die Bewohnerin hatte heute ein ausführliches Beratungsgespräch.",
  "Trage ein: Beratendes Gespräch mit Frau <Name> von 14 bis 15 Uhr.",
  "Das beratende Gespräch wurde heute erfolgreich abgeschlossen.",
  "Bitte vermerke, dass das beratende Gespräch heute stattgefunden hat.",
  "Frau <Name> hatte heute ein wichtiges Beratungsgespräch.",
  "Beratende Gespräche wurden bei Herrn <Name> dokumentiert.",
    # 26.KG extern  (Intent 25)
  "Frau <Name> hatte heute eine externe Krankengymnastik.",
  "Bitte dokumentiere die externe KG bei Herrn <Name>.",
  "Heute wurde bei Frau <Name> eine externe Krankengymnastik durchgeführt.",
  "Herr <Name> hat heute seine externe Krankengymnastik erhalten.",
  "Die Bewohnerin nahm an einer externen KG teil.",
  "Trage ein: Externe Krankengymnastik für Frau <Name> von 10 bis 11 Uhr.",
  "Die externe KG wurde heute erfolgreich abgeschlossen.",
  "Bitte vermerke, dass die externe Krankengymnastik heute stattgefunden hat.",
  "Frau <Name> hatte heute eine Behandlung bei der externen Krankengymnastik.",
  "Externe Krankengymnastik wurde bei Herrn <Name> dokumentiert.",
    # 27.Mobilitätsfaktor  (Intent 26)
  "Der Mobilitätsfaktor von Frau <Name> wurde heute bewertet.",
  "Bitte dokumentiere den Mobilitätsfaktor bei Herrn <Name>.",
  "Heute wurde der Mobilitätsfaktor für Frau <Name> aktualisiert.",
  "Herr <Name> hat einen verbesserten Mobilitätsfaktor.",
  "Die Bewohnerin zeigt einen stabilen Mobilitätsfaktor.",
  "Trage ein: Mobilitätsfaktor bei Frau <Name> wurde neu erfasst.",
  "Der Mobilitätsfaktor wurde heute erfolgreich bestimmt.",
  "Bitte vermerke, dass der Mobilitätsfaktor heute angepasst wurde.",
  "Frau <Name> hat einen niedrigeren Mobilitätsfaktor als zuvor.",
  "Mobilitätsfaktor bei Herrn <Name> wurde dokumentiert.",
    # 28.Validierende Gespräche  (Intent 27)
  "Frau <Name> hatte heute ein validierendes Gespräch.",
  "Bitte dokumentiere das validierende Gespräch mit Herrn <Name>.",
  "Heute wurde ein validierendes Gespräch bei Frau <Name> geführt.",
  "Herr <Name> nahm an einem validierenden Gespräch teil.",
  "Die Bewohnerin hatte heute ein ausführliches validierendes Gespräch.",
  "Trage ein: Validierendes Gespräch mit Frau <Name> von 15 bis 16 Uhr.",
  "Das validierende Gespräch wurde heute erfolgreich abgeschlossen.",
  "Bitte vermerke, dass das validierende Gespräch heute stattgefunden hat.",
  "Frau <Name> hatte heute ein wichtiges validierendes Gespräch.",
  "Validierende Gespräche wurden bei Herrn <Name> dokumentiert.",
    # 29.Lagerungsprotokoll  (Intent 28)
  "Das Lagerungsprotokoll für Frau <Name> wurde heute aktualisiert.",
  "Bitte dokumentiere die Lagerung bei Herrn <Name>.",
  "Heute wurde das Lagerungsprotokoll bei Frau <Name> ausgefüllt.",
  "Herr <Name> hat seine Lagerung gemäß Protokoll erhalten.",
  "Die Bewohnerin wurde heute entsprechend dem Lagerungsprotokoll gelagert.",
  "Trage ein: Lagerungsprotokoll für Frau <Name> von 8 bis 10 Uhr.",
  "Das Lagerungsprotokoll wurde heute erfolgreich abgeschlossen.",
  "Bitte vermerke, dass das Lagerungsprotokoll heute geführt wurde.",
  "Frau <Name> hat das Lagerungsprotokoll für heute erhalten.",
  "Lagerungsprotokoll wurde bei Herrn <Name> dokumentiert.",
    # 30.Duschen  (Intent 29)
  "Frau <Name> wurde heute geduscht.",
  "Bitte dokumentiere das Duschen bei Herrn <Name>.",
  "Heute hat Frau <Name> eine Dusche erhalten.",
  "Herr <Name> wurde heute beim Duschen unterstützt.",
  "Die Bewohnerin hat heute eine Dusche bekommen.",
  "Trage ein: Duschen für Frau <Name> von 9 bis 10 Uhr.",
  "Das Duschen wurde heute erfolgreich durchgeführt.",
  "Bitte vermerke, dass das Duschen heute stattgefunden hat.",
  "Frau <Name> wurde heute geduscht.",
  "Duschen wurde bei Herrn <Name> dokumentiert.",
    # 31.Frisör  (Intent 30)
  "Frau <Name> hatte heute einen Frisörtermin.",
  "Bitte dokumentiere den Frisörbesuch bei Herrn <Name>.",
  "Heute wurde Frau <Name> zum Frisör gebracht.",
  "Friseurtermin bei Herrn <Name> heute abgeschlossen – bitte notieren.",
  "Herr <Name> hatte heute einen Termin beim Frisör.",
  "Die Bewohnerin war heute beim Frisör.",
  "Trage ein: Frisörtermin für Frau <Name> um 14 Uhr.",
  "Der Frisörbesuch wurde heute erfolgreich durchgeführt.",
  "Bitte vermerke, dass der Frisörtermin heute stattgefunden hat.",
  "Frau <Name> hatte heute einen Frisörtermin.",
  "Frisörbesuch wurde bei Herrn <Name> dokumentiert.",
    # 32.Fußpflege  (Intent 31)
  "Frau <Name> hatte heute eine Fußpflege.",
  "Bitte dokumentiere die Fußpflege bei Herrn <Name>.",
  "Heute wurde Frau <Name> eine Fußpflege durchgeführt.",
  "Herr <Name> erhielt heute seine Fußpflege.",
  "Die Bewohnerin hatte heute eine Fußpflege-Behandlung.",
  "Trage ein: Fußpflege für Frau <Name> um 10 Uhr.",
  "Die Fußpflege wurde heute erfolgreich abgeschlossen.",
  "Bitte vermerke, dass die Fußpflege heute stattgefunden hat.",
  "Frau <Name> hatte heute einen Fußpflege-Termin.",
  "Fußpflege wurde bei Herrn <Name> dokumentiert.",
    # 33.Nagelpflege  (Intent 32)
  "Frau <Name> hatte heute eine Nagelpflege.",
  "Bitte dokumentiere die Nagelpflege bei Herrn <Name>.",
  "Heute wurde Frau <Name> eine Nagelpflege durchgeführt.",
  "Herr <Name> erhielt heute seine Nagelpflege.",
  "Die Bewohnerin hatte heute eine Nagelpflege-Behandlung.",
  "Trage ein: Nagelpflege für Frau <Name> um 11 Uhr.",
  "Die Nagelpflege wurde heute erfolgreich abgeschlossen.",
  "Bitte vermerke, dass die Nagelpflege heute stattgefunden hat.",
  "Frau <Name> hatte heute einen Nagelpflege-Termin.",
  "Nagelpflege wurde bei Herrn <Name> dokumentiert.",
    # 34.Haarwäsche  (Intent 33)
  "Frau <Name> hatte heute eine Haarwäsche.",
  "Bitte dokumentiere die Haarwäsche bei Herrn <Name>.",
  "Heute wurde Frau <Name> die Haare gewaschen.",
  "Herr <Name> erhielt heute eine Haarwäsche.",
  "Die Bewohnerin hatte heute eine Haarwäsche-Behandlung.",
  "Trage ein: Haarwäsche für Frau <Name> um 9 Uhr.",
  "Die Haarwäsche wurde heute erfolgreich durchgeführt.",
  "Bitte vermerke, dass die Haarwäsche heute stattgefunden hat.",
  "Frau <Name> hatte heute eine Haarwäsche.",
  "Haarwäsche wurde bei Herrn <Name> dokumentiert.",
    # 35.Fixierung Rollstuhlbremse anlegen  (Intent 34)
 "Bitte lege die Rollstuhlbremse bei Frau <Name> an.",
  "Die Rollstuhlbremse wurde heute bei Herrn <Name> angelegt.",
  "Herr <Name> benötigt, dass die Rollstuhlbremse angelegt wird.",
  "Rollstuhlbremse bei Frau <Name> wurde erfolgreich angelegt.",
  "Trage ein: Fixierung der Rollstuhlbremse bei Frau <Name>.",
  "Heute wurde die Rollstuhlbremse bei der Bewohnerin angelegt.",
  "Bitte dokumentiere, dass die Rollstuhlbremse heute angelegt wurde.",
  "Die Rollstuhlbremse wurde bei Herrn <Name> angelegt.",
  "Fixiere die Rollstuhlbremse bei Frau <Name>.",
  "Rollstuhlbremse anlegen wurde heute durchgeführt.",
    # 36.Fixierung Rollstuhlbremse entfernen  (Intent 35)
 "Bitte entferne die Rollstuhlbremse bei Frau <Name>.",
  "Die Rollstuhlbremse wurde heute bei Herrn <Name> entfernt.",
  "Herr <Name> benötigt, dass die Rollstuhlbremse entfernt wird.",
  "Rollstuhlbremse bei Frau <Name> wurde erfolgreich entfernt.",
  "Trage ein: Fixierung der Rollstuhlbremse bei Frau <Name> entfernt.",
  "Heute wurde die Rollstuhlbremse bei der Bewohnerin entfernt.",
  "Bitte dokumentiere, dass die Rollstuhlbremse heute entfernt wurde.",
  "Die Rollstuhlbremse wurde bei Herrn <Name> entfernt.",
  "Fixiere die Rollstuhlbremse bei Frau <Name> entfernen.",
  "Rollstuhlbremse entfernen wurde heute durchgeführt.",
    # 37.Individueller Toillettengang  (Intent 36)
 "Frau <Name> hatte heute einen individuellen Toilettengang.",
  "Bitte dokumentiere den Toilettengang bei Herrn <Name>.",
  "Heute wurde Frau <Name> beim Toilettengang unterstützt.",
  "Herr <Name> benötigte heute Hilfe beim Toilettengang.",
  "Unsere Pflegefachkraft hat die Medikamentenkontrolle bei <Name> durchgeführt.",
  "Die Bewohnerin hatte heute einen selbstständigen Toilettengang.",
  "Trage ein: Individueller Toilettengang für Frau <Name> um 15 Uhr.",
  "Der Toilettengang wurde heute erfolgreich durchgeführt.",
  "Bitte vermerke, dass der Toilettengang heute stattgefunden hat.",
  "Frau <Name> wurde heute beim Toilettengang begleitet.",
  "Toilettengang wurde bei Herrn <Name> dokumentiert.",
    # 38.Medikamentenkontrolle  (Intent 37)
 "Frau <Name> hat heute die Medikamentenkontrolle erhalten.",
  "Bitte dokumentiere die Medikamenten-Kontrolle bei Herrn <Name>.",
  "Pflegefachkraft hat Medikamentenkontrolle bei Herrn <Name> gemacht – bitte erfassen.",
  "Heute wurde bei Frau <Name> die Medikamentenkontrolle durchgeführt.",
  "Herr <Name> erhielt heute eine Medikamentenkontrolle.",
  "Unsere Pflegefachkraft hat die Medikamenten Kontrolle bei Herrn <Name> gemacht",
  "Pflegefachkraft hat Medikamentenkontrolle bei Herrn <Name> gemacht – bitte erfassen.",
  "Die Bewohnerin hatte heute eine Medikamentenkontrolle.",
  "Trage ein: Medikamentenkontrolle für Frau <Name> um 14 Uhr.",
  "Die Medikamentenkontrolle wurde heute erfolgreich abgeschlossen.",
  "Bitte vermerke, dass die Medikamentenkontrolle heute stattgefunden hat.",
  "Frau <Name> hatte heute eine Medikamentenkontrolle.",
  "Medikamentenkontrolle wurde bei Herrn <Name> dokumentiert.",
    # 39.Betreuung / Beobachtung  (Intent 38)
 "Frau <Name> wurde heute sorgfältig betreut und beobachtet.",
  "Bitte dokumentiere die Betreuung und Beobachtung bei Herrn <Name>.",
  "Heute fand eine intensive Beobachtung bei Frau <Name> statt.",
  "Herr <Name> erhielt heute eine besondere Betreuung und Beobachtung.",
  "Die Bewohnerin wurde heute aufmerksam betreut und beobachtet.",
  "Trage ein: Betreuung und Beobachtung für Frau <Name> um 13 Uhr.",
  "Die Betreuung und Beobachtung wurde heute erfolgreich durchgeführt.",
  "Bitte vermerke, dass die Beobachtung heute stattgefunden hat.",
  "Frau <Name> wurde heute umfassend betreut und beobachtet.",
  "Betreuung und Beobachtung wurde bei Herrn <Name> dokumentiert.",
    # 40.Fixierung  (Intent 39)
 "Frau <Name> wurde heute fixiert.",
  "Bitte dokumentiere die Fixierung bei Herrn <Name>.",
  "Heute fand eine Fixierung bei Frau <Name> statt.",
  "Herr <Name> benötigte heute eine Fixierung.",
  "Die Bewohnerin wurde heute sicher fixiert.",
  "Trage ein: Fixierung für Frau <Name> um 16 Uhr.",
  "Die Fixierung wurde heute erfolgreich durchgeführt.",
  "Bitte vermerke, dass die Fixierung heute stattgefunden hat.",
  "Frau <Name> wurde heute fixiert.",
  "Fixierung wurde bei Herrn <Name> dokumentiert.",
    # 41.Gewichtsauswertung  (Intent 40)
 "Bitte dokumentiere die Gewichtsauswertung von Frau <Name>.",
  "Heute wurde die Gewichtsauswertung bei Herrn <Name> durchgeführt.",
  "Frau <Name> hat heute ihre Gewichtsauswertung erhalten.",
  "Herr <Name> benötigt eine Gewichtsauswertung.",
  "Die Bewohnerin hat heute eine Gewichtsauswertung bekommen.",
  "Trage ein: Gewichtsauswertung für Frau <Name> um 10 Uhr.",
  "Die Gewichtsauswertung wurde heute erfolgreich abgeschlossen.",
  "Bitte vermerke, dass die Gewichtsauswertung heute stattgefunden hat.",
  "Frau <Name> hat heute ihre Gewichtsauswertung erhalten.",
  "Gewichtsauswertung wurde bei Herrn <Name> dokumentiert.",
    # 42.Medikamente vorbereiten bei Diabetikern  (Intent 41)
"Bitte bereite die Medikamente für den diabetischen Patienten vor.",
  "Heute wurden die Medikamente für Frau <Name>, die Diabetikerin ist, vorbereitet.",
  "Herr <Name> benötigt die Medikamentenvorbereitung wegen Diabetes.",
  "Die Medikamente für Frau <Name>, die Diabetikerin ist, wurden heute vorbereitet.",
  "Trage ein: Medikamente vorbereiten für den diabetischen Bewohner.",
  "Medikamentenvorbereitung bei Diabetikern wurde heute durchgeführt.",
  "Bitte dokumentiere die Medikamentenvorbereitung für die Diabetikerin Frau <Name>.",
  "Herr <Name> ist Diabetiker, Medikamente wurden vorbereitet.",
  "Die Medikamente für den Diabetiker wurden heute bereitgestellt.",
  "Medikamente für Frau <Name>, Diabetikerin, wurden heute vorbereitet.",
    # 43.Medikamente verabreichen(von PFK)  (Intent 42)
    "Die Pflegefachkraft hat heute die Medikamente verabreicht.",
  "Bitte dokumentiere, dass die PFK Medikamente bei Herrn <Name> verabreicht hat.",
  "Frau <Name> erhielt heute ihre Medikamente von der Pflegefachkraft.",
  "Medikamente wurden heute von der PFK an Frau <Name> verabreicht.",
  "Herr <Name> bekam heute seine Medikamente von der Pflegefachkraft.",
  "Trage ein: Medikamente verabreicht von der PFK bei Frau <Name>.",
  "Die PFK hat heute die Medikamentengabe bei Herrn <Name> durchgeführt.",
  "Medikamente wurden von der Pflegefachkraft erfolgreich verabreicht.",
  "Bitte vermerke, dass die PFK die Medikamente heute verabreicht hat.",
  "Frau <Name> bekam heute ihre Medikamente von der Pflegefachkraft."
]

labels_text = [
    # Blutzucker
    0,0,0,0,0,0,0,0,0,0,
    # Blutdruck
    1,1,1,1,1,1,1,1,1,1,
    # Temperatur (2)
    2,2,2,2,2,2,2,2,2,2,
    # Gewicht (3)
    3,3,3,3,3,3,3,3,3,3,
    # Medikamente verabreichen (4)
    4,4,4,4,4,4,4,4,4,4,
    # Ein- und Ausfuhr (5)
    5,5,5,5,5,5,5,5,5,5,
    # STUHLGANG
    6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
    # EINZELBETREUUNG
    7,7,7,7,7,7,7,7,7,7,
    # Verbandswechsel
    8,8,8,8,8,8,8,8,8,8,
    # Medikamente stellen
    9,9,9,9,9,9,9,9,9,9,9,9,9,
    # Nahrungsaufnahme
    10,10,10,10,10,10,10,10,10,10,
    # Flüssigkeitsaufnahme
    11,11,11,11,11,11,11,11,11,11,
    # Fixierung Bauchgurt an, im Bett
    12,12,12,12,12,12,12,12,12,12, 
    # Fixierung Bauchgurt im Bett entfernen
    13,13,13,13,13,13,13,13,13,13, 
    # GRUPPENBETREUUNG
    14,14,14,14,14,14,14,14,14,14,14, 
    # Tagesgruppe
    15,15,15,15,15,15,15,15,15,15, 
    # Medikamente vorbereiten
    16,16,16,16,16,16,16,16,16,16,16,16, 
    # Fixierung Bauchgurt im  Rollstuhl
    17,17,17,17,17,17,17,17,17,17,  
    # Fixierung Bauchgurt im Rollstuhl entfernen
    18,18,18,18,18,18,18,18,18,18,  
    # Fixierung Therapietisch
    19,19,19,19,19,19,19,19,19,19,  
    # Fixierung Therapietisch entfernen
    20,20,20,20,20,20,20,20,20,20,  
    # BETREUUNG NACH § 43b
    21,21,21,21,21,21,21,21,21,21,  
    # Behandlungspflege
    22,22,22,22,22,22,22,22,22,22,  
    # Medikamente bereit stellen
    23,23,23,23,23,23,23,23,23,23,  
    # Beratende Gespräche
    24,24,24,24,24,24,24,24,24,24,24,  
    # KG extern
    25,25,25,25,25,25,25,25,25,25,  
    # Mobilitätsfaktor
    26,26,26,26,26,26,26,26,26,26,  
    # Validierende Gespräche
    27,27,27,27,27,27,27,27,27,27,  
    # Lagerungsprotokoll
    28,28,28,28,28,28,28,28,28,28,  
    # Duschen
    29,29,29,29,29,29,29,29,29,29,  
    # Frisör
    30,30,30,30,30,30,30,30,30,30,30,  
    # Fußpflege
    31,31,31,31,31,31,31,31,31,31,  
    # Nagelpflege
    32,32,32,32,32,32,32,32,32,32,  
    # Haarwäsche
    33,33,33,33,33,33,33,33,33,33,  
    # Fixierung Rollstuhlbremse anlegen
    34,34,34,34,34,34,34,34,34,34,  
    # Fixierung Rollstuhlbremse entfernen
    35,35,35,35,35,35,35,35,35,35,  
    # Individueller Toillettengang
    36,36,36,36,36,36,36,36,36,36,  
    # Medikamentenkontrolle
    37,37,37,37,37,37,37,37,37,37,37,37,37,37,  
    # Betreuung / Beobachtung
    38,38,38,38,38,38,38,38,38,38,  
    # Fixierung
    39,39,39,39,39,39,39,39,39,39,  
    # Gewichtsauswertung
    40,40,40,40,40,40,40,40,40,40,  
    # Medikamente vorbeBew. ist diabetiker.reiten
    41,41,41,41,41,41,41,41,41,41,  
    # Medikamente verabreichen(von PFK)
    42,42,42,42,42,42,42,42,42,42
]
# Label encoding
label_map = {label: i for i, label in enumerate(sorted(set(labels_text)))}
labels = np.array([label_map[label] for label in labels_text])

# Tokenize sentences
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post')
vocab_size = len(tokenizer.word_index) + 1
num_classes = len(label_map)
print("📐 Padded shape:", padded.shape)
optimizer = Adam(learning_rate=0.005)
# -------------------------------
# ✅ Step 2: Model Architecture
# -------------------------------
def create_model():
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=2000, input_length=padded.shape[1]),
        Bidirectional(LSTM(64, return_sequences=True)),
        GlobalAveragePooling1D(),
        Dense(256, activation='relu'), 
        BatchNormalization(),
        Dropout(0.4),
        Dense(512, activation='relu'), 
        BatchNormalization(),
        Dropout(0.2),
        Dense(100, activation='relu'),  
        Dense(128, activation='relu'),
        Dropout(0.2), 
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# -------------------------------
# ✅ Step 3: DEAP Evolution Setup
# -------------------------------
def evaluate(individual):
    model = create_model()
    weights = model.get_weights()
    idx = 0
    new_weights = []
    for w in weights:
        shape = w.shape
        size = np.prod(shape)
        new_weights.append(np.array(individual[idx:idx + size]).reshape(shape))
        idx += size

    model.set_weights(new_weights)
    loss, acc = model.evaluate(padded, labels, verbose=0)
    return acc,

def get_model_size():
    model = create_model()
    
    model.build(input_shape=(None, padded.shape[1]))  # Force build
    return sum(np.prod(w.shape) for w in model.get_weights())

# Setup DEAP 
weight_size = get_model_size() 
print("🧠 Model genome size:", weight_size)
if weight_size < 2:
    raise ValueError(f"Model weight genome too small for crossover (size: {weight_size}). Increase model size.")
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -1.0, 1.0)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, weight_size)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# -------------------------------
# ✅ Step 4: Run the Evolution
# -------------------------------
def run_evolution():
    pop = toolbox.population(n=20)
    NGEN = 10  # For better results, use 50-100

    for gen in range(NGEN):
        print(f"🔄 Generation {gen}")
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.2)
        fits = list(map(toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit
        pop = toolbox.select(offspring, k=len(pop))

    best = tools.selBest(pop, k=1)[0]
    return best

# -------------------------------
# ✅ Step 5: Save Best Model
# -------------------------------
def save_best_model(best_genome):
    model = create_model()
    weights = model.get_weights()
    idx = 0
    new_weights = []
    for w in weights:
        shape = w.shape
        size = np.prod(shape)
        new_weights.append(np.array(best_genome[idx:idx + size]).reshape(shape))
        idx += size
    model.set_weights(new_weights)
    model.save("best_model.h5") 
    print("✅ Best model saved to 'best_model.h5'.")
import pickle 
with open('tokenizerupdate.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
# -------------------------------
# 🚀 Main Entry Point
# -------------------------------
if __name__ == "__main__":
    best = run_evolution()
    save_best_model(best)