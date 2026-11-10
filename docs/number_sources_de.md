Maps every number in the report to its source file under `runs/` (German).

# Zahlenherkunft der Projektarbeit

Diese Datei ordnet jeder Kennzahl des Berichts die Datei unter `runs/` zu, aus
der sie stammt. Sie dient der Nachprüfbarkeit: jede Zahl im Dokument und im
Notebook lässt sich damit ohne erneuten Trainingslauf bis zu den Rohdaten
zurückverfolgen.

Stand: 2026-08-10, nach vollständiger Nachrechnung aller Kennzahlen gegen `runs/`.

---

## F0 Rahmen

| Größe | Wert | Quelle |
|---|---|---|
| Kohorte gesamt | 30 virtuelle Patienten (je 10 Jugendliche, Erwachsene, Kinder) | `vpatient_params.csv` |
| Training | 20 Patienten (7 Jugendliche, 7 Erwachsene, 6 Kinder) | `runs/train/*/config.json` |
| Held-out | 10 Patienten (3 Jugendliche, 3 Erwachsene, 4 Kinder) | `runs/cohort_seed*/cohort_summary.json` |
| Episode | 1 simulierter Tag = 480 Schritte zu 3 min | `config.json` |
| Episoden je Zelle | 200 (10 Patienten x 20 Episoden) | `cohort_summary.json` |
| Auswertungs-Seeds | 100000 bis 100019 | `cohort_summary.json` |
| Trainings-Seeds | `[0, 100000)`, per `assert` erzwungen | `src/t1d_env.py` |
| Trainingsläufe | 44 = 2 Startwerte x 22 (20 personalisiert + 2 generalistisch) | `runs/*/2026*/config.json` |
| Schrittbudget | generalistisch 600000, personalisiert 250000 | `config.json` |
| Basalraten der Kohorte | 0,00481 bis 0,02874 U/min, **Faktor 6,0** | `vpatient_params.csv` |


---

## F1 Beobachtbarkeit: der physikalische Unterschied

Gemessen an `adult#001` unter konstanter Basalrate über **20 Episoden**
(Seeds 100000 bis 100019), nicht über eine einzelne.

| Größe | Wert | Quelle |
|---|---|---|
| Verzögerung subkutan gegen Plasma | **12 min** (4 Schritte) in **20 von 20** Episoden | Notebook Zelle 6 |
| Sensorrauschen, SD von CGM minus subkutan | **11,04 +/- 2,32 mg/dL** (Spanne 6,85 bis 16,37) | Notebook Zelle 6 |
| Mittlerer Betragsfehler CGM gegen Plasma | **10,01 +/- 1,64 mg/dL** (Spanne 7,67 bis 13,74) | Notebook Zelle 6 |


---

## F2 Aktionsraum, aus der klinischen Referenz hergeleitet

| Patient | Basalrate | 99. Perzentil | Maximum | Max/Basal |
|---|---|---|---|---|
| adult#001 | 0,02112 | 0,0211 | 2,6211 | 124,1 |
| adolescent#001 | 0,01394 | 0,0139 | 2,1806 | 156,5 |
| child#001 | 0,00658 | 0,0066 | 1,0466 | 159,1 |

Alle Angaben in U/min. Gewählt: `u = basal_ss * 160^a`, `a` in `[-1, 1]`.

---

## F3 Exploration, erwartete bolusgrosse Aktionen je Episode

| log_std | sigma | Abstand zu a = 0,897 | Treffer je Episode |
|---|---|---|---|
| -1,5 | 0,223 | 4,02 sigma | 0,014 |
| -1,0 | 0,368 | 2,44 sigma | 3,54 |
| -0,8 | 0,449 | 2,00 sigma | 11,02 |
| 0,0 | 1,000 | 0,90 sigma | 88,73 |

Gemessene Folgen, Quelle `runs/_negative_logstd15/`, `runs/_negative_logstd10/`:

| log_std | Schritte | Held-out-TIR | Insulin/Tag | Aktionsmittel | Aktions-SD | bolusgrosse Aktionen |
|---|---|---|---|---|---|---|
| -1,5 | 50000 | 20,33 % | 23,73 U | -0,049 | 0,008 | **0 von 9600** |
| -1,5 | 100000 | **69,46 %** | 41,64 U | +0,051 | 0,065 | **0 von 9600** |
| -1,0 | -- | nicht erreichbar | -- | -- | -- | 97 % Abbrüche |


---

## F4 Imitationsgüte (Behavioral Cloning)

| Schüler | ein Patient, MSE | ein Patient, r | 20 Patienten, MSE | 20 Patienten, r |
|---|---|---|---|---|
| privilegiert | 0,00008 | **0,995** | 0,00012 | **0,991** |
| nur CGM | 0,00589 | **0,091** | 0,00557 | **0,100** |

Quelle: `runs/pretrained/bc_*_info.json`. Lehrer-TIR 87,36 % (adult#001, 150
Episoden, 72000 Paare) und 84,46 % (20 Patienten, 300 Episoden, 143836 Paare).

---

## F5 Referenzregler auf den 10 Held-out-Patienten, 200 Episoden je Regler

Quelle: `runs/baselines/summary_heldout_heldout.json`. Am 2026-08-09 exakt
reproduziert.

| Regler | TIR | TBR ges. | TBR L2 | TAR ges. | TAR L2 | CV | mittl. BG | Insulin | Abbrüche | über 70 % | Ziele |
|---|---|---|---|---|---|---|---|---|---|---|---|
| konstante Basalrate | 36,6 % | 0,0 % | 0,00 % | 63,4 % | 41,4 % | 36,08 % | 252 | 15,5 U | 37/200 | 9/200 | **2/6** |
| Basal-Bolus (klinisch) | 82,4 % | **5,6 %** | **1,51 %** | 12,0 % | 4,2 % | 29,89 % | 135 | 40,3 U | 7/200 | 174/200 | **4/6** |


Nach Gruppe:

| Gruppe | konst. Basalrate TIR | Basal-Bolus TIR | Basal-Bolus TBR |
|---|---|---|---|
| Jugendliche | 34,7 % | 88,1 % | 0,29 % |
| Erwachsene | 33,3 % | 90,3 % | 2,85 % |
| **Kinder** | 40,5 % | **72,3 %** | **11,73 %** |

Befund: die klinische Referenztherapie verfehlt auf der Kohorte beide
Hypoglykämie-Ziele, praktisch vollständig getragen von den Kindern.

---

## F6 Das 2x2, Time in Range auf den 10 Held-out-Patienten

Quelle: `runs/cohort_seed0/cohort_summary.json`, `runs/cohort_seed1/cohort_summary.json`.

| Zelle | Lauf 1 | Lauf 2 | Mittel | Spanne |
|---|---|---|---|---|
| generalistisch, privilegiert | 71,0 % | 73,3 % | 72,1 % | 2,3 |
| generalistisch, nur CGM | 42,4 % | 37,4 % | 39,9 % | 5,0 |
| personalisiert, privilegiert | 74,8 % | 72,4 % | 73,6 % | 2,5 |
| personalisiert, nur CGM | 36,7 % | 38,3 % | 37,5 % | 1,6 |

| Effekt | Einzelmessungen | Mittel | Spanne |
|---|---|---|---|
| **Informationslücke** | +28,6, +34,1, +35,9, +38,1 | **34,2 +/- 4,1** | 28,6 bis 38,1 |
| **Personalisierung** | -5,7, -0,9, +0,9, +3,9 | **-0,5 +/- 4,0** | -5,7 bis +3,9 |

Streuung: die Standardabweichungen sind Stichproben-SD (`ddof=1`) über die vier
Messungen. Die größte Streuung einer einzelnen Zelle zwischen den Startwerten
beträgt 5,0 Punkte und ist damit so gross wie der gesamte gesuchte
Personalisierungseffekt.

### Sicherheit und Abbrüche je Zelle

| Zelle | TBR Lauf 1 | TBR Lauf 2 | TBR L2 Lauf 1 | TBR L2 Lauf 2 | Abbrüche |
|---|---|---|---|---|---|
| generalistisch, privilegiert | 1,02 % | 1,74 % | 0,00 % | 0,12 % | 12/200, 9/200 |
| generalistisch, nur CGM | 0,00 % | 0,00 % | 0,00 % | 0,00 % | 30/200, 36/200 |
| personalisiert, privilegiert | 0,36 % | 2,31 % | 0,00 % | **1,71 %** | 13/200, 20/200 |
| personalisiert, nur CGM | 0,00 % | 0,00 % | 0,00 % | 0,00 % | 39/200, 36/200 |


1. Die privilegierten Regler liegen bei **0,4 bis 2,3 % TBR**, nicht bei
   "0,1 bis 0,2 %". Der alte Wert stammte aus `runs/_invalid_cohort/`, dem
   seed-kontaminierten Lauf. Die qualitative Aussage bleibt richtig: alle vier
   Zellen liegen unter den 5,6 % der klinischen Referenz.
2. Ein Lauf (personalisiert, privilegiert, Startwert 1) verfehlt mit
   **1,71 % TBR Level 2** das Konsensziel von unter 1 %. Die gelernten Regler
   sind also **nicht** durchgehend sicherer als die Konsensgrenze, nur sicherer
   als die klinische Referenz.
3. Die Abbruchrate der CGM-Regler liegt bei **30 bis 39 von 200**, nicht bei
   "36 bis 39". Vergleichswert ohne Regelung: 37 von 200.

### Nach Patientengruppe, gemittelt über beide Durchläufe

| Regler | Jugendliche | Erwachsene | Kinder |
|---|---|---|---|
| Basal-Bolus (klinisch) | 88,1 % | 90,3 % | 72,3 % |
| privilegiert (beide Varianten) | **75,8 %** | **70,4 %** | **72,6 %** |
| nur CGM (beide Varianten) | **37,0 %** | **35,7 %** | **42,3 %** |


---

## F6b Ablation: Fenstergröße des CGM-Agenten

Quelle: `runs/ablation_cgm_w60/`, `runs/ablation_w60_seed0/`, `runs/ablation_w60_seed1/`.
Gelaufen am 2026-08-09.

**Frage.** Der CGM-Agent sah 12 Schritte, also 36 Minuten Sensorverlauf. Bleibt
die Lücke bestehen, wenn er 60 Schritte, also **3 Stunden** sieht?

**Ein-Faktor-Versuch.** Verändert wurde ausschliesslich `cgm_hist_len`. Die
Insulinhistorie (12 Schritte), das Ankündigungsfenster und der privilegierte
Beobachtungsraum bleiben unverändert. Der privilegierte Arm musste deshalb
nicht neu trainiert werden. Beobachtungsdimension des CGM-Agenten: 26 gegen 74.
Gleiches Netz (64x64), gleiches Schrittbudget (600000), gleiche Seeds, gleiche
zehn Held-out-Patienten.

### Imitationsgüte, identische Lehrerdaten (143836 Paare, Lehrer-TIR 84,5 %)

| Fenster | Beob.-Dim. | Val-MSE | r zum Lehrer |
|---|---|---|---|
| 36 min | 26 | 0,005572 | 0,0995 |
| **180 min** | 74 | **0,005572** | **0,1030** |
| privilegiert | 31 | 0,000125 | 0,9911 |

Das Fünffache an Sensorverlauf ändert die Vorhersagbarkeit der Lehreraktion
nicht: der Fehler ist auf sechs Nachkommastellen identisch.

### Regelgüte auf den zehn Held-out-Patienten, 200 Episoden je Zelle

| Zelle | Lauf 1 | Lauf 2 | Mittel | Spanne | Abbrüche | mittl. BG |
|---|---|---|---|---|---|---|
| CGM 36 min | 42,4 % | 37,4 % | **39,9 %** | 5,0 | 66/400 | 242 |
| CGM 180 min | 37,7 % | 39,5 % | **38,6 %** | 1,8 | 63/400 | 243 |
| privilegiert (unverändert) | 71,0 % | 73,3 % | **72,1 %** | 2,3 | 21/400 | 167 |

**Effekt der Fensterverlängerung: -1,3 Punkte.** Das liegt innerhalb der
Streuung zwischen Zufallsstartwerten: allein die 36-min-Zelle schwankt um 5,0
Punkte zwischen ihren beiden Läufen.

**Informationslücke generalistisch:** 32,2 Punkte bei 36 min, **33,6 Punkte bei
180 min**. Sie wird durch das längere Fenster nicht kleiner.

**Aussage für den Bericht.** Ein fünffach längeres Sensorfenster schliesst die
Lücke nicht, weder in der Imitation noch nach 600000 PPO-Schritten. Die Lücke
ist damit keine Frage der Fensterlänge. Zulässig ist die Aussage "unter
gleicher Netzkapazität und gleichem Budget bringt ein längeres Fenster keinen
messbaren Gewinn", **nicht** "längere Fenster sind schlechter": -1,3 Punkte sind
von null nicht zu unterscheiden.

**Was der Versuch nicht abdeckt.** Ein rekurrentes Netz, eine größere
Netzkapazität passend zu den 74 Eingangsdimensionen, und eine explizite
Mahlzeitenerkennung aus dem Anstiegsverhalten. Die Grenzen-Aussage in der
Diskussion bleibt also bestehen, sie wird nur um eine Dimension enger.

---

## F7 Einordnung gegen die klinische Referenz

| Regler | TIR auf 10 Held-out-Patienten |
|---|---|
| Basal-Bolus (klinisch) | **82,4 %** |
| personalisiert, privilegiert | 73,6 % |
| generalistisch, privilegiert | 72,1 % |
| personalisiert, nur CGM | 37,5 % |
| generalistisch, nur CGM | 39,9 % |
| konstante Basalrate | 36,6 % |

Auf der Kohorte schlägt **kein** gelernter Regler die Standardtherapie. Der
Vorversuch auf `adult#001` hatte das Gegenteil ergeben (94,5 % gegen 90,9 %);
dieser Vorsprung war eine Eigenschaft des Patienten, nicht des Verfahrens.

Der Vergleich ist nicht eindeutig: die klinische Therapie hat die höhere Time in
Range, die privilegierten RL-Regler das geringere Unterzuckerungsrisiko
(0,4 bis 2,3 % gegen 5,6 % TBR).

---

## F8 Morris-Screening (SALib, 10 Parameter, +/- 20 %, QoI = TIR)

Quelle: `runs/sensitivity/morris_bb_adult001.json`. 12 Trajektorien, 132
Parametersätze, 2 Szenarien, Regler = Basal-Bolus.

| Parameter | mu* | sigma | Bedeutung |
|---|---|---|---|
| kp1 | **33,02** | 34,43 | Basisrate der endogenen Glukoseproduktion |
| kp3 | 16,30 | 17,91 | Insulinwirkung auf die Glukoseproduktion |
| Km0 | 14,00 | 14,99 | Michaelis-Konstante der Glukoseaufnahme |
| Vmx | 9,56 | 9,55 | insulinabhängige Glukoseaufnahme |
| kmax | 7,01 | 3,51 | maximale Magenentleerungsrate |
| kd | 2,07 | 1,94 | Dissoziation des subkutanen Insulins |
| ka1 | 0,78 | 0,80 | Resorption des subkutanen Insulins |
| kabs | 0,60 | 0,60 | Resorptionsrate aus dem Darm |
| ksc | 0,59 | 0,47 | Diffusion Plasma zu subkutaner Glukose |
| u2ss | **0,00** | 0,00 | stationärer Insulinbedarf |

TIR über alle Parametersätze: **47,0 bis 96,1 %**, Median 75,4 %, bei
unverändertem Regler. kp1 wirkt genau doppelt so stark wie kp3 (33,02 / 16,30 =
2,03).

---

## F9 Laufzeit

| Größe | vorher | nachher | Faktor |
|---|---|---|---|
| Durchsatz einzelne Umgebung | 136,3 Schritte/s | 738,4 Schritte/s | **5,42** |
| Durchsatz 24 Umgebungen | 586/s | 3623/s | 6,18 |
| Reset | 16,4 ms | 3,99 ms | 4,1 |

Verifikation der `_FastParams`-Optimierung: BG-Verläufe bitidentisch,
maximale Abweichung 0,000e+00 mg/dL, `np.array_equal` liefert True.

---

## F10 Verworfene Läufe und Korrekturen

| Verzeichnis | Befund |
|---|---|
| `runs/_negative_logstd15` | Exploration zu eng, Boli werden nie entdeckt |
| `runs/_negative_logstd10` | Boli entdeckt, aber 97 % Abbrüche |
| `runs/_reward_v1` | Belohnung v1 treibt in die Unterzuckerung (TBR bis 12,70 %) |
| `runs/_invalid_seedleak` | Trainings- und Auswertungs-Seeds überlappten |
| `runs/_outdated_before_resetfix` | adult#001-Referenz von vor der Reset-Optimierung |

Seed-Leck: bei 24 Prozessen reichten die Trainings-Seeds bis 239999, der
Auswertungsbereich beginnt bei 100000. Personalisierte Regler trainierten auf
**19 von 20** Auswertungsepisoden. Der scheinbare Personalisierungsgewinn von
+5,4 Punkten verschwand nach der Korrektur vollständig.

Belohnung v1: Kosten je Schritt bei BG 60 betragen 1,36, bei BG 250 dagegen
2,24. Milde Unterzuckerung war billiger als mäßige Ueberzuckerung. Unter v2
sind es 2,79 gegen 2,24.

---
