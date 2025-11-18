## Preprocessing
In diesem Ordner wird das Preprocessing der Handbücher behandelt.
Die Verarbeitung der roh-Daten stellt einen, wenn nicht sogar den wichtigsten Schritt der Pipeline dar.
Denn die Input Daten entscheiden über die Performance des gesamten Systems.

Wir hatten hier viele HTML Dokumente mit Listen, Bildern und Icons.
Man will die Daten in Text-Form, deswegen convertieren wir zu Markdown.
Markdown hat den Vorteil, dass HTML Struktur nicht ganz verloren geht.
Basierend auf Überschriften, splitten wir die ca.80 Dokumente in ca. 550 Text Bausteine.

Mit diesen können wir:
* Die Vector-Datenbank erstellen (IR)
* Ein synthetisches Dataset generieren