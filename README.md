# Klassifizierung von Fake News im Zusammenhang mit Wahlen
## Contributors
- Karina Krebs
- Sophie Blum
- Leon Randzio
- Ingo Weber
- Tjark Gerken

## Hintergrund und Relevanz:
Fake News haben in den letzten Jahren einen signifikanten Einfluss auf politische Prozesse, insbesondere auf Wahlen, gewonnen. Sie tragen dazu bei, öffentliche Meinungen zu manipulieren, Vertrauen in demokratische Institutionen zu untergraben und Polarisierung in der Gesellschaft zu verstärken. Die Identifikation und Klassifizierung solcher Fehlinformationen ist daher ein zentraler Forschungsbereich. Mit der Entwicklung leistungsfähiger Sprachmodelle (LLMs, Large Language Models) wie GPT, BERT oder T5 eröffnen sich neue Möglichkeiten, Fake News automatisiert zu erkennen und einzuordnen. Dieses Projekt zielt darauf ab, die Leistungsfähigkeit verschiedener LLMs bei der Klassifizierung von Fake News zu evaluieren und das Modell mit der höchsten Präzision zu identifizieren.

## Zielsetzung:
Das primäre Ziel dieses Projekts ist es, die Effektivität und Effizienz verschiedener LLMs bei der Klassifizierung von Fake News im Zusammenhang mit Wahlen zu untersuchen. Dabei soll insbesondere analysiert werden, wie gut die Modelle zwischen echten und gefälschten Nachrichten unterscheiden können und welche Herausforderungen bei der Klassifizierung auftreten.

## Methodik:
Datensammlung:
Für dieses Projekt wird der bereits existierende Datensatz Fake News Elections Labelled Data (https://huggingface.co/datasets/newsmediabias/fake_news_elections_labelled_data) verwendet. Dieser Datensatz enthält Nachrichten im Zusammenhang mit Wahlen, die in die Kategorien „real“ und „fake“ unterteilt sind. Er bietet eine solide Grundlage, um LLMs für die Fake-News-Klassifikation zu trainieren und zu evaluieren.

## Modellauswahl:
Es werden verschiedene LLMs ausgewählt, darunter Modelle mit unterschiedlichen Architekturen (z. B. Transformer-basierte Modelle wie GPT-4, BERT, RoBERTa und T5). Die Modelle werden für die binäre Klassifikation (Fake/Real) feinabgestimmt (Fine-Tuning).

| Person | Model            |
|--------|------------------|
| Karina | T5               |
| Sophie | Log. Reg. & ElMo |
| Leon   | BeRT             |
| Ingo   | Llama            |
| Tjark  | SVM, RGR         |

## Evaluierung der Modelle:
Die Modelle werden anhand verschiedener Metriken wie der Genauigkeit evaluiert.

## Vergleich und Analyse:
Nach der Evaluierung wird eine vergleichende Analyse durchgeführt, um festzustellen, welches Modell die beste Performance aufweist.

## Erwartete Ergebnisse:
Ein umfassender Vergleich der Leistungsfähigkeit verschiedener LLMs bei der Klassifizierung von Fake News.
Identifikation des Modells mit der besten Präzision und Robustheit.
Erkenntnisse über typische Fehler und Schwächen der Modelle.
Empfehlungen zur Weiterentwicklung von Modellen für die Fake-News-Erkennung.




