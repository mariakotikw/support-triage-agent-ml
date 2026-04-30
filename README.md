Track: B

# Support Triage Agent

AI-агент для первичной сортировки клиентских обращений.

Агент принимает текст обращения, вызывает локальную sklearn-модель как инструмент, определяет категорию и срочность обращения, а затем формирует объяснение и черновик ответа клиенту.

## Почему это Track B

Проект использует классическую ML-модель как tool.

В проекте есть:

- `train.py` — скрипт обучения модели;
- `data/tickets.csv` — датасет обращений;
- `models/ticket_classifier.joblib` — сохранённая sklearn-модель;
- `models/metrics.json` — метрики качества;
- `agent.py` — агент, который вызывает модель как инструмент.

## Классы

Категории:

- `bug`
- `billing`
- `access`
- `feature`
- `question`

Срочность:

- `low`
- `medium`
- `high`

## Архитектура

```text
Пользовательское обращение
→ tool call: classify_ticket
→ sklearn TF-IDF + LogisticRegression
→ category + urgency + probabilities
→ финальный ответ агента