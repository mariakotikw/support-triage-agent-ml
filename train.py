from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


DATA_DIR = Path("data")
MODEL_DIR = Path("models")

DATA_PATH = DATA_DIR / "tickets.csv"
MODEL_PATH = MODEL_DIR / "ticket_classifier.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"


def create_seed_dataset() -> pd.DataFrame:

    rows = [
        # BUG
        {
            "text": "После обновления приложение вылетает при запуске.",
            "category": "bug",
            "urgency": "high",
        },
        {
            "text": "Кнопка оплатить не нажимается, заказ оформить невозможно.",
            "category": "bug",
            "urgency": "high",
        },
        {
            "text": "При открытии отчёта появляется ошибка 500.",
            "category": "bug",
            "urgency": "high",
        },
        {
            "text": "Поиск не находит товары, хотя они есть в каталоге.",
            "category": "bug",
            "urgency": "medium",
        },
        {
            "text": "Уведомления приходят два раза подряд.",
            "category": "bug",
            "urgency": "medium",
        },
        {
            "text": "Темная тема сбрасывается после перезапуска.",
            "category": "bug",
            "urgency": "low",
        },
        {
            "text": "В профиле не сохраняется новая фотография.",
            "category": "bug",
            "urgency": "medium",
        },
        {
            "text": "Страница настроек долго грузится и иногда зависает.",
            "category": "bug",
            "urgency": "medium",
        },

        # BILLING
        {
            "text": "С карты списали деньги два раза за одну подписку.",
            "category": "billing",
            "urgency": "high",
        },
        {
            "text": "Не пришёл чек после оплаты тарифа.",
            "category": "billing",
            "urgency": "medium",
        },
        {
            "text": "Хочу вернуть деньги за случайно оформленную подписку.",
            "category": "billing",
            "urgency": "high",
        },
        {
            "text": "Где скачать закрывающие документы для бухгалтерии?",
            "category": "billing",
            "urgency": "medium",
        },
        {
            "text": "Почему стоимость тарифа изменилась в этом месяце?",
            "category": "billing",
            "urgency": "medium",
        },
        {
            "text": "Как изменить способ оплаты?",
            "category": "billing",
            "urgency": "low",
        },
        {
            "text": "Не получается оплатить корпоративный тариф.",
            "category": "billing",
            "urgency": "high",
        },
        {
            "text": "Нужно добавить реквизиты компании в счёт.",
            "category": "billing",
            "urgency": "medium",
        },

        # ACCESS
        {
            "text": "Не могу войти в аккаунт, пароль не подходит.",
            "category": "access",
            "urgency": "high",
        },
        {
            "text": "Не приходит код двухфакторной аутентификации.",
            "category": "access",
            "urgency": "high",
        },
        {
            "text": "Сотрудник уволился, нужно заблокировать ему доступ.",
            "category": "access",
            "urgency": "high",
        },
        {
            "text": "Как восстановить доступ к старому аккаунту?",
            "category": "access",
            "urgency": "medium",
        },
        {
            "text": "Не могу пригласить нового участника в команду.",
            "category": "access",
            "urgency": "medium",
        },
        {
            "text": "Хочу изменить email для входа.",
            "category": "access",
            "urgency": "low",
        },
        {
            "text": "Пользователь не видит проект после приглашения.",
            "category": "access",
            "urgency": "medium",
        },
        {
            "text": "Нужно выдать админские права коллеге.",
            "category": "access",
            "urgency": "medium",
        },

        # FEATURE
        {
            "text": "Добавьте экспорт отчётов в PDF.",
            "category": "feature",
            "urgency": "low",
        },
        {
            "text": "Было бы удобно подключить интеграцию с Telegram.",
            "category": "feature",
            "urgency": "low",
        },
        {
            "text": "Нужна возможность создавать шаблоны ответов.",
            "category": "feature",
            "urgency": "low",
        },
        {
            "text": "Хотим видеть аналитику по активности команды.",
            "category": "feature",
            "urgency": "medium",
        },
        {
            "text": "Можно ли добавить массовое редактирование задач?",
            "category": "feature",
            "urgency": "medium",
        },
        {
            "text": "Нужен API для выгрузки данных в нашу CRM.",
            "category": "feature",
            "urgency": "medium",
        },
        {
            "text": "Добавьте сортировку проектов по дате изменения.",
            "category": "feature",
            "urgency": "low",
        },
        {
            "text": "Хотелось бы настроить собственные статусы задач.",
            "category": "feature",
            "urgency": "low",
        },

        # QUESTION
        {
            "text": "Как подключить интеграцию с календарём?",
            "category": "question",
            "urgency": "low",
        },
        {
            "text": "Где посмотреть историю изменений проекта?",
            "category": "question",
            "urgency": "low",
        },
        {
            "text": "Можно ли использовать сервис на нескольких устройствах?",
            "category": "question",
            "urgency": "low",
        },
        {
            "text": "Как удалить лишний проект?",
            "category": "question",
            "urgency": "low",
        },
        {
            "text": "Подскажите, где найти документацию по API.",
            "category": "question",
            "urgency": "low",
        },
        {
            "text": "Как изменить язык интерфейса?",
            "category": "question",
            "urgency": "low",
        },
        {
            "text": "Есть ли мобильное приложение?",
            "category": "question",
            "urgency": "low",
        },
        {
            "text": "Как перенести данные из другого сервиса?",
            "category": "question",
            "urgency": "medium",
        },
    ]

    return pd.DataFrame(rows)


def train_models():

    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    if not DATA_PATH.exists():
        df = create_seed_dataset()
        df.to_csv(DATA_PATH, index=False)
        print(f"Dataset created: {DATA_PATH}")
    else:
        df = pd.read_csv(DATA_PATH)
        print(f"Dataset loaded: {DATA_PATH}")

    print(f"Dataset size: {len(df)} rows")

    train_df, test_df = train_test_split(
        df,
        test_size=0.25,
        random_state=42,
        stratify=df["category"],
    )

    category_model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    urgency_model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    print("Training category model...")
    category_model.fit(train_df["text"], train_df["category"])

    print("Training urgency model...")
    urgency_model.fit(train_df["text"], train_df["urgency"])

    category_pred = category_model.predict(test_df["text"])
    urgency_pred = urgency_model.predict(test_df["text"])

    metrics = {
        "category_accuracy": accuracy_score(test_df["category"], category_pred),
        "category_f1_macro": f1_score(
            test_df["category"],
            category_pred,
            average="macro",
            zero_division=0,
        ),
        "urgency_accuracy": accuracy_score(test_df["urgency"], urgency_pred),
        "urgency_f1_macro": f1_score(
            test_df["urgency"],
            urgency_pred,
            average="macro",
            zero_division=0,
        ),
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "category_report": classification_report(
            test_df["category"],
            category_pred,
            output_dict=True,
            zero_division=0,
        ),
        "urgency_report": classification_report(
            test_df["urgency"],
            urgency_pred,
            output_dict=True,
            zero_division=0,
        ),
    }

    bundle = {
        "category_model": category_model,
        "urgency_model": urgency_model,
        "metrics": metrics,
    }

    joblib.dump(bundle, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    print()
    print("Training finished.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print()
    print(f"Category accuracy: {metrics['category_accuracy']:.3f}")
    print(f"Category F1 macro: {metrics['category_f1_macro']:.3f}")
    print(f"Urgency accuracy: {metrics['urgency_accuracy']:.3f}")
    print(f"Urgency F1 macro: {metrics['urgency_f1_macro']:.3f}")


if __name__ == "__main__":
    train_models()