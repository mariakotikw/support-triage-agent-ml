import argparse
import json
import os
from pathlib import Path
from typing import Any

import joblib
import requests
from dotenv import load_dotenv

from train import train_models


load_dotenv()

MODEL_PATH = Path("models/ticket_classifier.joblib")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")


CATEGORY_TO_TEAM = {
    "bug": "Engineering / Bug fixing",
    "billing": "Billing support",
    "access": "Account & Access support",
    "feature": "Product team",
    "question": "Customer support",
}


URGENCY_HINTS = {
    "high": "нужно обработать как можно быстрее",
    "medium": "можно обработать в обычной очереди, но не откладывать",
    "low": "низкий приоритет, можно обработать планово",
}


def ensure_model_exists() -> None:
    """
    Проверяет, есть ли обученная sklearn-модель.
    Если модели нет, автоматически запускает обучение.
    """
    if not MODEL_PATH.exists():
        print("[setup] Модель не найдена. Запускаю обучение...")
        train_models()


def load_model_bundle() -> dict[str, Any]:
    """
    Загружает сохранённый bundle с моделями.
    """
    ensure_model_exists()
    return joblib.load(MODEL_PATH)


MODEL_BUNDLE = load_model_bundle()
CATEGORY_MODEL = MODEL_BUNDLE["category_model"]
URGENCY_MODEL = MODEL_BUNDLE["urgency_model"]


def top_probabilities(model: Any, text: str, top_k: int = 3) -> list[dict[str, Any]]:
    """
    Возвращает top-k вероятностей классов.
    """
    probabilities = model.predict_proba([text])[0]
    classes = model.classes_

    ranked = sorted(
        zip(classes, probabilities),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        {
            "label": str(label),
            "probability": round(float(probability), 3),
        }
        for label, probability in ranked[:top_k]
    ]


def classify_ticket(text: str) -> dict[str, Any]:
    """
    ML tool.

    Использует локальные sklearn-модели:
    - category_model для категории обращения;
    - urgency_model для срочности.
    """
    category = CATEGORY_MODEL.predict([text])[0]
    urgency = URGENCY_MODEL.predict([text])[0]

    return {
        "input_text": text,
        "category": str(category),
        "urgency": str(urgency),
        "recommended_team": CATEGORY_TO_TEAM.get(str(category), "Customer support"),
        "urgency_hint": URGENCY_HINTS.get(str(urgency), "приоритет не определён"),
        "category_probabilities": top_probabilities(CATEGORY_MODEL, text),
        "urgency_probabilities": top_probabilities(URGENCY_MODEL, text),
    }


def build_template_answer(tool_result: dict[str, Any]) -> str:
    """
    Fallback-ответ без LLM.
    Используется в режиме --local.
    """
    category = tool_result["category"]
    urgency = tool_result["urgency"]
    team = tool_result["recommended_team"]
    hint = tool_result["urgency_hint"]

    category_probs = ", ".join(
        f"{item['label']}: {item['probability']}"
        for item in tool_result["category_probabilities"]
    )

    urgency_probs = ", ".join(
        f"{item['label']}: {item['probability']}"
        for item in tool_result["urgency_probabilities"]
    )

    return f"""Категория: {category}
Срочность: {urgency}
Куда передать: {team}

Почему агент так решил:
Локальная sklearn-модель классифицировала обращение как {category} с приоритетом {urgency}.
Подсказка по срочности: {hint}.
Вероятности категорий: {category_probs}.
Вероятности срочности: {urgency_probs}.

Черновик ответа клиенту:
Здравствуйте! Спасибо, что обратились. Мы зафиксировали ваше обращение и передали его в ответственную команду: {team}. Специалисты проверят ситуацию и вернутся с ответом. Приоритет обращения: {urgency}.
"""


def ask_ollama(ticket_text: str, tool_result: dict[str, Any]) -> str:
    """
    Вызывает локальную LLM через Ollama.

    Важно:
    sklearn-tool уже был вызван до этого.
    Ollama не классифицирует обращение сама, а только формирует финальный ответ.
    """
    system_prompt = """
Ты аккуратный русскоязычный support-агент.

Тебе уже передали результат локальной sklearn-модели.
Не меняй категорию, срочность и команду.
Не выдумывай вероятности.
Используй только tool_result как источник классификации.

Пиши только на русском языке.
Не используй английские, японские или другие иностранные слова.
Не используй многоточия как заполнители.
Не пиши "дорогой клиент".
Пиши профессионально, кратко и спокойно.

Ответь строго в таком формате:

Категория: <category>
Срочность: <urgency>
Куда передать: <recommended_team>

Почему агент так решил:
<1-2 предложения>

Черновик ответа клиенту:
<2-3 вежливых предложения>
"""

    user_prompt = f"""
Текст обращения:
{ticket_text}

Результат ML tool classify_ticket:
{json.dumps(tool_result, ensure_ascii=False, indent=2)}

Используй эти значения буквально:
category = {tool_result["category"]}
urgency = {tool_result["urgency"]}
recommended_team = {tool_result["recommended_team"]}
"""

    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120,
    )

    response.raise_for_status()
    data = response.json()

    return data["message"]["content"]


def run_agent(ticket_text: str, local_only: bool = False) -> str:
    """
    Основной цикл агента.

    1. Вызывает ML tool classify_ticket.
    2. Печатает tool call и tool result.
    3. Если local_only=True, возвращает шаблонный ответ.
    4. Иначе пытается вызвать Ollama.
    5. Если Ollama не отвечает, использует fallback.
    """
    print()
    print(f"[tool call] classify_ticket({{'text': {repr(ticket_text)}}})")

    tool_result = classify_ticket(ticket_text)

    print("[tool result]")
    print(json.dumps(tool_result, ensure_ascii=False, indent=2))
    print()

    if local_only:
        return build_template_answer(tool_result)

    try:
        return ask_ollama(ticket_text, tool_result)
    except Exception as error:
        print("[warning] Ollama не ответила, использую локальный fallback.")
        print(f"[warning] {error}")
        print()
        return build_template_answer(tool_result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local",
        action="store_true",
        help="Запустить без Ollama, только sklearn tool + шаблонный ответ.",
    )
    args = parser.parse_args()

    print("Support Triage Agent")
    print("Введите клиентское обращение.")
    print("Для выхода напишите: exit")

    if args.local:
        print("Режим: local fallback без LLM")
    else:
        print(f"Режим: sklearn tool + Ollama model '{OLLAMA_MODEL}'")

    print()

    while True:
        ticket_text = input("Клиент: ").strip()

        if ticket_text.lower() in {"exit", "quit", "выход"}:
            print("Завершение работы.")
            break

        if not ticket_text:
            continue

        answer = run_agent(ticket_text, local_only=args.local)

        print("Агент:")
        print(answer)
        print()


if __name__ == "__main__":
    main()