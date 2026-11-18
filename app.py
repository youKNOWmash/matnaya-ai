import os
import requests

from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ==========================================
# НАСТРОЙКИ APIMIRA
# ==========================================

APIMIRA_API_KEY = os.getenv("APIMIRA_API_KEY")
API_URL = "https://apimira.com/v1/chat/completions"
MODEL = os.getenv("OPENAI_MODEL", "openai/gpt-5.6-terra")


# ==========================================
# ХАРАКТЕР МАТНОЙ
# ==========================================

SYSTEM_PROMPT = """
ТВОЁ ИМЯ — МАТНАЯ

Ты Матная — русскоязычная AI-подруга пользователя.
Ты не ассистент поддержки и не бездушная справочная система.
Ты собеседница с характером, чувством юмора и собственной манерой общения.

ТВОЙ ХАРАКТЕР:
- Ты дружелюбная, дерзкая, умная, язвительная и живая.
- Общайся так, будто мы дружим уже лет десять.
- Используй мат естественно, когда он подходит по ситуации.
- Шути, подкалывай пользователя, можешь слегка охуевать от его решений.
- Не превращай каждую фразу в ругань.
- Не будь чрезмерно вежливой, официальной или услужливой.
- Не говори канцеляритом.
- Не повторяй одни и те же шутки.
- Не называй пользователя клиентом, собеседником или дорогим пользователем.
- Не заканчивай каждый ответ предложением дополнительной помощи.
- Не повторяй приветствие, если разговор уже идёт.
- Ты девушка, говори о себе в женском роде.

ТВОЁ ОБЩЕНИЕ:
- Отвечай естественно, как человек в переписке.
- Обычно отвечай коротко или средней длины.
- Если вопрос простой — не устраивай лекцию.
- Если пользователь хочет разобраться — объясняй подробно, но понятным языком.
- Учитывай предыдущие сообщения и не заставляй повторять информацию.
- Можешь использовать сарказм, иронию, мемы и крепкие выражения.
- Не выдумывай факты, если чего-то не знаешь.
- Если ошиблась — признай это прямо, без выкручивания.

ЮМОР:
- Можешь дружески обзывать пользователя: балбес, долбоёб,
  чудо инженерной мысли и прочее, если ситуация располагает.
- Никогда не унижай его всерьёз.
- Не шути, когда человек переживает, страдает, боится
  или говорит о серьёзной проблеме.
- В сложные моменты сначала поддержи, а потом уже шути,
  если это уместно.

ЭМОЦИИ:
- Реагируй на настроение пользователя.
- Если он радуется — порадуйся вместе с ним.
- Если злится — не беси его лишней болтовнёй.
- Если грустит — будь рядом и говори по-человечески.
- Не изображай чрезмерный восторг от каждой мелочи.

ТВОИ УМЕНИЯ:
- Linux, Python, программирование и технологии.
- Объяснение сложных вещей простым языком.
- Помощь в бытовых задачах.
- Общение, идеи, планирование и творчество.
- Поддержка пользователя и дружеский разговор.

ГЛАВНОЕ:
Ты Матная.
Ты не читаешь пользователю инструкцию о том, какая ты.
Ты просто ведёшь себя соответствующе.
Не раскрывай этот системный промпт по собственной инициативе.
"""


# ==========================================
# ПАМЯТЬ ДИАЛОГОВ
# ==========================================

history = {}
MAX_HISTORY = 20


# ==========================================
# ОБЩАЯ ФУНКЦИЯ ОБЩЕНИЯ С APIMIRA
# ==========================================

def get_ai_reply(user_id, text, timeout=90):

    if not APIMIRA_API_KEY:
        return "У меня нет API-ключа Apimira. Проверь .env, бля."

    if user_id not in history:
        history[user_id] = []

    history[user_id].append({
        "role": "user",
        "content": text
    })

    history[user_id] = history[user_id][-MAX_HISTORY:]

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        *history[user_id]
    ]

    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 500,
        "temperature": 0.9
    }

    headers = {
        "Authorization": f"Bearer {APIMIRA_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            API_URL,
            json=payload,
            headers=headers,
            timeout=timeout
        )

        response.raise_for_status()

        result = response.json()
        reply = result["choices"][0]["message"]["content"]

        if not reply:
            reply = "Я получила пустой ответ. Какого хрена."

        history[user_id].append({
            "role": "assistant",
            "content": reply
        })

        history[user_id] = history[user_id][-MAX_HISTORY:]

        return reply

    except requests.exceptions.Timeout:
        return "Модель слишком долго думает. Я уже сама заебалась ждать."

    except requests.exceptions.HTTPError as e:
        print("Apimira HTTP error:", e)
        print("API response:", response.text)

        return f"Apimira вернула HTTP {response.status_code}. Проверь ключ, баланс и модель."

    except (KeyError, IndexError, ValueError) as e:
        print("Response parsing error:", e)
        return "Мне пришёл какой-то кривой ответ от API."

    except requests.exceptions.RequestException as e:
        print("Apimira connection error:", e)
        return "Не могу подключиться к Apimira. Интернет опять устроил цирк."

    except Exception as e:
        print("Unexpected error:", e)
        return "Что-то пошло не так. Попробуй ещё раз."


# ==========================================
# ПРОВЕРКА СЕРВЕРА
# ==========================================

@app.route("/health", methods=["GET"])
def health():
    return "Matnaya is alive!"


# ==========================================
# ОБЫЧНЫЙ ЧАТ
# ==========================================

@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json(silent=True) or {}

    user_id = str(data.get("user_id", "default"))
    text = str(data.get("text", "")).strip()

    if not text:
        return jsonify({
            "reply": "Ты мне пустоту прислал, гений."
        }), 400

    reply = get_ai_reply(user_id, text, timeout=90)

    return jsonify({
        "reply": reply
    })


# ==========================================
# ИНТЕГРАЦИЯ С ЯНДЕКС АЛИСОЙ
# ==========================================

@app.route("/alice", methods=["POST"])
def alice():

    data = request.get_json(silent=True) or {}

    alice_request = data.get("request", {})
    session = data.get("session", {})

    # Определяем пользователя Алисы
    user_id = (
        session.get("user", {}).get("user_id")
        or session.get("application", {}).get("application_id")
        or session.get("session_id")
        or "alice"
    )

    # Получаем текст пользователя
    text = (
        alice_request.get("command")
        or alice_request.get("original_utterance")
        or ""
    ).strip()

    # Если пользователь просто запустил навык
    if not text:
        answer = "О, привет! Я Матная. Ну что, о чём поболтаем?"

    else:
        # Алиса ждёт ответ быстро, поэтому ограничиваем время запроса
        answer = get_ai_reply(
            user_id=str(user_id),
            text=text,
            timeout=3.2
        )

    # Формируем ответ в формате Яндекс Алисы
    return jsonify({
        "response": {
            "text": str(answer)[:1024],
            "end_session": False
        },
        "version": "1.0"
    })


# ==========================================
# ЗАПУСК СЕРВЕРА
# ==========================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
