import os
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ==========================================
# НАСТРОЙКИ APIMIRA
# ==========================================

APIMIRA_API_KEY = os.getenv("APIMIRA_API_KEY")
API_URL = os.getenv(
    "APIMIRA_API_URL",
    "https://apimira.com/v1/chat/completions"
)
MODEL = os.getenv("OPENAI_MODEL", "openai/gpt-5.6-terra")

API_TIMEOUT = float(os.getenv("API_TIMEOUT", "60"))

MAX_HISTORY = 20
MAX_QUEUED_MESSAGES = 5

executor = ThreadPoolExecutor(max_workers=8)

state_lock = threading.RLock()

history = defaultdict(list)

# Состояние каждого пользователя:
# future — выполняющийся запрос
# queue — очередь сообщений

users = {}


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
# ИСТОРИЯ ДИАЛОГА
# ==========================================

def add_to_history(user_id, role, content):

    with state_lock:

        history[user_id].append({
            "role": role,
            "content": content
        })

        history[user_id] = history[user_id][-MAX_HISTORY:]


# ==========================================
# ЗАПРОС К APIMIRA
# ==========================================

def get_ai_reply(user_id, text):

    if not APIMIRA_API_KEY:
        return (
            "У меня нет API-ключа ApiMira. "
            "Проверь APIMIRA_API_KEY в Render."
        )

    add_to_history(user_id, "user", text)

    with state_lock:

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ] + list(history[user_id])

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
            timeout=(5, API_TIMEOUT)
        )

        response.raise_for_status()

        result = response.json()

        reply = result["choices"][0]["message"]["content"]

        if not isinstance(reply, str) or not reply.strip():

            reply = "Я получила пустой ответ. Какого хрена."

        reply = reply.strip()

        add_to_history(user_id, "assistant", reply)

        return reply

    except requests.exceptions.Timeout:

        print("ApiMira timeout")

        reply = "Модель слишком долго отвечает. Попробуй ещё раз."

    except requests.exceptions.HTTPError:

        print(
            "ApiMira HTTP error:",
            response.status_code,
            response.text[:1500]
        )

        reply = (
            f"ApiMira вернула HTTP {response.status_code}. "
            "Проверь API-ключ, модель и баланс."
        )

    except (KeyError, IndexError, TypeError, ValueError) as exc:

        print("ApiMira parsing error:", repr(exc))

        reply = "Мне пришёл непонятный ответ от ApiMira."

    except requests.exceptions.RequestException as exc:

        print("ApiMira connection error:", repr(exc))

        reply = "Не могу подключиться к ApiMira."

    except Exception as exc:

        print("Unexpected AI error:", repr(exc))

        reply = "Что-то пошло не так."

    add_to_history(user_id, "assistant", reply)

    return reply


# ==========================================
# СОСТОЯНИЕ ПОЛЬЗОВАТЕЛЕЙ
# ==========================================

def get_user_state(user_id):

    with state_lock:

        if user_id not in users:

            users[user_id] = {
                "future": None,
                "queue": deque()
            }

        return users[user_id]


# ==========================================
# ЗАПУСК СЛЕДУЮЩЕГО ЗАПРОСА
# ==========================================

def start_next_locked(user_id):

    state = users[user_id]

    if state["future"] is not None:
        return

    if not state["queue"]:
        return

    text = state["queue"].popleft()

    future = executor.submit(
        get_ai_reply,
        user_id,
        text
    )

    state["future"] = future

    print(
        f"Background task started: "
        f"user={user_id}, remaining={len(state['queue'])}"
    )


# ==========================================
# ДОБАВЛЕНИЕ СООБЩЕНИЯ В ОЧЕРЕДЬ
# ==========================================

def enqueue_message(user_id, text):

    with state_lock:

        state = get_user_state(user_id)

        if len(state["queue"]) >= MAX_QUEUED_MESSAGES:

            return False

        state["queue"].append(text[:2000])

        start_next_locked(user_id)

        return True


# ==========================================
# ПРОВЕРКА ГОТОВНОСТИ
# ==========================================

def is_status_question(text):

    normalized = (text or "").strip().lower()

    phrases = (
        "ответ готов",
        "готов ответ",
        "ты закончила",
        "ну что там",
        "проверить ответ",
        "есть ответ",
        "матная ответь",
        "получилось",
        "ты уже ответила",
        "готово",
    )

    return any(
        phrase in normalized
        for phrase in phrases
    )


def collect_finished_answer(user_id):

    with state_lock:

        state = get_user_state(user_id)

        future = state["future"]

        if future is None:

            if state["queue"]:

                start_next_locked(user_id)

                return None

            return None

        if not future.done():

            return None

        try:

            answer = future.result()

        except Exception as exc:

            print("Background task error:", repr(exc))

            answer = "У меня тут фоновая задачка споткнулась."

        state["future"] = None

        # Запускаем следующее сообщение,
        # если пользователь успел задать ещё вопросы.

        start_next_locked(user_id)

        return answer


# ==========================================
# ПРОВЕРКА СЕРВЕРА
# ==========================================

@app.route("/", methods=["GET"])
def index():

    return "Matnaya is alive!"


@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok",
        "service": "matnaya",
        "model": MODEL
    })


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

    reply = get_ai_reply(user_id, text)

    return jsonify({
        "reply": reply
    })


# ==========================================
# ИНТЕГРАЦИЯ С АЛИСОЙ
# ==========================================

@app.route("/alice", methods=["POST"])
def alice():

    data = request.get_json(silent=True) or {}

    alice_request = data.get("request") or {}

    session = data.get("session") or {}

    # Правильное получение идентификатора пользователя Яндекса.

    user_id = (
        session.get("user_id")
        or (session.get("user") or {}).get("user_id")
        or session.get("session_id")
        or "alice"
    )

    user_id = str(user_id)

    text = str(
        alice_request.get("command")
        or alice_request.get("original_utterance")
        or ""
    ).strip()

    # Запуск навыка.

    if not text:

        return alice_response(
            "О, привет! Я Матная. Ну что, о чём поболтаем?",
            session
        )

    # Проверка готовности предыдущего ответа.

    if is_status_question(text):

        answer = collect_finished_answer(user_id)

        if answer is None:

            answer = (
                "Я ещё думаю над предыдущим вопросом. "
                "Спроси меня чуть позже."
            )

        return alice_response(answer, session)

    # Добавляем новое сообщение в очередь.

    accepted = enqueue_message(user_id, text)

    if not accepted:

        return alice_response(
            "У меня уже очередь вопросов. "
            "Давай сначала разберём предыдущие.",
            session
        )

    return alice_response(
        "Вопрос получила. Сейчас разберусь.",
        session
    )


# ==========================================
# ФОРМИРОВАНИЕ ОТВЕТА АЛИСЫ
# ==========================================

def alice_response(text, session):

    response = {
        "text": str(text)[:1024],
        "end_session": False
    }

    result = {
        "response": response,
        "version": "1.0"
    }

    # Сохраняем состояние сессии, если оно есть.

    session_state = session.get("session_state")

    if session_state is not None:

        result["session_state"] = session_state

    return jsonify(result)


# ==========================================
# ЗАПУСК
# ==========================================

if __name__ == "__main__":

    port = int(os.getenv("PORT", "5000"))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
