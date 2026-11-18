import os
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError

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

MODEL = os.getenv(
    "OPENAI_MODEL",
    "openai/gpt-5.6-terra"
)

API_TIMEOUT = float(os.getenv("API_TIMEOUT", "60"))

# Сколько Алиса ждёт модель непосредственно
ALICE_WAIT_TIMEOUT = 3.0

MAX_HISTORY = 20
MAX_QUEUED_MESSAGES = 5

executor = ThreadPoolExecutor(max_workers=8)

state_lock = threading.RLock()

history = defaultdict(list)

# Состояние пользователей
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
# ИСТОРИЯ
# ==========================================

def add_to_history(user_id, role, content):

    with state_lock:

        history[user_id].append({
            "role": role,
            "content": content
        })

        history[user_id] = history[user_id][-MAX_HISTORY:]


# ==========================================
# APIMIRA
# ==========================================

def get_ai_reply(user_id, text):

    if not APIMIRA_API_KEY:

        return "Нет API-ключа ApiMira. Проверь настройки Render."

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

            reply = "Что-то модель мне пустоту прислала."

        reply = reply.strip()

        add_to_history(user_id, "assistant", reply)

        print(f"AI response OK: user={user_id}")

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
            f"Ошибка ApiMira HTTP {response.status_code}. "
            "Проверь API-ключ и баланс."
        )

    except (KeyError, IndexError, TypeError, ValueError) as exc:

        print("ApiMira parsing error:", repr(exc))

        reply = "Получила непонятный ответ от ApiMira."

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
# ЗАПУСК СЛЕДУЮЩЕГО ВОПРОСА
# ==========================================

def start_next_locked(user_id):

    state = users[user_id]

    if state["future"] is not None:
        return

    if not state["queue"]:
        return

    text = state["queue"].popleft()

    state["future"] = executor.submit(
        get_ai_reply,
        user_id,
        text
    )

    print(
        f"Started background request: {user_id}"
    )


# ==========================================
# ПРОВЕРКА ЗАВЕРШЕНИЯ ЗАДАЧИ
# ==========================================

def finish_task_locked(user_id):

    state = get_user_state(user_id)

    future = state["future"]

    if future is None:
        return None, False

    if not future.done():
        return None, False

    try:

        answer = future.result()

    except Exception as exc:

        print("Future error:", repr(exc))

        answer = "У меня произошла ошибка при обработке вопроса."

    state["future"] = None

    # Запускаем следующий вопрос из очереди.

    start_next_locked(user_id)

    return answer, True


# ==========================================
# ПРОВЕРКА ФРАЗЫ О ГОТОВНОСТИ
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
        "получилось",
        "ты уже ответила",
    )

    return any(
        phrase in normalized
        for phrase in phrases
    )


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
# АЛИСА
# ==========================================

@app.route("/alice", methods=["POST"])
def alice():

    data = request.get_json(silent=True) or {}

    alice_request = data.get("request") or {}

    session = data.get("session") or {}

    # Идентификатор пользователя

    user_id = (
        (session.get("user") or {}).get("user_id")
        or session.get("user_id")
        or session.get("session_id")
        or "alice"
    )

    user_id = str(user_id)

    text = str(
        alice_request.get("command")
        or alice_request.get("original_utterance")
        or ""
    ).strip()

    # Запуск навыка

    if not text:

        return alice_response(
            "О, привет! Я Матная. Ну что, о чём поболтаем?",
            session
        )

    state = get_user_state(user_id)

    # ======================================
    # ПРОВЕРЯЕМ ПРЕДЫДУЩИЙ ЗАПРОС
    # ======================================

    with state_lock:

        answer, finished = finish_task_locked(user_id)

        if finished:

            # Если пользователь проверяет готовность,
            # отдаём готовый ответ.

            if is_status_question(text):

                return alice_response(answer, session)

            # Если пришёл новый вопрос,
            # сохраняем его в очередь.

            if len(state["queue"]) < MAX_QUEUED_MESSAGES:

                state["queue"].append(text[:2000])

            return alice_response(answer, session)

        # Если задача ещё работает.

        if state["future"] is not None:

            if is_status_question(text):

                return alice_response(
                    "Ещё немного, я заканчиваю предыдущий ответ.",
                    session
                )

            if len(state["queue"]) < MAX_QUEUED_MESSAGES:

                state["queue"].append(text[:2000])

                return alice_response(
                    "Секунду, я ещё разбираюсь с предыдущим вопросом.",
                    session
                )

            return alice_response(
                "У меня уже очередь вопросов. Давай немного подождём.",
                session
            )

        # Если очередь осталась, запускаем её.

        if state["queue"]:

            start_next_locked(user_id)

            return alice_response(
                "Секунду, я заканчиваю предыдущий вопрос.",
                session
            )

        # Проверка статуса без активной задачи.

        if is_status_question(text):

            return alice_response(
                "У меня сейчас нет ожидающего ответа. Что случилось?",
                session
            )

        # Запускаем новую задачу.

        future = executor.submit(
            get_ai_reply,
            user_id,
            text
        )

        state["future"] = future

    # ======================================
    # ПЫТАЕМСЯ ПОЛУЧИТЬ ОТВЕТ СРАЗУ
    # ======================================

    try:

        answer = future.result(
            timeout=ALICE_WAIT_TIMEOUT
        )

        with state_lock:

            # Проверяем, не забрал ли ответ
            # параллельный запрос.

            if state["future"] is future:

                state["future"] = None

                start_next_locked(user_id)

        print("Alice received direct AI response")

        return alice_response(answer, session)

    except TimeoutError:

        print(
            f"Alice background processing: {user_id}"
        )

        return alice_response(
            "Секунду, я думаю над ответом.",
            session
        )

    except Exception as exc:

        print("Alice future error:", repr(exc))

        with state_lock:

            if state["future"] is future:

                state["future"] = None

                start_next_locked(user_id)

        return alice_response(
            "Что-то пошло не так при обработке вопроса.",
            session
        )


# ==========================================
# ФОРМИРОВАНИЕ ОТВЕТА АЛИСЫ
# ==========================================

def alice_response(text, session):

    result = {
        "response": {
            "text": str(text)[:1024],
            "end_session": False
        },
        "version": "1.0"
    }

    session_state = session.get("session_state")

    if session_state is not None:

        result["session_state"] = session_state

    return jsonify(result)


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
# ЗАПУСК
# ==========================================

if __name__ == "__main__":

    port = int(os.getenv("PORT", "5000"))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
