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
# НАСТРОЙКИ
# ==========================================

APIMIRA_API_KEY = os.getenv("APIMIRA_API_KEY")

API_URL = os.getenv(
    "APIMIRA_API_URL",
    "https://apimira.com/v1/chat/completions"
)

MODEL = os.getenv(
    "OPENAI_MODEL",
    "openai/gpt-5.4-mini"
)

API_TIMEOUT = float(os.getenv("API_TIMEOUT", "60"))

# Алиса ждёт настоящий ответ модели до 3 секунд.
# Остальное время задача работает в фоне.
ALICE_WAIT_TIMEOUT = 3.0

MAX_HISTORY = 20
MAX_QUEUE = 5

executor = ThreadPoolExecutor(max_workers=8)

lock = threading.RLock()

# История отдельно для каждого пользователя.
history = defaultdict(list)

# Состояние пользователей.
users = {}


# ==========================================
# СИСТЕМНЫЙ ПРОМПТ МАТНОЙ
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

    with lock:

        history[user_id].append({
            "role": role,
            "content": content
        })

        history[user_id] = history[user_id][-MAX_HISTORY:]


def get_history(user_id):

    with lock:

        return list(history[user_id])


# ==========================================
# ЗАПРОС К APIMIRA
# ==========================================

def get_ai_reply(user_id, text):

    if not APIMIRA_API_KEY:
        return "Ошибка: не найден API-ключ ApiMira."

    # История обновляется последовательно
    # внутри обработчика очереди конкретного пользователя.

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    messages.extend(get_history(user_id))

    messages.append({
        "role": "user",
        "content": text
    })

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

        print(f"Sending to ApiMira: user={user_id}")

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
            raise ValueError("Empty AI response")

        reply = reply.strip()

        # Записываем обе реплики только после получения ответа.
        add_to_history(user_id, "user", text)
        add_to_history(user_id, "assistant", reply)

        print(f"ApiMira success: user={user_id}")

        return reply

    except requests.exceptions.Timeout:

        print("ApiMira timeout")

        return "Модель слишком долго отвечает. Попробуй ещё раз."

    except requests.exceptions.HTTPError:

        print(
            "ApiMira HTTP error:",
            response.status_code,
            response.text[:1500]
        )

        return f"Ошибка ApiMira HTTP {response.status_code}."

    except (KeyError, IndexError, TypeError, ValueError) as exc:

        print("ApiMira parsing error:", repr(exc))

        return "Получила непонятный ответ от ApiMira."

    except requests.exceptions.RequestException as exc:

        print("ApiMira connection error:", repr(exc))

        return "Не могу подключиться к ApiMira."

    except Exception as exc:

        print("Unexpected AI error:", repr(exc))

        return "Что-то пошло не так."


# ==========================================
# СОСТОЯНИЕ ПОЛЬЗОВАТЕЛЕЙ
# ==========================================

def get_user_state(user_id):

    with lock:

        if user_id not in users:

            users[user_id] = {
                "jobs": deque(),
                "worker_running": False
            }

        return users[user_id]


# ==========================================
# СОЗДАНИЕ ЗАДАЧИ
# ==========================================

def create_job(user_id, text):

    return {
        "text": text,
        "status": "queued",
        "result": None,
        "event": threading.Event()
    }


# ==========================================
# ФОНОВЫЙ ОБРАБОТЧИК ОЧЕРЕДИ
# ==========================================

def process_user_queue(user_id):

    state = get_user_state(user_id)

    while True:

        with lock:

            # Находим первый необработанный вопрос.

            job = next(
                (
                    item for item in state["jobs"]
                    if item["status"] == "queued"
                ),
                None
            )

            if job is None:

                state["worker_running"] = False

                print(
                    f"Queue empty: user={user_id}"
                )

                return

            job["status"] = "running"

        print(
            f"Processing question: user={user_id}"
        )

        # Запрос к модели выполняется вне блокировки.

        try:

            result = get_ai_reply(
                user_id,
                job["text"]
            )

        except Exception as exc:

            print("Worker exception:", repr(exc))

            result = "Произошла ошибка при обработке вопроса."

        with lock:

            job["result"] = result
            job["status"] = "done"

            job["event"].set()

            print(
                f"Question completed: user={user_id}"
            )


# ==========================================
# ДОБАВЛЕНИЕ В ОЧЕРЕДЬ
# ==========================================

def enqueue_job(user_id, text):

    state = get_user_state(user_id)

    with lock:

        unfinished = [
            job for job in state["jobs"]
            if job["status"] != "delivered"
        ]

        if len(unfinished) >= MAX_QUEUE:
            return None

        job = create_job(user_id, text)

        state["jobs"].append(job)

        if not state["worker_running"]:

            state["worker_running"] = True

            executor.submit(
                process_user_queue,
                user_id
            )

        print(
            f"Question queued: user={user_id}, "
            f"queue_size={len(state['jobs'])}"
        )

        return job


# ==========================================
# ПОЛУЧЕНИЕ ГОТОВОГО ОТВЕТА
# ==========================================

def take_completed_answer(user_id):

    state = get_user_state(user_id)

    with lock:

        for job in state["jobs"]:

            if job["status"] == "done":

                job["status"] = "delivered"

                answer = job["result"]

                # Удаляем уже выданную задачу.

                state["jobs"].remove(job)

                return answer

        return None


# ==========================================
# ПРОВЕРКА ФРАЗЫ ГОТОВНОСТИ
# ==========================================

def is_status_question(text):

    text = (text or "").strip().lower()

    phrases = (
        "ответ готов",
        "готов ответ",
        "матная ответ готов",
        "проверь ответ",
        "проверить ответ",
        "есть ответ",
        "ты закончила",
        "ну что там"
    )

    return any(
        phrase in text
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

    # Обычный чат выполняет запрос синхронно.

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

    print("ALICE INPUT:", {
        "text": text,
        "user_id": user_id
    })

    # Приветствие
    if not text:
        return alice_response(
            "О, привет! Я Матная. Ну что, о чём поболтаем?",
            session
        )

    # Проверяем, не готов ли предыдущий ответ
    if is_status_question(text):
        answer = take_completed_answer(user_id)

        if answer is not None:
            return alice_response(answer, session)

        state = get_user_state(user_id)

        with lock:
            has_pending = any(
                job["status"] in ("queued", "running")
                for job in state["jobs"]
            )

        if has_pending:
            return alice_response(
                "Ещё думаю над твоим вопросом. Подожди немного.",
                session
            )

        return alice_response(
            "Пока готовых ответов нет. Чего хотел?",
            session
        )

    # СНАЧАЛА забираем старый готовый ответ
    previous_answer = take_completed_answer(user_id)

    if previous_answer is not None:

        # Текущую реплику тоже сохраняем в очередь
        new_job = enqueue_job(user_id, text)

        if new_job is None:
            return alice_response(
                previous_answer + " А новый вопрос пока не влез в очередь. Повтори его.",
                session
            )

        print("Delivering previous answer:", user_id)

        return alice_response(
            previous_answer,
            session
        )

    # Проверяем, есть ли незавершённые задачи
    state = get_user_state(user_id)

    with lock:
        has_unanswered = any(
            job["status"] in ("queued", "running")
            for job in state["jobs"]
        )

    # Добавляем новый вопрос
    job = enqueue_job(user_id, text)

    if job is None:
        return alice_response(
            "У меня уже очередь набилась. Дай сначала закончить предыдущие вопросы.",
            session
        )

    # Если уже обрабатывается другой вопрос
    if has_unanswered:
        return alice_response(
            "Запомнила. Сначала закончу предыдущий вопрос.",
            session
        )

    # Ждём ответ модели
    if job["event"].wait(timeout=ALICE_WAIT_TIMEOUT):

        with lock:
            if job["status"] == "done":
                answer = job["result"]
                job["status"] = "delivered"

                try:
                    state["jobs"].remove(job)
                except ValueError:
                    pass

                return alice_response(answer, session)

    return alice_response(
        "Секунду, я ещё думаю над твоим вопросом.",
        session
    )


# ==========================================
# ОТВЕТ АЛИСЕ
# ==========================================

def alice_response(text, session):
    text = str(text)

    # Ограничиваем ответ до 1000 байт UTF-8
    encoded = text.encode("utf-8")[:1000]

    # Убираем обрезанный символ, если он попал на границу
    safe_text = encoded.decode("utf-8", errors="ignore")

    result = {
        "response": {
            "text": safe_text,
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
