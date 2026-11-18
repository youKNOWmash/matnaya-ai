import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ==========================================
# НАСТРОЙКИ APIMIRA
# ==========================================

APIMIRA_API_KEY = os.getenv("APIMIRA_API_KEY")
API_URL = os.getenv("APIMIRA_API_URL", "https://apimira.com/v1/chat/completions")
MODEL = os.getenv("OPENAI_MODEL", "openai/gpt-5.6-terra")

# Алиса должна получить HTTP-ответ быстро. Долгий запрос к модели
# выполняется в фоновом потоке, а результат забирается отдельной фразой.
ALICE_REQUEST_TIMEOUT = float(os.getenv("ALICE_REQUEST_TIMEOUT", "60"))
MAX_HISTORY = 20
MAX_QUEUED_MESSAGES = 1

# Один процесс приложения хранит задачи в памяти.
# При перезапуске Render незавершённые фоновые задачи пропадут.
executor = ThreadPoolExecutor(max_workers=4)
state_lock = threading.RLock()
history = defaultdict(list)
pending = {}       # user_id -> {"future": Future, "queued_text": str|None}
completed = {}     # user_id -> готовый ответ, если задача завершилась


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
# ОБЩАЯ ФУНКЦИЯ ОБЩЕНИЯ С APIMIRA
# ==========================================

def get_ai_reply(user_id, text, timeout=ALICE_REQUEST_TIMEOUT):
    """Синхронный запрос к ApiMira. Вызывается обычным чатом или фоновым потоком."""
    if not APIMIRA_API_KEY:
        return "У меня нет API-ключа ApiMira. Проверь переменную APIMIRA_API_KEY в Render."

    with state_lock:
        history[user_id].append({"role": "user", "content": text})
        history[user_id] = history[user_id][-MAX_HISTORY:]
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(history[user_id])

    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 500,
        "temperature": 0.9,
    }
    headers = {
        "Authorization": f"Bearer {APIMIRA_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            API_URL,
            json=payload,
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()

        result = response.json()
        reply = result["choices"][0]["message"]["content"]
        if not isinstance(reply, str) or not reply.strip():
            reply = "Я получила пустой ответ. Какого хрена."

        with state_lock:
            history[user_id].append({"role": "assistant", "content": reply})
            history[user_id] = history[user_id][-MAX_HISTORY:]

        return reply.strip()

    except requests.exceptions.Timeout:
        return "Модель слишком долго отвечает. Попробуй проверить готовность ещё раз чуть позже."
    except requests.exceptions.HTTPError:
        # Подробности остаются в логах Render, ключ авторизации туда не выводим.
        print("ApiMira HTTP error:", response.status_code, response.text[:1500])
        return f"ApiMira вернула HTTP {response.status_code}. Проверь модель, API-ключ и доступный баланс."
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        print("ApiMira response parsing error:", repr(exc))
        return "Мне пришёл непонятный ответ от ApiMira."
    except requests.exceptions.RequestException as exc:
        print("ApiMira connection error:", repr(exc))
        return "Не могу подключиться к ApiMira. Попробуй ещё раз."
    except Exception as exc:
        print("Unexpected AI error:", repr(exc))
        return "Что-то пошло не так. Попробуй ещё раз."


# ==========================================
# ФОНОВАЯ ОБРАБОТКА ДЛЯ АЛИСЫ
# ==========================================

def is_status_question(text):
    normalized = (text or "").strip().lower()
    phrases = (
        "ответ готов", "готов ответ", "ты закончила", "ну что там",
        "проверить ответ", "есть ответ", "матная ответь", "получилось",
    )
    return any(phrase in normalized for phrase in phrases)


def start_background_answer(user_id, text):
    future = executor.submit(get_ai_reply, user_id, text, ALICE_REQUEST_TIMEOUT)
    with state_lock:
        pending[user_id] = {"future": future, "queued_text": None}


def collect_finished_answer(user_id):
    """Переносит результат Future в completed, если он уже готов."""
    with state_lock:
        task = pending.get(user_id)
        if not task:
            return completed.pop(user_id, None)

        future = task["future"]
        if not future.done():
            return None

        try:
            answer = future.result()
        except Exception as exc:
            print("Background task error:", repr(exc))
            answer = "У меня тут фоновая задачка споткнулась. Попробуй ещё раз."

        queued_text = task.get("queued_text")
        pending.pop(user_id, None)

        if queued_text:
            # Запускаем отложенное сообщение после завершения предыдущего.
            start_background_answer(user_id, queued_text)

        return answer


# ==========================================
# ПРОВЕРКА СЕРВЕРА
# ==========================================

@app.route("/", methods=["GET"])
def index():
    return "Matnaya is alive!"


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "matnaya", "model": MODEL})


# ==========================================
# ОБЫЧНЫЙ ЧАТ
# ==========================================

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_id = str(data.get("user_id", "default"))
    text = str(data.get("text", "")).strip()

    if not text:
        return jsonify({"reply": "Ты мне пустоту прислал, гений."}), 400

    return jsonify({"reply": get_ai_reply(user_id, text)})


# ==========================================
# ИНТЕГРАЦИЯ С ЯНДЕКС АЛИСОЙ
# ==========================================

@app.route("/alice", methods=["POST"])
def alice():
    data = request.get_json(silent=True) or {}
    alice_request = data.get("request") or {}
    session = data.get("session") or {}

    # Для одного пользователя навыка используем стабильный user_id.
    user_id = (
        (session.get("user") or {}).get("user_id")
        or (session.get("application") or {}).get("application_id")
        or session.get("session_id")
        or "alice"
    )
    user_id = str(user_id)

    text = str(
        alice_request.get("command")
        or alice_request.get("original_utterance")
        or ""
    ).strip()

    if not text:
        answer = (
            "О, привет! Я Матная. Если мой ответ задержится, "
            "скажи: «Матная, ответ готов?»"
        )
        return alice_response(answer, session)

    # Пользователь проверяет, готов ли предыдущий ответ.
    if is_status_question(text):
        answer = collect_finished_answer(user_id)
        if answer is None:
            answer = "Ещё думаю, не торопи лошадей. Спроси «Матная, ответ готов?» чуть позже."
        elif answer == "":
            answer = "У меня пока нет ожидающего ответа. Скажи, что хочешь узнать."
        return alice_response(answer, session)

    # Если предыдущая задача ещё выполняется — сохраняем одно следующее сообщение,
    # чтобы не потерять его, и просим проверить готовность позже.
    with state_lock:
        task = pending.get(user_id)
        if task:
            if task.get("queued_text") is None:
                task["queued_text"] = text[:2000]
                answer = "Я ещё отвечаю на предыдущий вопрос и запомнила следующий. Спроси «Матная, ответ готов?» чуть позже."
            else:
                answer = "Я уже запомнила следующий вопрос. Сначала забери мой предыдущий ответ: скажи «Матная, ответ готов?»."
            return alice_response(answer, session)

    # Быстро подтверждаем запрос, а модель отвечает в фоне.
    start_background_answer(user_id, text)
    return alice_response(
        "Приняла вопрос. Я уже думаю над ответом. Скажи «Матная, ответ готов?» через несколько секунд.",
        session,
    )


def alice_response(text, session):
    response = {
        "text": str(text)[:1024],
        "end_session": False,
    }

    # Сохраняем session_state, если Яндекс передал его в исходном запросе.
    session_state = session.get("session_state")
    result = {
        "response": response,
        "version": "1.0",
    }
    if session_state is not None:
        result["session_state"] = session_state

    return jsonify(result)


# ==========================================
# ЗАПУСК ЛОКАЛЬНО
# ==========================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
