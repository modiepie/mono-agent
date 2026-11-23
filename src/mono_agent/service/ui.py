"""Streamlit demo UI. All interface text is Ukrainian; the assistant answers in the user's
language. The UI only talks to the HTTP API (API_URL), never to the graph directly.

    uv run streamlit run src/mono_agent/service/ui.py
"""

import json
import os
from collections.abc import Iterator

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

STEPS = {
    "guardrail": "Перевіряю запит",
    "router": "Визначаю, що потрібно зробити",
    "transactions": "Аналізую транзакції",
    "transactions_tools": "Звертаюся до даних",
    "knowledge": "Шукаю в документах банку",
    "export": "Готую звіт",
    "composer": "Формую й перевіряю відповідь",
    "refuse": "Перевіряю запит",
}
TOOLS = {
    "sync_statement": "оновлення виписки",
    "query_transactions": "пошук транзакцій",
    "aggregate_spending": "підсумки витрат",
    "compare_periods": "порівняння періодів",
    "detect_subscriptions": "пошук підписок",
    "currency_rates": "курси валют",
    "forecast_savings": "розрахунок заощаджень",
}
EXAMPLES = [
    "Скільки я витратив на таксі в жовтні і чому більше, ніж у вересні?",
    "Які підписки я оплачую і які виглядають зайвими?",
    "Як працює пільговий період кредитного ліміту?",
    "Якщо скоротити ресторани на 30%, скільки я заощаджу за 3 місяці?",
]


def sse_events(path: str, payload: dict) -> Iterator[tuple[str, dict]]:
    with httpx.stream("POST", f"{API_URL}{path}", json=payload, timeout=300) as response:
        response.raise_for_status()
        event = None
        for line in response.iter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: ") and event:
                yield event, json.loads(line[6:])


@st.cache_data(ttl=30, show_spinner=False)
def health() -> dict | None:
    try:
        return httpx.get(f"{API_URL}/health", timeout=5).json()
    except httpx.HTTPError:
        return None


def render_details(answer: dict) -> None:
    tools, docs = answer.get("tools") or [], answer.get("documents") or []
    if not tools and not docs:
        return
    with st.expander("Як отримано відповідь"):
        for t in tools:
            mark = "✓" if t["ok"] else "✗"
            st.markdown(
                f"{mark} **{TOOLS.get(t['tool'], t['tool'])}** `{
                    json.dumps(t['args'], ensure_ascii=False)
                }`"
            )
        if docs:
            st.markdown("**Джерела:**")
            cited = {c["ref"] for c in answer.get("citations", []) if c["kind"] == "doc"}
            for d in docs:
                note = " _(зразок документа, не офіційні умови)_" if d.get("sample") else ""
                used = " — використано у відповіді" if d["id"] in cited else ""
                st.markdown(f"- `{d['id']}` {d['title']} › {d['section']}{note}{used}")


def run_turn(path: str, payload: dict) -> None:
    """Stream one turn: progress inside a status box, then the answer or a confirmation."""
    with st.chat_message("assistant"):
        status = st.status("Думаю…", expanded=False)
        try:
            for event, data in sse_events(path, payload):
                if event == "thread":
                    st.session_state.thread_id = data["thread_id"]
                elif event == "step":
                    label: str = STEPS.get(data["node"]) or str(data["node"])
                    tool = data["detail"].get("tool")
                    if tool:
                        label += f": {TOOLS.get(tool, tool)}"
                    status.update(label=label)
                    status.write(label)
                elif event == "confirm":
                    status.update(label="Потрібне ваше підтвердження", state="complete")
                    st.session_state.pending = data
                elif event == "answer":
                    status.update(label="Готово", state="complete")
                    st.markdown(data["text"])  # includes the report path after an export
                    render_details(data)
                    st.session_state.messages.append({"role": "assistant", "answer": data})
                elif event == "error":
                    status.update(label="Помилка", state="error")
                    st.error("Не вдалося отримати відповідь. Спробуйте ще раз.")
        except httpx.HTTPError:
            status.update(label="Помилка", state="error")
            st.error("Сервіс недоступний. Перевірте, що API запущено.")


def main() -> None:
    st.set_page_config(page_title="Фінансовий помічник mono", page_icon="💳")
    ss = st.session_state
    ss.setdefault("messages", [])
    ss.setdefault("thread_id", None)
    ss.setdefault("pending", None)

    info = health()
    with st.sidebar:
        st.header("Налаштування")
        if info is None:
            st.error("API недоступне за адресою " + API_URL)
        else:
            data = "реальні дані monobank" if info["data"] == "real" else "синтетичні дані (демо)"
            st.markdown(f"**Дані:** {data}")
            st.markdown(f"**Сьогодні для помічника:** {info['today']}")
            st.markdown(f"**Модель відповідей:** `{info['models']['composer']}`")
            st.markdown(
                "**Трасування Langfuse:** " + ("увімкнено" if info["tracing"] else "вимкнено")
            )
        if st.button("Нова розмова", use_container_width=True):
            ss.messages, ss.thread_id, ss.pending = [], None, None
            st.rerun()
        st.subheader("Спробуйте запитати")
        for example in EXAMPLES:
            if st.button(example, use_container_width=True):
                ss.queued = example

    st.title("Фінансовий помічник monobank")
    st.caption(
        "Аналізує ваші витрати, підписки й кешбек і відповідає на питання про тарифи. "
        "Відповіді не є фінансовою порадою."
    )

    for m in ss.messages:
        with st.chat_message(m["role"]):
            if m["role"] == "user":
                st.markdown(m["text"])
            else:
                st.markdown(m["answer"]["text"])
                render_details(m["answer"])

    def choose(approve: bool) -> None:
        # Runs before the page is redrawn, so the buttons disappear right after a click.
        ss.choice, ss.pending = approve, None
        ss.messages.append({"role": "user", "text": "Так, зберегти" if approve else "Ні"})

    if ss.pending:
        with st.chat_message("assistant"):
            st.info("Зберегти звіт у файл? Дані з вашої виписки буде записано на диск.")
            yes, no = st.columns(2)
            yes.button(
                "Так, зберегти",
                type="primary",
                use_container_width=True,
                on_click=choose,
                args=(True,),
            )
            no.button("Ні, не потрібно", use_container_width=True, on_click=choose, args=(False,))
        return

    if "choice" in ss:
        run_turn(f"/chat/{ss.thread_id}/resume", {"approve": ss.pop("choice")})
        return

    prompt = st.chat_input("Запитайте про свої витрати або умови банку…")
    prompt = prompt or ss.pop("queued", None)
    if prompt:
        ss.messages.append({"role": "user", "text": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        payload = {"message": prompt, **({"thread_id": ss.thread_id} if ss.thread_id else {})}
        run_turn("/chat", payload)
        if ss.pending:
            st.rerun()  # show the confirmation buttons


main()
