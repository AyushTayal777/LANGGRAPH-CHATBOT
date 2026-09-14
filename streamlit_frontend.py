import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from langgraph_backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
)


# =========================== Utilities ===========================

def generate_thread_id():
    """
    Generate an internal thread ID.

    Users never see this ID.
    """
    return str(uuid.uuid4())


def reset_chat():
    """
    Start a completely new conversation.
    """
    thread_id = generate_thread_id()

    st.session_state["thread_id"] = thread_id

    add_thread(thread_id)

    st.session_state["message_history"] = []

    st.session_state["pending_interrupt"] = None


def add_thread(thread_id):
    """
    Add a thread to the current session's chat history.
    """
    thread_id = str(thread_id)

    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    """
    Load messages for a conversation from LangGraph checkpointing.
    """
    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": str(thread_id)
            }
        }
    )

    return state.values.get("messages", [])


def get_config(thread_key):
    """
    LangGraph configuration for the current conversation.
    """
    return {
        "configurable": {
            "thread_id": str(thread_key)
        },
        "metadata": {
            "thread_id": str(thread_key)
        },
        "run_name": "chat_turn",
    }


def get_chat_title(thread_id):
    """
    Generate a user-friendly chat title from
    the first user message.

    The actual LangGraph thread ID remains hidden.
    """

    try:

        messages = load_conversation(thread_id)

        for message in messages:

            if isinstance(message, HumanMessage):

                content = str(
                    message.content
                ).strip()

                if content:

                    # Keep sidebar titles short
                    if len(content) > 38:

                        return (
                            content[:38]
                            .rstrip()
                            + "..."
                        )

                    return content

    except Exception:
        pass

    return "New chat"


# ======================= Session Initialization ===================

if "message_history" not in st.session_state:

    st.session_state["message_history"] = []


if "thread_id" not in st.session_state:

    st.session_state["thread_id"] = (
        generate_thread_id()
    )


if "chat_threads" not in st.session_state:

    st.session_state["chat_threads"] = (
        retrieve_all_threads()
    )


# Make sure all existing thread IDs are strings
st.session_state["chat_threads"] = [
    str(thread_id)
    for thread_id in st.session_state["chat_threads"]
]


if "ingested_docs" not in st.session_state:

    st.session_state["ingested_docs"] = {}


if "pending_interrupt" not in st.session_state:

    st.session_state["pending_interrupt"] = None


# Make sure current conversation exists
add_thread(
    st.session_state["thread_id"]
)


# Current internal thread ID
thread_key = str(
    st.session_state["thread_id"]
)


# PDF information belonging to current conversation
thread_docs = (
    st.session_state["ingested_docs"]
    .setdefault(
        thread_key,
        {}
    )
)


# Newest conversations first
threads = [
    str(thread_id)
    for thread_id
    in st.session_state["chat_threads"]
][::-1]


selected_thread = None


# ============================ Sidebar ============================

st.sidebar.title("AgentFlow AI")


# ============================ New Chat ============================

if st.sidebar.button(
    "＋ New chat",
    use_container_width=True,
):

    reset_chat()

    st.rerun()


st.sidebar.divider()


# ============================ Current PDF ==========================

if thread_docs:

    latest_doc = list(
        thread_docs.values()
    )[-1]

    st.sidebar.markdown(
        "**📎 Current PDF**"
    )

    st.sidebar.caption(
        latest_doc.get(
            "filename",
            "Document"
        )
    )

    st.sidebar.caption(
        f"{latest_doc.get('documents', 0)} pages • "
        f"{latest_doc.get('chunks', 0)} chunks"
    )

else:

    st.sidebar.caption(
        "No PDF attached"
    )


# ============================ PDF Upload ===========================

uploaded_pdf = st.sidebar.file_uploader(
    "Upload a PDF",
    type=["pdf"],
)


if uploaded_pdf:

    if uploaded_pdf.name in thread_docs:

        st.sidebar.info(
            "This PDF is already indexed."
        )

    else:

        with st.sidebar.status(
            "Indexing PDF…",
            expanded=True
        ) as status_box:

            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )

            thread_docs[
                uploaded_pdf.name
            ] = summary

            status_box.update(
                label="✅ PDF indexed",
                state="complete",
                expanded=False,
            )


st.sidebar.divider()


# ============================ Chats ================================

st.sidebar.markdown(
    "**Chats**"
)


if not threads:

    st.sidebar.caption(
        "No conversations yet."
    )

else:

    for thread_id in threads:

        title = get_chat_title(
            thread_id
        )

        # Highlight currently selected chat
        if str(thread_id) == thread_key:

            label = f"● {title}"

        else:

            label = f"  {title}"


        if st.sidebar.button(
            label,
            key=f"side-thread-{thread_id}",
            use_container_width=True,
        ):

            selected_thread = str(
                thread_id
            )


# ======================= Load Selected Chat ========================

if selected_thread:

    st.session_state[
        "thread_id"
    ] = selected_thread


    messages = load_conversation(
        selected_thread
    )


    st.session_state[
        "message_history"
    ] = [

        {
            "role": (
                "user"
                if isinstance(
                    msg,
                    HumanMessage
                )
                else "assistant"
            ),
            "content": msg.content,
        }

        for msg in messages

        if isinstance(
            msg,
            (
                HumanMessage,
                AIMessage
            )
        )
    ]


    # Make sure PDF metadata container exists
    st.session_state[
        "ingested_docs"
    ].setdefault(
        selected_thread,
        {}
    )


    # Clear any HITL interrupt
    st.session_state[
        "pending_interrupt"
    ] = None


    st.rerun()


# ============================ Main Layout ===========================

st.title(
    "AgentFlow AI"
)

st.caption(
    "Ask questions, search the web, use tools, "
    "or chat with your PDF."
)


# ============================ Chat History ==========================

for message in st.session_state[
    "message_history"
]:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================ HITL Resume ===========================
#
# IMPORTANT:
# This is OUTSIDE `if user_input`.
# Therefore the buttons survive Streamlit reruns.


if st.session_state[
    "pending_interrupt"
]:

    pending = st.session_state[
        "pending_interrupt"
    ]


    st.warning(
        f"⚠️ {pending['message']}"
    )


    col1, col2 = st.columns(2)


    # ======================== Approve ==============================

    with col1:

        approve = st.button(
            "✅ Approve",
            key=f"approve-{thread_key}",
            use_container_width=True,
        )


    # ========================= Reject ==============================

    with col2:

        reject = st.button(
            "❌ Reject",
            key=f"reject-{thread_key}",
            use_container_width=True,
        )


    # ======================= Resume Graph ==========================

    if approve or reject:

        decision = (
            "yes"
            if approve
            else "no"
        )


        CONFIG = get_config(
            thread_key
        )


        with st.chat_message(
            "assistant"
        ):

            status_holder = {
                "box": None
            }


            resumed_chunks = []


            for (
                message_chunk,
                _
            ) in chatbot.stream(

                Command(
                    resume=decision
                ),

                config=CONFIG,

                stream_mode="messages",
            ):


                # ================= Tool Message ===================

                if isinstance(
                    message_chunk,
                    ToolMessage
                ):

                    tool_name = getattr(
                        message_chunk,
                        "name",
                        "tool"
                    )


                    if (
                        status_holder[
                            "box"
                        ]
                        is None
                    ):

                        status_holder[
                            "box"
                        ] = st.status(

                            f"🔧 Using `{tool_name}` …",

                            expanded=True,
                        )


                # ================= AI Message ======================

                if isinstance(
                    message_chunk,
                    AIMessage
                ):

                    if message_chunk.content:

                        resumed_chunks.append(
                            message_chunk.content
                        )


            resumed_message = "".join(
                resumed_chunks
            )


            # ================= Tool Finished ======================

            if (
                status_holder["box"]
                is not None
            ):

                status_holder[
                    "box"
                ].update(

                    label="✅ Tool finished",

                    state="complete",

                    expanded=False,
                )


            # ================= AI Response =========================

            if resumed_message:

                st.markdown(
                    resumed_message
                )


        # ================= Save AI Message ==========================

        if resumed_message:

            st.session_state[
                "message_history"
            ].append(

                {
                    "role": "assistant",
                    "content": resumed_message,
                }

            )


        # ================= Clear Interrupt ==========================

        st.session_state[
            "pending_interrupt"
        ] = None


        st.rerun()


# ============================ User Input =============================

user_input = st.chat_input(
    "Ask about your document or use tools"
)


if user_input:

    # ======================== Save User Message ====================

    st.session_state[
        "message_history"
    ].append(

        {
            "role": "user",
            "content": user_input,
        }

    )


    # ======================== Display User =========================

    with st.chat_message(
        "user"
    ):

        st.markdown(
            user_input
        )


    # ======================== Config ===============================

    CONFIG = get_config(
        thread_key
    )


    # ======================== Run Graph =============================

    with st.chat_message(
        "assistant"
    ):

        status_holder = {
            "box": None
        }


        response_chunks = []


        for (
            message_chunk,
            _
        ) in chatbot.stream(

            {
                "messages": [
                    HumanMessage(
                        content=user_input
                    )
                ],

                "thread_id": thread_key,
            },

            config=CONFIG,

            stream_mode="messages",
        ):


            # ================= Tool Message ========================

            if isinstance(
                message_chunk,
                ToolMessage
            ):

                tool_name = getattr(
                    message_chunk,
                    "name",
                    "tool"
                )


                if (
                    status_holder["box"]
                    is None
                ):

                    status_holder[
                        "box"
                    ] = st.status(

                        f"🔧 Using `{tool_name}` …",

                        expanded=True,
                    )


                else:

                    status_holder[
                        "box"
                    ].update(

                        label=(
                            f"🔧 Using `{tool_name}` …"
                        ),

                        state="running",

                        expanded=True,
                    )


            # ================= AI Message ===========================

            if isinstance(
                message_chunk,
                AIMessage
            ):

                if message_chunk.content:

                    response_chunks.append(
                        message_chunk.content
                    )


        # ================= Combine Response ========================

        ai_message = "".join(
            response_chunks
        )


        # ================= Tool Finished ============================

        if (
            status_holder["box"]
            is not None
        ):

            status_holder[
                "box"
            ].update(

                label="✅ Tool finished",

                state="complete",

                expanded=False,
            )


        # ================= Display AI ===============================

        if ai_message:

            st.markdown(
                ai_message
            )


    # ================================================================
    # Check if LangGraph paused at an interrupt
    # ================================================================

    graph_state = chatbot.get_state(
        CONFIG
    )


    if graph_state.tasks:

        interrupts = (
            graph_state
            .tasks[0]
            .interrupts
        )


        if interrupts:

            interrupt_value = (
                interrupts[0].value
            )


            # ================= Store HITL ==========================

            st.session_state[
                "pending_interrupt"
            ] = {

                "message": str(
                    interrupt_value
                )

            }


            # ================= Rerun ================================

            st.rerun()


    # ================================================================
    # Only save normal AI response if there wasn't an interrupt
    # ================================================================

    if ai_message:

        st.session_state[
            "message_history"
        ].append(

            {
                "role": "assistant",
                "content": ai_message,
            }

        )


    # ======================== Document Info =========================

    doc_meta = thread_document_metadata(
        thread_key
    )


    if doc_meta:

        st.caption(

            f"📄 Document indexed: "
            f"{doc_meta.get('filename')} "
            f"({doc_meta.get('chunks')} chunks, "
            f"{doc_meta.get('documents')} pages)"

        )


# ============================ Divider ===============================

st.divider()