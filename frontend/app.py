import requests
import streamlit as st

BACKEND_URL = "http://localhost:8001"

st.set_page_config(
    page_title="SOVA TEAMS",
    layout="wide"
)

st.title("🧠 SOVA AI Knowledge System")

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("📂 Project")

project = st.sidebar.text_input(
    "Project name",
    value="SOVA"
)

st.sidebar.divider()

# =====================================================
# FILE UPLOAD
# =====================================================

st.sidebar.header("📄 Upload PDF")

uploaded_file = st.sidebar.file_uploader(
    "Choose PDF file",
    type=["pdf"]
)

if uploaded_file:

    if st.sidebar.button("Upload"):

        with st.spinner("Uploading and indexing document..."):

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file,
                    "application/pdf"
                )
            }

            data = {
                "project": project
            }

            response = requests.post(
                f"{BACKEND_URL}/upload",
                files=files,
                data=data
            )

            if response.status_code == 200:

                st.sidebar.success("✅ File uploaded")

            else:

                st.sidebar.error("❌ Upload failed")

# =====================================================
# CHAT HISTORY
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# =====================================================
# DISPLAY MESSAGES
# =====================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# =====================================================
# CHAT INPUT
# =====================================================

question = st.chat_input(
    "Ask question about your documents..."
)

if question:

    # User message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Assistant message
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            response = requests.post(
                f"{BACKEND_URL}/ask",
                json={
                    "question": question,
                    "project": project
                }
            )

            if response.status_code == 200:
                result = response.json()

                answer = result["answer"]

                sources = result.get("sources", [])

                st.markdown(answer)
                if sources:
                    st.divider()
                    st.markdown("### Sources")
                    for source in sources:
                        st.markdown(f"- {source}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            else:

                st.error("❌ Error while generating response")