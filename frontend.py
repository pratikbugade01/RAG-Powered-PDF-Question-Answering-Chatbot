import streamlit as st
import requests
import uuid

# Backend URL
BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Document Based Question Answering Chatbot.", layout="centered")

st.title("📄 Chat With Documents.")
st.write("Ask questions on your document.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

doc = st.file_uploader("Upload your document",type=["PDF"],max_upload_size=20)

if st.button("Load Document"):
    if doc:

        if doc.size > 20 * 1024 * 1024:
            st.error("File too large. Maximum size is 20MB.")
            st.stop()

        with st.spinner("Loading Document..."):
            response = requests.post(
                f"{BASE_URL}/upload-pdf",
                files={
                    "file":doc
                },
                data={"session_id": st.session_state["session_id"]}
            )

            data = response.json()

            if data["success"]:             
                st.success(data["message"])
                st.session_state["loaded"] = True
                st.session_state["chat_history"] = []
                
            else:
                st.error(data["message"])
    else:
        st.warning("Please upload a valid document")
    


# -------------------------
# Ask Question Section
# -------------------------
if st.session_state.get("loaded", False):

    st.subheader("Ask a Question")

    question = st.text_input("Enter your question.")

    if st.button("Get Answer"):
        if question:
            with st.spinner("Thinking..."):
                try:                                    
                    response = requests.post(
                        f"{BASE_URL}/ask",
                        json={
                            "session_id": st.session_state["session_id"],
                            "question": question,
                        }
                    )
                    answer = response.json()["answer"]

                    st.session_state["chat_history"].append({
                        "question": question,
                        "answer": answer
                    })

                except requests.exceptions.ConnectionError:                          
                    st.error("Cannot connect to backend. Make sure FastAPI is running.")
                except Exception as e:
                    st.error(f"Something went wrong: {e}")

    for chat in reversed(st.session_state["chat_history"]):  # newest first
        st.markdown(f"**Question :** {chat['question']}")
        st.markdown(f"**Answer :** {chat['answer']}")
        st.divider()