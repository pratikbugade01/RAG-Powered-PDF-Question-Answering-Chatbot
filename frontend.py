import streamlit as st
import requests

# Backend URL
BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Document Based Question Answering Chatbot.", layout="centered")

st.title("🎥 Chat With Documents.")
st.write("Ask questions on your document.")


doc = st.file_uploader("Upload your document")
typeofdoc = st.selectbox(label="select Document type",options=["PDF","Text","doc"])

if st.button("Load Document"):
    if doc:
        with st.spinner("Loading Document..."):
            response = requests.post(
                f"{BASE_URL}/upload-pdf",
                files={
                    "file":doc
                }
            )

            data = response.json()

            if data["success"]:             
                st.success(data["message"])
                st.session_state["loaded"] = True
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
                response = requests.post(
                    f"{BASE_URL}/ask",
                    json={"question": question}
                )

                answer = response.json()["answer"]

                st.markdown("### 💡 Answer:")
                st.write(answer)
        else:
            st.warning("Please enter a question")