import os
import time
import streamlit as st
from streamlit_chat import message
from chat_ollama import ChatPDF
from utils import save_audio_wav, audio_to_text, text_to_audio
from dotenv import load_dotenv
import whisper

st.set_page_config(page_title="EQUIPE 3", layout="wide")
header = st.container()
col1, col2 = st.columns(2)
PDF_FOLDER = "./data/papers"

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


def display_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        if is_user:
            message(msg, is_user=is_user, key=str(i), seed=100)
        else:
            message(msg, is_user=is_user, key=str(i), seed=11)
    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    if st.session_state["user_input_audio"]:
        save_audio_wav(st.session_state["user_input_audio"])
        time.sleep(2)

        st.session_state["user_input"] = audio_to_text(
            st.session_state["whisper_model"], "./data/input_audio/recorded_audio.wav"
        )

        user_text = st.session_state["user_input"]

        with st.session_state["thinking_spinner"], st.spinner("Pensando..."):
            agent_text = st.session_state["assistant"].ask_com_db(user_text)
            st.session_state["last_model_message"] = agent_text

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))


def read_and_save_files_from_folder():
    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    # Lista todos os arquivos PDF na pasta especificada
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    print(pdf_files)
    for pdf_file in pdf_files:
        file_path = os.path.join(PDF_FOLDER, pdf_file)

        t0 = time.time()
        st.session_state["assistant"].ingest(file_path)
        t1 = time.time()

    st.session_state["messages"].append(
        (
            f"Arquivo {pdf_file} carregado em {t1 - t0:.2f} segundos",
            False,
        )
    )


def page():
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["assistant"] = ChatPDF()
        st.session_state["last_model_message"] = ""
        st.session_state["mount_chromadb"] = read_and_save_files_from_folder()
        st.session_state["whisper_model"] = whisper.load_model("medium")
    with header:
        st.header("ChatAudio")

    with col1:
        st.session_state["ingestion_spinner"] = st.empty()
        st.subheader("Audio Chat")
        st.audio_input("", key="user_input_audio", on_change=process_input)
        if st.session_state["last_model_message"]:
            text_to_audio(
                text=st.session_state["last_model_message"],
                language="en-us",
                path_to_save="./data/output_audio/audio_returned.wav",
            )
            st.audio(data="./data/output_audio/audio_returned.wav", autoplay=True)
    with col2:
        display_messages()


if __name__ == "__main__":
    page()
