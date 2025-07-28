import os
from io import BytesIO

import fitz  # PyMuPDF
import streamlit as st
from dotenv import load_dotenv
from gtts import gTTS
from mistralai import Mistral, DocumentURLChunk

# -----------------------------------------------------------------------------
# Configurações de página
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Chat com PDF via Mistral", layout="wide")

# -----------------------------------------------------------------------------
# Estilo visual
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    body {
        background-color: #0f1117;
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
    }
    .main-header {
        background: linear-gradient(90deg, #4f00ff 0%, #1e1b4b 100%);
        border-radius: 12px;
        padding: 2rem 2rem 1rem 2rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    .main-header h1 {
        font-size: 2.2rem;
        color: white;
        margin-bottom: 0.4rem;
    }
    .main-header p {
        font-size: 1rem;
        color: #ddddff;
    }
    .logo-container {
        display: flex;
        justify-content: center;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Cabeçalho
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.markdown('<div class="logo-container">', unsafe_allow_html=True)
# st.image("static/logo_skyone.png", use_container_width=False, width=100)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<h1>Agente de Inteligência Artificial com PDF</h1>", unsafe_allow_html=True)
st.markdown(
    "<p>Converse em linguagem natural com documentos PDF escaneados ou nativos.</p>",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Obtenção segura da API Key (Cloud + Local)
# -----------------------------------------------------------------------------
def get_api_key() -> str | None:
    try:
        return st.secrets["MISTRAL_API_KEY"]
    except Exception:
        pass
    load_dotenv()
    return os.getenv("MISTRAL_API_KEY")


api_key = get_api_key()
if not api_key:
    st.error("❌ MISTRAL_API_KEY não encontrada. Configure em st.secrets (Cloud) ou .env (Local).")
    st.stop()

client = Mistral(api_key=api_key)

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
use_audio = st.sidebar.checkbox("🔊 Ativar resposta por voz")
force_ocr = st.sidebar.checkbox("📸 Forçar OCR mesmo com texto detectável")

# -----------------------------------------------------------------------------
# Histórico da conversa
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------------------------------------------------------
# Upload do PDF
# -----------------------------------------------------------------------------
pdf_file = st.file_uploader("Envie o PDF:", type=["pdf"])

if pdf_file:
    st.success(f"✅ Arquivo '{pdf_file.name}' carregado com sucesso.")

    try:
        with fitz.open(stream=pdf_file.getvalue(), filetype="pdf") as doc:
            extracted_text = "\n".join([page.get_text() for page in doc])
    except Exception as e:
        st.error(f"Erro ao ler PDF com PyMuPDF: {e}")
        extracted_text = ""

    if force_ocr or len(extracted_text.strip()) < 30:
        st.warning("📷 OCR necessário. Enviando para o Mistral...")

        try:
            uploaded = client.files.upload(
                file={"file_name": pdf_file.name, "content": pdf_file.getvalue()},
                purpose="ocr",
            )
            signed_url = client.files.get_signed_url(file_id=uploaded.id, expiry=1)

            with st.spinner("🔎 Executando OCR via Mistral..."):
                result = client.ocr.process(
                    document=DocumentURLChunk(document_url=signed_url.url),
                    model="mistral-ocr-latest",
                )
                content_text = "\n\n".join([p.markdown for p in result.pages])
        except Exception as e:
            st.error(f"❌ Erro no OCR via Mistral: {e}")
            st.stop()
    else:
        st.info("📝 Texto nativo detectado no PDF. OCR não necessário.")
        content_text = extracted_text

    if not any("Conteúdo do PDF:" in msg["content"] for msg in st.session_state.messages):
        st.session_state.messages.append(
            {
                "role": "system",
                "content": "Você é um assistente que responde com base no conteúdo de um PDF fornecido.",
            }
        )
        st.session_state.messages.append(
            {"role": "user", "content": f"Conteúdo do PDF:\n\n{content_text[:20000]}"}
        )

    user_input = st.text_input("Faça uma pergunta sobre o PDF:")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("💬 Pensando..."):
            try:
                response = client.chat.complete(
                    model="mistral-small-latest",
                    messages=st.session_state.messages,
                )
                reply = response.choices[0].message.content
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.success("✅ Resposta gerada!")

                if use_audio:
                    try:
                        tts = gTTS(reply, lang="pt")
                        audio_fp = BytesIO()
                        tts.write_to_fp(audio_fp)
                        audio_fp.seek(0)
                        st.audio(audio_fp, format="audio/mp3")
                    except Exception as e:
                        st.warning(f"⚠️ Erro ao gerar áudio: {e}")
            except Exception as e:
                st.error(f"❌ Erro ao gerar resposta: {e}")

    for msg in st.session_state.messages[2:]:
        role = "Você" if msg["role"] == "user" else "Assistente"
        st.markdown(f"**{role}:** {msg['content']}")
else:
    st.info("📄 Envie um PDF para começar.")
