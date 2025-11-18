import streamlit as st
from streamlit_chat import message
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain_community.vectorstores import FAISS
from LLM import Generator
import re
from torch.nn import Sigmoid

@st.cache_resource()
def load_cross_encoder_model():
    cross_encoder_model = CrossEncoder("./models/retrieval/cross-electra-ms-marco-german-uncased", default_activation_function=Sigmoid())
    return cross_encoder_model

@st.cache_resource()
def load_retriever():
    db = FAISS.load_local("./models/vectorstore/store-multilingual-e5-small", embeddings, allow_dangerous_deserialization= True)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 7})
    return retriever

@st.cache_resource()
def load_embeddings_model():
    embeddings = HuggingFaceEmbeddings(model_name = "./models/retrieval/RAG-multilingual-e5-small")
    return embeddings

@st.cache_resource()
def load_llm():
    llm = Generator(True)
    return llm

def rerank(query, docs):
    hits = []
    for doc in docs:
        js = {
                'text': doc.page_content,
                'meta': doc.metadata
            }
        hits.append(js)

    sentence_pairs = [[query, hit["text"]] for hit in hits]
    similarity_scores = cross_encoder_model.predict(sentence_pairs)
    
    for idx in range(len(hits)):
        hits[idx]["cross-encoder_score"] = similarity_scores[idx]

    hits = sorted(hits, key=lambda x: x["cross-encoder_score"], reverse=True)
    return hits

def processDocument(data):
    # Die Header 1-4 und das source-document aus dem meta-Feld extrahieren
    headers = []
    for i in range(1, 5):  # Für Header 1 bis Header 4
        header_key = f"Header {i}"
        if header_key in data['meta']:
            headers.append(data['meta'][header_key])

    # Text aus dem JSON extrahieren
    text = processText(data.get('text', ''))
    score = data.get("cross-encoder_score")

    # Quelle (source-document) extrahieren
    source_document = data['meta'].get('source-document', 'Unbekannte Quelle')

    # Den finalen String aufbauen
    final_string = "\n\n".join(headers) + "\n\n" + text + "\n\nQuelle: " + source_document + f"\n | Relevance Score: {round(score*100,2)}%\n\n" 

    # Ergebnis ausgeben
    return final_string

def processText(text):
    # Regex für das Entfernen von ![Image](source) und ![Icon](source)
    text = re.sub(r"!\[(?:Icon)]\([^\)]+\)", "[Icon]", text)
    text = re.sub(r"!\[(?:Image|Icon)]\([^\)]+\)", "", text)

    # Regex für das Entfernen von "Abbildung zahl/zahl"
    text = re.sub(r"Abbildung\s\d+/\d+", "", text)

    # Optional: Entferne überflüssige Leerzeichen, falls nötig
    text = re.sub(r"\s+", " ", text).strip()

    return text

# Retrieval logic
def retrieve(query: str):
    documents = retriever.invoke(query)
    reranked = rerank(query, documents)
    cut = reranked[:k]
    return "".join([processDocument(el) for el in cut])


def generate_response(query, context):
    answer = llm.query(query, context, False)
    for chunk in answer:
        text = chunk["choices"][0]["text"]
        yield text


# Streamlit app
st.set_page_config(page_title="LocalRAG Prototype", layout="wide")

st.title("LocalRAG Prototype")

k = st.selectbox("Wähle wie viele Kontext Teile verwendet werden sollen (1-2 empfohlen):", list(range(1, 8)), index=0)

with st.chat_message("assistant"):
    st.markdown("Hi 👋 Ich bin der LocalRAG zur Beantwortung deiner Fragen zu ** 4**. Was möchtest du wissen?")

if "cross_encoder" not in st.session_state.keys():
    st.session_state["cross_encoder"] = load_cross_encoder_model()
cross_encoder_model = st.session_state["cross_encoder"]

if "embeddings" not in st.session_state.keys():
    st.session_state["embeddings"] = load_embeddings_model()
embeddings = st.session_state["embeddings"]

if "retriever" not in st.session_state.keys():
    st.session_state["retriever"] = load_retriever()
retriever = st.session_state["retriever"]

if "llm" not in st.session_state.keys():
    st.session_state["llm"] = load_llm()
llm = st.session_state["llm"]



# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



# User input section
if input := st.chat_input("Was willst du wissen?"):
    with st.chat_message("user"):
        st.markdown(input)
    st.session_state.messages.append({"role": "user", "content": input})

    docs = retrieve(input)
    st.info(f"Kontext zur Beantwortung deiner Frage:\n\n {docs}", icon="ℹ️")

    with st.chat_message("assistant"):
        response = st.write_stream(generate_response(input, docs))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

