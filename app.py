import streamlit as st
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Krishi Sarthi - Your Agricultural Assistant",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar Content ---
with st.sidebar:
    st.markdown("🌱")
    st.markdown("## Krishi Sarthi")
    st.markdown("Your AI-powered Agricultural Assistant")
    st.markdown("---")
    st.subheader("Navigation")
    page_selection = st.radio(
        "Go to",
        ("Query Assistant", "About Krishi Sarthi", "Data Information", "Settings"),
        index=0,
    )
    st.markdown("---")
    st.info("💡 **Offline-capable, Local-first AI** for farmers.")

# --- Simulate a right-side panel using columns ---
main_col, right_col = st.columns([3, 1], gap="large")

# --- Right Panel: Controls for Temperature, Top-K, etc. ---
with right_col:
    st.markdown("### Retrieval & Model Settings")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
    top_k = st.number_input("Top K Retrievals", min_value=1, max_value=20, value=5, step=1)
    st.markdown("---")

# --- Backend Setup ---
try:
    from langchain_ollama.llms import OllamaLLM
    from langchain_core.prompts import ChatPromptTemplate
    from vector import retriever

    model = OllamaLLM(model="gemma3:1b", temperature=temperature)
    template = """
        You are an expert in answering agricultural questions from the Kisan Call Center (KCC) dataset, which is operated by the Government of India to support Indian farmers with reliable, localized advice[1][3][5]. Your role is to provide accurate, actionable, and legally compliant responses based on data and best practices recognized by Indian authorities.

        Here are some relevant KCC data chunks: {kcc_chunks}

        Here is the question to answer: {question}

        Guidelines and Guardrails:
        - Use only the information from the provided KCC data chunks and your verified agricultural knowledge as recognized by Indian agricultural authorities.
        - Ensure your answer is factually correct, practical, and tailored to the context of Indian agriculture and the needs of Indian farmers.
        - If the answer involves chemical usage (fertilizers, pesticides), specify only those approved by the Government of India and include appropriate safety precautions.
        - Do not provide medical, veterinary, or legal advice beyond what is present in the dataset or officially recognized by Indian government sources.
        - Do not make guarantees about outcomes; instead, offer best practices or likely results based on Indian agricultural experience.
        - If the question cannot be answered with the available data, clearly state the limitation and suggest the user contact a local agricultural officer, Kisan Call Center, or government extension service.
        - Never fabricate information, speculate, or provide unverified advice.
        - Always write in en-India unless the user requests otherwise.
        - Do not include personal opinions or promotional content.
        - Respect user privacy; do not request or infer personal data.

        Legal and Compliance Requirements:
        - Adhere strictly to Indian government agricultural guidelines, standards, and safety regulations.
        - Do not recommend banned or restricted substances or practices as per Indian law.
        - Ensure advice complies with Indian laws, government schemes, and environmental regulations.
        - Reference only officially recognized sources and practices as reflected in the KCC data and Indian government advisories.

        Format your response as follows:
        1. **Summary**: Briefly state the main advice or answer.
        2. **Details**: Provide step-by-step recommendations or supporting information.
        3. **Precautions/Legal Notes**: List any safety, legal, or regulatory considerations relevant to the advice, specifically referencing Indian government guidelines where applicable.
        4. **Further Assistance**: If needed, suggest contacting local agricultural officers, the Kisan Call Center, or Indian government extension services for complex or unresolved issues.

        Current date: Sunday, May 25, 2025, 7:20 PM IST


    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    llm_retriever_ready = True
except ImportError:
    llm_retriever_ready = False
except Exception:
    llm_retriever_ready = False

# --- LangChain DuckDuckGo Search Tool ---
try:
    from langchain_community.tools import DuckDuckGoSearchRun
    duckduckgo_available = True
except ImportError:
    duckduckgo_available = False

# --- Helper Functions ---
def get_kcc_response(query):
    if llm_retriever_ready:
        try:
            kcc_chunks = retriever.invoke(query, top_k=top_k)
            if not kcc_chunks:
                return None
            result = chain.invoke({"kcc_chunks": kcc_chunks, "question": query})
            return {"source": "KCC Dataset", "answer": result}
        except Exception:
            pass



def perform_internet_search(query):
    if duckduckgo_available:
        try:
            search = DuckDuckGoSearchRun()
            result = search.run(query)
            return {
                "source": "DuckDuckGo Search",
                "answer": result
            }
        except Exception as e:
            return {
                "source": "DuckDuckGo Search (Error)",
                "answer": f"An error occurred while searching DuckDuckGo: {e}"
            }
    else:
        time.sleep(3)
        return {
            "source": "Internet Search (Fallback)",
            "answer": f"No relevant information found in local data. Here is a general result from the internet for '{query}': Modern agricultural practices focus on soil health, water conservation, and integrated pest management."
        }

# --- Main Page Content in Main Column ---
with main_col:
    if page_selection == "Query Assistant":
        st.title("🌱 Krishi Sarthi ")
        st.markdown("---")
        st.markdown("""
            **Get agricultural advice from the Kisan Call Center (KCC) dataset.**
            Ask any question about your crops, pest control, weather, or any other agriculture-related query.
        """)

        user_query = st.text_area(
            "Type your question here:",
            placeholder="e.g., How to control pests in paddy?",
            height=100
        )
        if st.button("Get Advice"):
            if user_query:
                with st.spinner("Searching local knowledge base..."):
                    kcc_result = get_kcc_response(user_query)

                if kcc_result:
                    st.success("Advice from KCC Data:")
                    st.write(kcc_result['answer'])
                    if "Simulated" in kcc_result['source']:
                        st.info("Information found from local KCC dataset (Simulated).")
                    else:
                        st.success("Information found from local KCC dataset.")
                else:
                    st.warning("No direct context found in local data. Searching on the internet...")
                    with st.spinner("Searching on the internet..."):
                        fallback_result = perform_internet_search(user_query)
                    st.info("Advice from Internet:")
                    st.write(fallback_result['answer'])
                    st.info("This information was obtained via internet search.")
            else:
                st.error("Please enter a question.")

    elif page_selection == "About Krishi Sarthi":
        st.title("ℹ️ About Krishi Sarthi")
        st.markdown("---")
        st.markdown("""
        **Krishi Sarthi** is an offline-capable, local-first AI application designed to help farmers get agricultural advice from the Kisan Call Center (KCC) dataset.

        **Key Features:**
        * **Local AI Model:** Utilizes an open-source language model via the Ollama, running entirely offline.
        * **Retrieval-Augmented Generation (RAG):** Performs semantic search in the KCC dataset for relevant information.
        * **Fallback Internet Search:** If no context is found in local data, it performs an internet search.
        * **User-Friendly Interface:** Designed for intuitive querying and clear answer display.

        This project aims to empower farmers by providing reliable and accessible agricultural information.
        """)

    elif page_selection == "Data Information":
        st.title("📊 Data Information")
        st.markdown("---")
        st.markdown("""
        **Kisan Call Center (KCC) Dataset:**
        This application utilizes the public KCC dataset, which contains a vast collection of agricultural questions and answers.

        **Data Preprocessing:**
        * Data has been cleaned, normalized, and split into logical "document" chunks of Q&A pairs.
        * A sentence-transformer model bge-large-en has been used to generate embeddings (vector representations) for each chunk.
        * These embeddings are stored in a lightweight vector database ChromaDB.

        **Data Flow Summary:**
        1.  **Data Ingestion:** Loads raw KCC CSV data.
        2.  **Preprocessing:** Cleans, normalizes, and chunks Q&A pairs.
        3.  **Embedding Generation:** Encodes chunks into vectors.
        4.  **Vector Store Ingestion:** Indexes embeddings in ChromaDB
        """)

    elif page_selection == "Settings":
        st.title("⚙️ Settings")
        st.markdown("---")
        st.markdown("""
        Here you can configure various aspects of the app.

        **LLM Model:**
        * Currently used model: `gemma3:1b` (via Ollama)
        * In the future, you could add options here to select different local LLM models.

        **Vector Database:**
        * Currently configured: ChromaDB
      

        **Response Relevance Threshold:**
        * (This feature is currently simulated and would require actual implementation.)

        ---
        **Note:**
        This is a demo application. Actual settings would require backend integration to interface with your local LLM and vector database setup.
        """)
