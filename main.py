from util import *
from streamlit_option_menu import option_menu
from streamlit_extras.metric_cards import style_metric_cards
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="PDF Genie", page_icon=":robot_face:", layout="centered")

# --- SETUP SESSION STATE VARIABLES ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = False
if "response" not in st.session_state:
    st.session_state.response = None
if "prompt_activation" not in st.session_state:
    st.session_state.prompt_activation = False
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = None
if "prompt" not in st.session_state:
    st.session_state.prompt = False
if "total_token" not in st.session_state:
    st.session_state.total_token = 0
if "successful_requests" not in st.session_state:
    st.session_state.successful_requests = 0
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0

# --- SIDEBAR CONFIGURATION ---
openai_api_key = sidebar_api_key_configuration()

# --- MAIN PAGE CONFIGURATION ---
st.title("PDF Genie :robot_face:")
st.write("*Interrogate Documents :books:, Ignite Insights: AI at Your Service*")

# ---- NAVIGATION MENU -----
selected = option_menu(
    menu_title=None,
    options=["PDF Genie", "Analytics", "Reference", "About"],
    icons=["robot", "bar-chart-fill", "bi-file-text-fill", "app"],  # https://icons.getbootstrap.com
    orientation="horizontal",
)

llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only. If question is not within the context, do not try to answer
    and respond that the asked question is out of context or something similar.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    Questions: {input}
    """
)
# ----- SETUP PDF GENIE MENU ------
if selected == "PDF Genie":
    st.subheader("Upload PDF(s)")
    pdf_docs = st.file_uploader("Upload your PDFs", type=['pdf'], accept_multiple_files=True,
                                disabled=not st.session_state.prompt_activation)
    process = st.button("Process", type="primary", key="process", disabled=not pdf_docs)

    if process:
        with st.spinner("Processing ..."):
            st.session_state.vector_store = create_vectorstore(openai_api_key, pdf_docs)
            st.session_state.prompt = True
            st.success('Database is ready')

    st.divider()

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    container = st.container(border=True)
    if question := st.chat_input(placeholder='Enter your question related to uploaded document',
                                 disabled=not st.session_state.prompt):
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        with st.spinner('Processing...'):
            st.session_state.response = get_llm_response(llm, prompt, question)
            st.session_state.messages.append({"role": "assistant", "content": st.session_state.response['answer']})
            st.chat_message("assistant").write(st.session_state.response['answer'])

# ----- SETUP ANALYTICS MENU ------
if selected == "Analytics":
    st.title("Analytics")
    st.write("*This page shows the analytics like total token used, total cost etc.*")
    st.divider()
    # Create metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Token", f"{st.session_state.total_token}")
    col2.metric("Total Requests", f"{st.session_state.successful_requests}")
    col3.metric("Total Cost (USD)", f"${round(st.session_state.total_cost, 6)}")
    style_metric_cards()

# ----- SETUP REFERENCE MENU ------
if selected == "Reference":
    st.title("Reference & Context")
    if st.session_state.response is not None:
        for i, doc in enumerate(st.session_state.response["context"]):
            with st.expander(f'Reference # {i + 1}'):
                st.write(doc.page_content)

# ----- SETUP ABOUT MENU ------
if selected == "About":
    with st.expander("About this App"):
        st.markdown(''' This app allows you to chat with your PDF documents. It has following functionality:

    - Allows to chat with multiple PDF documents
    - Display the response context and document reference
    - Display the analytics
        ''')
    with st.expander("Which Large Language models are supported by this App?"):
        st.markdown(''' This app supports the following LLMs:

    - Chat Model -- OpenAI gpt-3.5-turbo
    - Embeddings -- OpenAI Text-embedding-ada-002-v2
        ''')

    with st.expander("Which library is used for vectorstore?"):
        st.markdown(''' This app supports the FAISS for AI similarity search and vectorstore:
        ''')

    with st.expander("Where to get the source code of this app?"):
        st.markdown(''' Source code is available at:
    *  https://github.com/mzeeshanaltaf/genai-rag-openai-faiss
        ''')
    with st.expander("Whom to contact regarding this app?"):
        st.markdown(''' Contact [Zeeshan Altaf](zeeshan.altaf@gmail.com)
        ''')
