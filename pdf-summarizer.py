import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_anthropic import ChatAnthropic


import google.generativeai as genai

if 'model_name' not in st.session_state:
    st.session_state['model_name'] = 'Please select the LLM model to use'


def change_name(name):
    st.session_state['model_name'] = name


# Sidebar contents
with st.sidebar:
    st.title('🤗💬 Pdf reader and QA with LLM')
    st.markdown('''
    ## About
    This app is an LLM-powered using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) ChatGPT model
    - [Gemini](https://ai.google.dev/docs/gemini_api_overview) Gemini-Pro model
    ''')
    st.write('Made by WinEisEis')

load_dotenv()


def main():
    st.header("Chat with pdf")

    # Upload a pdf file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")

        # LLM choices
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            b1 = st.button(label='ChatGPT (OpenAI)', on_click=change_name, args=['Select ChatGPT (OpenAI)'],
                           key='OpenAI')
        with col2:
            b2 = st.button(label='Gemini', on_click=change_name, args=['Select Gemini'], key='Gemini')

        with col3:
            b3 = st.button(label='Claude 3', on_click=change_name, args=['Select Claude 3'], key='Claude 3')

        if query and b1:

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=3000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text=text)

            # Convert the chunks of text into embeddings to form the knowledge base
            embeddings = OpenAIEmbeddings()
            knowledgeBase = FAISS.from_texts(chunks, embeddings)
            llm = OpenAI(model_name="gpt-4-0125-preview",
                         temperature=0.3,
                         )

            print(f"Using LLM: {llm} model")
            docs = knowledgeBase.similarity_search(query=query)

            with st.spinner("Processing ChatGPT (OpenAI)"):
                chain = load_qa_chain(llm=llm, chain_type="stuff")

                with get_openai_callback() as cost:
                    response = chain.run(input_documents=docs, question=query)
                    print(cost)
                st.write(response)
                st.success("Done")

        elif query and b2:

            genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=10000,
                chunk_overlap=500,
                length_function=len
            )
            chunks = text_splitter.split_text(text=text)
            knowledgeBase = FAISS.from_texts(chunks, embeddings)
            llm = ChatGoogleGenerativeAI(model="gemini-pro", temperatue=0, convert_system_message_to_human=True)
            print(f"Using LLM: {llm} model")

            docs = knowledgeBase.similarity_search(query=query)

            # Define the prompt
            prompt_template = """
            Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
            provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
            Context:\n {context}?\n
            Question: \n{question}\n

            Answer:
            """
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            with st.spinner("Processing Gemini"):
                chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)
                response = chain.run(input_documents=docs, question=query)

                print(response)
                st.write(response)
                st.success("Done")

        elif query and b3:

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=3000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text=text)

            # Convert the chunks of text into embeddings to form the knowledge base
            embeddings = OpenAIEmbeddings()
            knowledgeBase = FAISS.from_texts(chunks, embeddings)
            llm = ChatAnthropic(model='claude-3-opus-20240229')

            print(f"Using LLM: {llm} model")
            docs = knowledgeBase.similarity_search(query=query)

            with st.spinner("Processing Claude"):
                chain = load_qa_chain(llm=llm, chain_type="stuff")

                with get_openai_callback() as cost:
                    response = chain.run(input_documents=docs, question=query)
                    print(cost)
                st.write(response)
                st.success("Done")
        else:
            st.write(st.session_state['model_name'])


if __name__ == '__main__':
    main()
    # Prompt for testing = 'รูปแบบการให้บริการซ่อมบำรุงรถยนต์ มีกี่ Mode และแต่ละ Mode คืออะไรบ้าง'
