import sys
sys.modules['torch.classes'] = None

import os
import uuid
import nest_asyncio
import streamlit as st
import pandas as pd
from langchain_core.documents import Document

from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate, MessagesPlaceholder, ChatPromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains import create_history_aware_retriever
from langchain_openai import ChatOpenAI
from filteration import filter_by_price, filter_by_keyword, filter_by_specifications, filter_by_purpose


nest_asyncio.apply()

os.environ["OPENAI_API_KEY"] = "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"  # Add your key
os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"

st.title("💻 AI Powered Laptop Chatbot")

uploaded_file = st.file_uploader("Upload laptop dataset (flipkart_laptop_cleaned.csv)", type=["csv"])

df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

user_query = st.text_input("🔍 Ask for laptop recommendations")
follow_up = st.text_input("💬 Ask a follow-up question like 'Compare 1st and 2nd'")

if st.button("🔄 Reset Session"):
    st.session_state.clear()
    st.experimental_rerun()


def setup_recommender_chain(documents):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(documents, embeddings)
    retriever = db.as_retriever()

    llm = ChatOpenAI(model="llama3-70b-8192")

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given chat history and the question, rephrase to a standalone query if needed."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         """You are an expert laptop recommender. ONLY use the context below (Top 5 laptops) for any answers.

Only recommend LAPTOPS. Ignore mobile phones, tablets, or any unrelated product even if the question asks for it.

Refer to laptops using numbers 1 to 5. Do not go outside this list.

Provide structured comparison and insights when asked.

Context:
{context}
"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )


#if df is not None:
   #    df["Brand"].astype(str)
      #  + ", " + df["Product Name"].astype(str)
        #+ ", ₹" + df["Price"].astype(str)
        #+ ", Processor: " + df["Processor"].astype(str)
        #+ ", RAM: " + df["RAM"].astype(str)
        #+ ", Storage: " + df["Storage"].astype(str)
       # + ", Display: " + df["Display"].astype(str)
      #  + ", Specs: " + df["Specifications"].astype(str)
        #+ ", OS: " + df["OS"].astype(str)
       # + ", URL: " + df["Product URL"].astype(str)
    #)

if user_query:
    st.write("🧪 User Query:", user_query)  # Debugging output
    
    #if not is_laptop_related(user_query):
        #st.stop()  # Stop further processing if query is unrelated
    

    # 🔁 Reset chain if the query changes
    if "last_query" not in st.session_state or st.session_state["last_query"] != user_query:
            st.session_state.pop("chat_chain", None)
            st.session_state["last_query"] = user_query

            # 🧹 Optional: clear old recommendations and documents
            st.session_state.pop("recommended_laptops", None)

    # 📊 Filtering
    filtered_df = filter_by_price(df, user_query)
    filtered_df = filter_by_keyword(filtered_df,user_query) 
    filtered_df = filter_by_specifications(filtered_df, user_query)
    filtered_df = filter_by_purpose(filtered_df, user_query)

    if filtered_df.empty:
        st.warning("⚠️ No laptops match your query. Try refining it.")
    else:
        # ✅ Top 5 laptop recommendations
        top5 = filtered_df.head(5).reset_index(drop=True)
        st.session_state.recommended_laptops = top5

        # 📄 Convert to LangChain documents
        documents = []
        for i, row in top5.iterrows():
            content = (
                    f"Laptop {i+1}:\n"
                    f"Brand: {row['Brand']}\n"
                    f"Model: {row['Product Name']}\n"
                    f"Price: ₹{row['Price']}\n"
                    f"Processor: {row['Processor']}\n"
                    f"RAM: {row['RAM']}\n"
                    f"Storage: {row['Storage']}\n"
                    f"Display: {row['Display']}\n"
                    f"Specifications: {row['Specifications']}\n"
                    f"OS: {row['OS']}\n"
                    f"Link: {row['Product URL']}"
                )
            documents.append(Document(page_content=content))

        # 🔁 Always reset chain with new documents
        st.session_state.chat_chain = setup_recommender_chain(documents)

        # 💻 Display recommended laptops
        st.subheader("✅ Top 5 Recommended Laptops (Detailed):")
        for idx, row in top5.iterrows():
            st.markdown(f"""
**{idx+1}. {row['Brand']} - {row['Product Name']}**
- 💰 Price: ₹{row['Price']}
- ⚙️ Processor: {row['Processor']}
- 🧠 RAM: {row['RAM']}
- 💾 Storage: {row['Storage']}
- 🖥️ Display: {row['Display']}
- 🔍 Specs: {row['Specifications']}
- 💼 OS: {row['OS']}
- 🔗 [Product Link]({row['Product URL']})
""")

# 💬 Handle follow-up questions
if follow_up and "chat_chain" in st.session_state:
    response = st.session_state.chat_chain.invoke(
        {"input": follow_up},
        config={"configurable": {"session_id": str(uuid.uuid4())}}
    )
    st.subheader("🤖 Follow-up Response:")
    st.markdown(response["answer"])

else:
    st.warning("📂 Please upload a valid CSV file to begin.")

