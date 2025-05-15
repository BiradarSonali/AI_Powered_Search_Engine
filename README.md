# 🧠 AI-Powered Laptop Search Engine

An end-to-end GenAI pipeline that scrapes laptop listings from Flipkart, cleans and enriches the data, and uses LangChain, FAISS, and Streamlit to build a chatbot that recommends laptops based on user intent.

-----------------------------------------------------------------------------------------------------------------------------------------------------------

## ✅ Features

- 🔍 Web scraping of laptop data from Flipkart (Selenium + BeautifulSoup)
- 🧹 Cleans and enriches product specs (battery, weight, display, webcam, etc.)
- 🧠 LangChain-based chatbot powered by local LLM (Groq)
- 🔎 Intelligent filtering by price, specs, and use-case (e.g. gaming, office)
- 🤖 Context-aware question answering and follow-ups
- 📊 Product comparison and full specification summary

-------------------------------------------------------------------------------------------------------------------------------------------------------------

## 📂 Project Structure

.
├── flipkart_scraper.py # Step 1: Scrape Flipkart laptop listings

├── flipkart_laptop_final.csv # scraped dataset

├── cleaning_data.py # Step 2: Clean data

├── flipkart_laptop_cleaned.csv # Final processed dataset

├── filteration.py # Custom filtering logic for chatbot

├── laptop_recommender.py # Streamlit chatbot app

├── README.md # Documentation


-------------------------------------------------------------------------------------------------------------------------------------------------------------

## 1️⃣ Web Scraping from Flipkart

We use headless Chrome (Selenium) + BeautifulSoup to:
- Loop through 25+ pages of Flipkart laptop results
- Visit individual product pages for extra specs (battery life, webcam, etc.)
- Output data to `flipkart_laptop_final.csv`

-----------------------------------------------------------------------------------------------------------------------------------------------------------

## 2️⃣ Data Cleaning & Enrichment

Using pandas, we:
- Extract structured fields: Brand, Processor, RAM, Storage, etc.
- Add enriched columns: battery, webcam, weight, display size
- Create combined text fields for embeddings (`all_text`)
- Output: `flipkart_laptop_cleaned.csv`

--------------------------------------------------------------------------------------------------------------------------------------------------------------

## 3️⃣ Chatbot with LangChain + FAISS + Streamlit

How it works:
- Upload: Cleaned CSV is loaded at runtime
- Vectorization: Product descriptions are embedded using HuggingFace
- Retrieval: FAISS index + LangChain RAG used for query answering
- UI: Chatbot interface via Streamlit with filters + follow-ups

🧠 Supported Queries:
- “Show me laptops under 60k for gaming”
- “Which laptop is best for students?”
- “Compare laptop 1 and 2”
- “What are the specifications of laptop 3?”

--------------------------------------------------------------------------------------------------------------------------------------------------------------

## 🛠 Tech Stack

- 🐍 Python
- 🌐 Selenium + BeautifulSoup – Web scraping
- 📊 pandas – Data cleaning
- 🧠 LangChain + FAISS – RAG + vector search
- 🗣️ Groq – Local LLM for query answering
- 🔤 HuggingFace – Text embeddings
- 📺 Streamlit – Chatbot UI

--------------------------------------------------------------------------------------------------------------------------------------------------------------

## ▶️ How to Run

 1. Clone the repo

     git clone https://github.com/your-username/flipkart-laptop-chatbot.git
    
     cd flipkart-laptop-chatbot

    

 3. Install dependencies

     pip install -r requirements.txt
     


 4. Run the app

     streamlit run laptop_recommender.py
    


 6. Upload your CSV (use flipkart_laptop_cleaned.csv)


