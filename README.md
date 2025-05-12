# AI_Powered_Search_Engine

This project is an end-to-end pipeline that scrapes laptop listings from Flipkart, cleans and enriches the data, and uses LangChain, FAISS, and Streamlit to build a chatbot that provides laptop recommendations based on user queries.
----------------------------------------------------------------------------------------------

Features
-----------------------------------------------------------------------------------------------
✅ Web scraping of laptop data from Flipkart using Selenium + BeautifulSoup
✅ Enriches each product with battery life, weight, webcam, and display size
✅ Cleans and structures the data with pandas
✅ Chatbot interface powered by Streamlit + LangChain
✅ Intelligent filtering by price, specifications, and user intent (e.g., gaming, office use)
✅ Context-aware question answering and follow-up support
✅ Product comparison and specification summary
---------------------------------------------------------------------------------------------

Project Structure
----------------------------------------------------------------------------------------------
.
├── flipkart_scraper.py          # Step 1: Web scraping script
├── flipkart_laptop_final.csv    # Scraped raw data
├── data_cleaning.py             # Step 2: Data cleaning & enrichment
├── flipkart_laptop_cleaned.csv  # Final structured dataset
├── app.py                       # Streamlit chatbot app
├── filter.py                    # Custom filtering logic
├── README.md                    # Project documentation
---------------------------------------------------------------------------------------------

1️⃣ Web Scraping from Flipkart
=============================================================================================
We use headless Chrome with Selenium to navigate through search result pages and extract product details and extra specs (battery, webcam, etc.) from individual product pages.

📁 Output: flipkart_laptop_final.csv

# Setup headless Chrome and loop through 30 pages of laptop results
# Extract details + visit product pages for specs like battery and webcam
# Save all collected data to CSV

=============================================================================================

2️⃣ Data Cleaning
============================================================================================
After scraping, we clean and structure the data using pandas.

Adds columns like Brand, Processor, RAM, Storage, etc.
Creates combined text and all_text fields for embedding.
📁 Output: flipkart_laptop_cleaned.csv

============================================================================================

3️⃣ Chatbot with LangChain + FAISS + Streamlit
============================================================================================
⚙️ How it works
Upload CSV: The cleaned laptop data is uploaded.
Filter Laptops: Apply filters by price, specs, and purpose (e.g., gaming, office).
RAG Pipeline: User question is passed through a LangChain Retrieval-Augmented Generation chain using FAISS and Ollama.
Recommendations: Laptops are recommended with price, specs, and URLs.
Follow-Up Queries: Users can ask to compare laptops or get full specifications.

🧠 Supported Queries
"Show me laptops under 60k for gaming"
"Which laptop is best for students?"
"Compare laptop 1 and 2"
"What are the specifications of laptop 3?"
=============================================================================================

🛠 Tech Stack
---------------------------------------------------------------------------------------------
Python
Selenium + BeautifulSoup – Scraping
pandas – Data Cleaning
Streamlit – Web App
LangChain + FAISS – Retrieval-Augmented Generation (RAG)
Ollama – Local LLM
HuggingFace Embeddings – Vector representations
=============================================================================================

📥 How to Run
---------------------------------------------------------------------------------------------
1.Clone the repo
git clone https://github.com/your-username/flipkart-laptop-chatbot.git
cd flipkart-laptop-chatbot

2.Install dependencies
pip install -r requirements.txt

3.Run the Streamlit app
streamlit run app.py

4.Upload your CSV
Use flipkart_laptop_cleaned.csv


