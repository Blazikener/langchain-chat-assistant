import os
import openai
import sys
import datetime
import json
import re

sys.path.append('../..')

# GUI
import panel as pn  
import param
pn.extension()

# Environment
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

# LLM setup
current_date = datetime.datetime.now().date()
llm_name = "gpt-3.5-turbo-0301" if current_date < datetime.date(2023, 9, 2) else "gpt-3.5-turbo"

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import PyPDFLoader
from textblob import TextBlob

# Prompt template
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

# Utility functions
def detect_mood(text):
    polarity = TextBlob(text).sentiment.polarity
    return "happy" if polarity > 0.3 else "sad" if polarity < -0.3 else "neutral"

def extract_topics(text):
    keywords = ['breakup', 'stress', 'family', 'exams', 'friends', 'job', 'love']
    return [word for word in keywords if word in text.lower()]

def save_user_context(user_email, mood_history, discussed_topics):
    os.makedirs("user_context", exist_ok=True)
    with open(f"user_context/{user_email}.json", "w") as f:
        json.dump({"moods": mood_history, "topics": list(discussed_topics)}, f)

def load_user_context(user_email):
    try:
        with open(f"user_context/{user_email}.json", "r") as f:
            data = json.load(f)
            return data["moods"], set(data["topics"])
    except FileNotFoundError:
        return [], set()

# PDF loader
def load_db(file, chain_type, k):
    loader = PyPDFLoader(file)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=0), 
        chain_type=chain_type, 
        retriever=retriever, 
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa

# ChatBot class
class cbfs(param.Parameterized):
    user_email = param.String("guest@example.com")
    mood_history = param.List([])
    discussed_topics = param.List([])
    chat_history = param.List([])
    answer = param.String("")
    db_query = param.String("")
    db_response = param.List([])

    def __init__(self, **params):
        super(cbfs, self).__init__(**params)
        self.panels = []
        self.loaded_file = "docs/cs229_lectures/MachineLearning-Lecture01.pdf" #make sure to put or use the pdf you want
        self.qa = load_db(self.loaded_file, "stuff", 4)

    def login_user(self, email):
        self.user_email = email
        self.mood_history, loaded_topics = load_user_context(email)
        self.discussed_topics = list(loaded_topics)
        self.clr_history()
        return pn.pane.Markdown(f"Logged in as: {email}")

    def call_load_db(self, count):
        if count == 0 or file_input.value is None:
            return pn.pane.Markdown(f"Loaded File: {self.loaded_file}")
        else:
            file_input.save("temp.pdf")
            self.loaded_file = file_input.filename
            button_load.button_style = "outline"
            self.qa = load_db("temp.pdf", "stuff", 4)
            button_load.button_style = "solid"
        self.clr_history()
        return pn.pane.Markdown(f"Loaded File: {self.loaded_file}")

    def convchain(self, query):
        if not query:
            return pn.WidgetBox(pn.Row('User:', pn.pane.Markdown("", width=600)), scroll=True)

        mood = detect_mood(query)
        self.mood_history.append(mood)
        new_topics = extract_topics(query)
        self.discussed_topics.extend([topic for topic in new_topics if topic not in self.discussed_topics])
        save_user_context(self.user_email, self.mood_history[-5:], set(self.discussed_topics))

        context_info = f"""
User mood over last few chats: {self.mood_history[-5:]}
Previously discussed topics: {self.discussed_topics}
"""
        full_prompt = f"{context_info}\nUser: {query}"

        result = self.qa({"question": full_prompt, "chat_history": self.chat_history})
        self.chat_history.append((query, result["answer"]))
        self.db_query = result["generated_question"]
        self.db_response = result["source_documents"]
        self.answer = result['answer']

        self.panels.extend([
            pn.Row('User:', pn.pane.Markdown(query, width=600)),
            pn.Row('ChatBot:', pn.pane.Markdown(self.answer, width=600, style={'background-color': '#F6F6F6'}))
        ])
        inp.value = ''
        return pn.WidgetBox(*self.panels, scroll=True)

    @param.depends('db_query')
    def get_lquest(self):
        if not self.db_query:
            return pn.Column(
                pn.pane.Markdown("Last question to DB:", styles={'background-color': '#F6F6F6'}),
                pn.pane.Str("No DB accesses so far")
            )
        return pn.Column(
            pn.pane.Markdown("DB query:", styles={'background-color': '#F6F6F6'}),
            pn.pane.Str(self.db_query)
        )

    @param.depends('db_response')
    def get_sources(self):
        if not self.db_response:
            return
        rlist = [pn.pane.Markdown("Result of DB lookup:", styles={'background-color': '#F6F6F6'})]
        for doc in self.db_response:
            rlist.append(pn.pane.Str(doc))
        return pn.WidgetBox(*rlist, width=600, scroll=True)

    @param.depends('chat_history')
    def get_chats(self):
        if not self.chat_history:
            return pn.WidgetBox(pn.pane.Str("No History Yet"), width=600, scroll=True)
        rlist = [pn.pane.Markdown("Current Chat History variable", styles={'background-color': '#F6F6F6'})]
        for exchange in self.chat_history:
            rlist.append(pn.pane.Str(exchange))
        return pn.WidgetBox(*rlist, width=600, scroll=True)

    def clr_history(self, count=0):
        self.chat_history = []
        self.panels = []

# Setup Panel app
cb = cbfs()
file_input = pn.widgets.FileInput(accept='.pdf')
button_load = pn.widgets.Button(name="Load DB", button_type='primary')
button_clearhistory = pn.widgets.Button(name="Clear History", button_type='warning')
button_clearhistory.on_click(cb.clr_history)
inp = pn.widgets.TextInput(placeholder='Enter text here…')

bound_button_load = pn.bind(cb.call_load_db, button_load.param.clicks)
conversation = pn.bind(cb.convchain, inp)

jpg_pane = pn.pane.Image('./img/convchain.jpg')

# Tabs
tab1 = pn.Column(
    pn.Row(inp),
    pn.layout.Divider(),
    pn.panel(conversation, loading_indicator=True, height=300),
    pn.layout.Divider(),
)
tab2 = pn.Column(
    pn.panel(cb.get_lquest),
    pn.layout.Divider(),
    pn.panel(cb.get_sources),
)
tab3 = pn.Column(
    pn.panel(cb.get_chats),
    pn.layout.Divider(),
)
tab4 = pn.Column(
    pn.Row(file_input, button_load, bound_button_load),
    pn.Row(button_clearhistory, pn.pane.Markdown("Clears chat history. Can use to start a new topic")),
    pn.layout.Divider(),
    pn.Row(jpg_pane.clone(width=400))
)

dashboard = pn.Column(
    pn.Row(pn.pane.Markdown('# ChatWithYourData_Bot')),
    pn.Tabs(('Conversation', tab1), ('Database', tab2), ('Chat History', tab3), ('Configure', tab4))
)

dashboard.servable()
