from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

parser = StrOutputParser()

model = ChatGoogleGenerativeAI(model='gemini-3-flash-preview')

model_embedding = 'gemini-embedding-001'

embedder = GoogleGenerativeAIEmbeddings(model=model_embedding)

chat_history = []

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """you are a helpful AI assistant, given an answer within the context and chat history of the given information ONLY.
        If you are unable to answer the question within the constraints of the context or chat history just say you dont know"""),
        MessagesPlaceholder(variable_name= "chat_history"),
        ("human", "question: {question}\ncontext: {context}")
    ]
)

def extract_video_id(url: str) -> str:
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return url


def get_transcript(video_id: str) -> str:
    try:
        ytt = YouTubeTranscriptApi()
        transcript_list = ytt.fetch(video_id)
        return " ".join(obj.text for obj in transcript_list.snippets)
    except TranscriptsDisabled:
        raise ValueError("Transcripts are disabled for this video")


def create_vectordb(video_id: str, transcript_text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.create_documents([transcript_text])

    vector_db = Chroma(
        collection_name=video_id,
        embedding_function=embedder,
        persist_directory="./chromadb_langchain"
    )

    if vector_db._collection.count() == 0:
        vector_db.add_documents(documents=chunks)

    return vector_db


def format_docs(docs):
    op = ''
    for i in docs:
        op += '\n\n' + i.page_content
    return op

def ask_question(url: str, question: str) -> str:
    
    video_id = extract_video_id(url)
    transcript_text = get_transcript(video_id)
    vector_db = create_vectordb(video_id, transcript_text)

    retriever = vector_db.as_retriever(search_type='mmr', search_kwargs={'k': 3})

    parallel_chain = RunnableParallel({
        'question': RunnablePassthrough(),
        'chat_history': RunnableLambda(lambda _: chat_history),
        'context': retriever | RunnableLambda(format_docs)
    })

    main_chain = parallel_chain | prompt | model | parser

    response = main_chain.invoke(question)

    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response))

    return response