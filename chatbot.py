# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
# from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from langchain_core.prompts import ChatPromptTemplate
# from dotenv import load_dotenv

# load_dotenv()

# parser = StrOutputParser()

# model = ChatGoogleGenerativeAI(model='gemini-3-flash-preview')

# model_embedding = 'gemini-embedding-001'

# embedder = GoogleGenerativeAIEmbeddings(model=model_embedding)

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "you are a helpful AI assistant, given an answer within the context of the given information ONLY. If you are unable to answer the question within the constraints of the context just say you dont know"),
#         ("human", "question: {question}\ncontext: {context}")
#     ]
# )


# def extract_video_id(url: str) -> str:
#     if "v=" in url:
#         return url.split("v=")[1].split("&")[0]
#     elif "youtu.be/" in url:
#         return url.split("youtu.be/")[1].split("?")[0]
#     return url


# def get_transcript(video_id: str) -> str:
#     try:
#         ytt = YouTubeTranscriptApi()
#         transcript_list = ytt.fetch(video_id)
#         return " ".join(obj.text for obj in transcript_list.snippets)
#     except TranscriptsDisabled:
#         raise ValueError("Transcripts are disabled for this video")


# def create_vectordb(video_id: str, transcript_text: str):
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,
#         chunk_overlap=100
#     )
#     chunks = splitter.create_documents([transcript_text])

#     vector_db = Chroma(
#         collection_name=video_id,
#         embedding_function=embedder,
#         persist_directory="./chromadb_langchain"
#     )

#     if vector_db._collection.count() == 0:
#         vector_db.add_documents(documents=chunks)

#     return vector_db


# def format_docs(docs):
#     op = ''
#     for i in docs:
#         op += '\n\n' + i.page_content
#     return op


# def ask_question(url: str, question: str) -> str:
    
#     video_id = extract_video_id(url)
#     transcript_text = get_transcript(video_id)
#     vector_db = get_or_create_vectordb(video_id, transcript_text)

#     retriever = vector_db.as_retriever(search_type='mmr', search_kwargs={'k': 3})

#     parallel_chain = RunnableParallel({
#         'question': RunnablePassthrough(),
#         'context': retriever | RunnableLambda(format_docs)
#     })

#     main_chain = parallel_chain | prompt | model | parser

#     return main_chain.invoke(question)

# print(ask_question("https://www.youtube.com/watch?v=BHdbsHFs2P0&t=1559s",'what is the theorem discussed in this video'))

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableLambda,RunnablePassthrough
from youtube_transcript_api import YouTubeTranscriptApi , TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv

load_dotenv()

parser = StrOutputParser()

model = ChatGoogleGenerativeAI(
	model = 'gemini-3-flash-preview'
	)

video_id = 'KGVpKPNUdzA'

try: 
	transcript_list = YouTubeTranscriptApi().fetch(video_id,languages=['en'])

except TranscriptsDisabled:
	print('no caption enabled')


pre_chunk = " ".join(obj.text for obj in transcript_list.snippets)

splitter = RecursiveCharacterTextSplitter(
	chunk_size=500,
	chunk_overlap=100
)

chunks = splitter.create_documents([pre_chunk])

model_embedding = 'gemini-embedding-001'

embedder = GoogleGenerativeAIEmbeddings(model=model_embedding)

vector_db = Chroma(
	collection_name="chroma_collection",
	embedding_function=embedder
)

vector_db.add_documents(documents=chunks)

retriever = vector_db.as_retriever(search_type='mmr',search_kwargs={'k':3})

def format_docs(docs):
	op = ''
	for i in docs:
		op += '\n\n' + i.page_content
	return op

question = 'who are the guests in this podcast.'

prompt = ChatPromptTemplate(
    [
        ("system", "you are a helpful AI assistant, given an answer within the context of the given information ONLY. If you are unable to answer the question within the constraints of the context just say you dont know"),
        ("human", "question: {question}\ncontext: {context}")
    ]
)

parallel_chain = RunnableParallel({
	'question' : RunnablePassthrough(),
	'context' :  retriever | RunnableLambda(format_docs)
	})

main_chain = parallel_chain | prompt | model | parser

# print(main_chain.invoke(question))