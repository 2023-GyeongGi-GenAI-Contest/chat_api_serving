from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

from langchain.agents import Tool
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent
from langchain.agents import AgentType

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

embeddings = OpenAIEmbeddings()

db = FAISS.load_local("faiss_index", embeddings)

qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=1, max_tokens=2000), chain_type="stuff", retriever=db.as_retriever())

params = {
    "engine": "google",
    'gl': "kr",
    'hl': "ko"
}

search = SerpAPIWrapper(params=params)

tools = [
    Tool(
        name = "화장품 제품 추천/리뷰 질의응답 시스템",
        func=qa.run,
        description="화장품의 리뷰나 화장품 추천에 관한 질문에 답을 해야할 때 유용합니다."
    ),
    Tool(
        name = "화장품 정보 검색 시스템",
        func=search.run,
        description="화장품 성분(화학 성분)에 대한 질문이나 화장품에 대한 용량, 가격, 성분과 같은 상세정보에 관한 질문에 답을 해야할 때 유용합니다."
    ),
]

def Agent (usermessage, memory):
    agent_chain = initialize_agent(tools, ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0, max_tokens=2400), agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)
    return agent_chain.run(usermessage + " 질문의 답변은 무조건 '한국어'로 대답해")
