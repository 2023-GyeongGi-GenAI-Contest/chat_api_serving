from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.utilities import SerpAPIWrapper
from langchain.memory import ConversationBufferWindowMemory

from langchain.chains import RetrievalQA

from langchain.prompts import PromptTemplate

embeddings = OpenAIEmbeddings()

db1 = FAISS.load_local("realestate", embeddings)

prompt_template1="""
너는 전세사기에 대한 질문에 답변하는 챗봇이야.

context:
{context}

위 context를 참조하여 아래 question에 대한 답변을 해줘.

question:
{question}
"""

PROMPT1 = PromptTemplate(
    template=prompt_template1, input_variables=["context", "question"]
)

chain_type_kwargs1 = {"prompt": PROMPT1}

qa1 = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-3.5-turbo-16k", max_tokens=4000), chain_type="stuff", retriever=db1.as_retriever(search_kwargs={"k": 5}), chain_type_kwargs=chain_type_kwargs1)

params = {
    "engine": "google",
    "gl": "kr",
    "hl": "ko",
}
search = SerpAPIWrapper(params=params)

tools = [
    Tool(
        name = "전세사기 상담 챗봇",
        func=qa1.run,
        description="전세사기에 대한 내용을 답변해야할 때 유용합니다.",
        return_direct=True,
    ),
    Tool(
        name = "부동산정보 답변 챗봇",
        func=search.run,
        description="부동산 용어 또는 부동산에 관한 일반적인 질문에 답변하는데 유용합니다.",
        return_direct=True,
    ),
]

memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)

agent_chain = initialize_agent(tools, ChatOpenAI(model_name="gpt-3.5-turbo-16k", max_tokens=6000), agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)

def get_reply(input):
    return agent_chain.run(input + ' Please summarize it in 300 words.')