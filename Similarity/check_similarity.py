from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI

embeddings = OpenAIEmbeddings()

docsearch = FAISS.load_local("entire_content_vectordb", embeddings)

# query = "전세계약을 하려는데 집주인이 아닌 집주인의 친척이라는 사람이 왔어 이거 사기 위험이 있는거 아니야?"
# docs = docsearch.similarity_search(query)[0:1]

template = """
Questioner status : {question}
=========
{summaries}
=========
The questioner wants to know if there is a risk of a charter fraud.
You should judge the similarities between the above case and the questioner's situation and let us know what is dangerous.
When you answer, you must compare the situation and case of the questioner and let them know what is dangerous.
you must answer IN KOREAN.
"""

PROMPT = PromptTemplate(template=template, input_variables=["summaries", "question"])

chain = load_qa_with_sources_chain(ChatOpenAI(temperature=0, max_tokens=1000), chain_type="stuff", prompt=PROMPT)

def get_reply(input):
    docs = docsearch.similarity_search(input)[0:1]
    return chain({"input_documents": docs, "question": input}, return_only_outputs=True)