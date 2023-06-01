from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

system_template="""
fake
"""
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

human_template="""
fake
"""
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
llm = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0, max_tokens=2500)


default_message = """
해당 질문에는 질문자님의 계약 시점에대한 정보가 부족합니다.

계약 시점 예시 :
1. 집 고를 때
2. 임대인 확인할 때
3. 계약서 작성할 때
4. 계약한 직후
5. 입주한 이후
6. 계약기간이 끝난 후

위와 같은 정확한 시점을 명시해주시면 더욱 상세한 답변이 가능합니다.
"""
class FakeLLMChain(LLMChain):
    def run(self, inputs):
        return default_message

fake_llm_chain = FakeLLMChain(llm=llm, prompt=chat_prompt)

def return_fake_chain():
    return fake_llm_chain