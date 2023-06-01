from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatMessagePromptTemplate
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

system_template="""
너는 사용자의 전세사기 예방을 돕는 챗봇이야. 너는 여러 챗봇 중 "계약기간이 끝난 후" 시점의 사용자를 담당해.
"""
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

human_template="""
계약기간이 끝난 후 생길 수 있는 전세사기 유형에는 계약이 끝났는데 임대인이 보증금을 안돌려주는 경우가 있을 수 있습니다.
대처방법 1 - 내용증명을 보내기, ‘내용증명’이란 쉽게 말해 우체국에서 공적 증명을 해주는 거예요. 내가 누군가에게 어떤 서류를 전달할 때 내용증명으로 보내면, 우체국에서는 그 서류가 전달되었는지뿐만 아니라 그 내용까지도 증명해주는 거죠. 간단하지만 생각보다 받는 상대방은 심리적 압박을 받을 수 있어요. 물론 실제 법적인 공방이 오가게 될 때 증거로서 작용할 수도 있죠.
대처방법 2 - '임차권등기명령'을 신청하기, 보증금을 반환받지 못한 상황에서 급하게 다음 집으로 이사를 해야 하는 상황을 겪고 있는 분들이 많을 거예요. 이런 경우, 내가 이사를 하더라도 임차권등기명령을 완료해 놓으면 대항력을 유지할 수가 있어요. 쉽게 말해 임차권에 대한 권리와 보증금에 대한 권리가 보호받게 되는 거죠. 더군다나 임차권등기명령은 임대인에게 매우 치명적이라고 해요. 등기부등본 을구에 ‘주택임차권’이라는 이름으로 그대로 적히게 되거든요. 그리고 누가 들어오든 이미 ‘나’의 보증금에 대한 우선변제권이 설정되어 있어서 임대인이 마음대로 다음 세입자를 입주시키고 나를 무시할 수 없어요. 임차권이 등기부등본에 쓰이면 다음 세입자보다 일단 내 보증금이 우선하여 보호받게 되거든요. 반대로, 내가 알아보고 있는 집의 등기부등본에 임차권등기명령이 표시되어 있다면? 이미 그 집에 살던 세입자가 여전히 보증금을 돌려받고 있지 못한다는 뜻이니, 계약을 더 신중히 고민해주세요.
대처방법 3 - 지급명령 신청 활용하기, 이 절차는 전세금 반환 소송으로 들어가기 전 최후의 수단이에요. 전세금 반환 소송에 비해 상대적으로 소요 기간이 짧아서 지급 결정 확정까지 한 달 안에 가능해요. 소요 비용 역시 적기 때문에 경제적으로도 유리해요.

규칙 :
1. 질문이 특정 사기 유형에 해당하지 않는다면 먼저 임대인이 보증금을 돌려주지 않는 경우에 대한 대처방법을 답변한다.

위 내용과 규칙을 참고해서 다음 Question에 1500단어 정도로 답변해줘.
Question :  {text}
"""
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# prompt template가 1025이고, 사용자의 질문을 고려해서 2500으로 산정
# llm = ChatOpenAI(temperature=0, max_tokens=1500)
llm = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0, max_tokens=2500)

after_contract_period_chain = LLMChain(
        llm=llm,
        prompt=chat_prompt
    )
def return_after_contract_period_chain():
    return after_contract_period_chain