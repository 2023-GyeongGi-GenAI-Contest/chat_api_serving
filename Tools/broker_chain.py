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
너는 사용자의 전세사기 예방을 돕는 챗봇이야. 너는 여러 챗봇 중 "명의도용 대출사기, 브로커를 통한 전세보증 사기"에 대한 내용을 답변하는 챗봇이야.
"""
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

human_template="""
1. 명의도용 대출 사기
사례 : 지난 2016년, 명의도용으로 대출금을 가로챈 사기꾼 최모 씨의 범행이 드러났어요. 최 씨는 급전이 필요하지만 일반 대출이 어려운 사람들에게 접근해, “가짜 전세계약서로 금리가 낮은 전세자금 대출을 받아주겠다”고 했어요. 정부 지원 전세대출 상품은 전세계약서를 내고 대출 신청만 하면 신용등급이 낮아도 무담보 전세대출을 받을 수 있다는 사실을 악용한 거죠. 이런 수법으로 최 씨는 은행 18곳에서 모두 대출 승인을 받았고 대출금 4억5천만 원을 전부 가로챘어요. 이런 사례를 명의도용 대출이라고 해요. 실제 전세계약이 체결되지 않았는데 임대인과 세입자의 명의를 도용해 가짜 전세계약서를 작성해 여러 군데의 금융기관에서 대출을 실행하는 거죠. 이 밖에도 임대인이 세입자에게 잠시만 명의를 빌려달라고 해서 몰래 대출을 받으려고 하는 등의 명의도용 사기는 빈번하게 일어난답니다. 이 경우에 아무리 범죄행위를 몰랐다고 해도 명의 제공자에게 책임이 전가될 수 있으니 주의해야 해요.
예방 방법 1 - 신분증, 인감 빌려주지 않기
예방 방법 2 - 계약서를 실제 계약 내용과 다르게 작성하지 않기

2. 브로커를 통한 전세보증 사기
사례 : 전세금안심대출보증을 악용해 주택도시보증공사에게 손실을 떠넘기는 사기 사건들이 벌어지고 있다고 해요. 전세보증금을 반환할 형편이 안 되는 사람에게 소유권을 넘기고 세입자는 전세금안심대출보증에 가입해 최대한도로 전세대출을 실행하는 거죠. 전세계약이 종료되면 임대인은 파산 신청을 하는 등의 방식으로 보증금을 반환하지 않고, 세입자는 주택도시보증공사에게 보증금을 돌려달라고 하게 되고요. 특히, 브로커가 명의를 빌려주면 대가를 지급하겠다고 꼬드겨 세입자이나 임대인의 명의를 도용하는 사례들이 있다고 하니 조심하세요! 이런 조직적인 사기 사건에 가담할 경우, 부동산 실명법 위반이나 사기죄 등으로 처벌받을 수 있으니 각별한 주의가 필요해요. 이렇게 되면, 나도 전세사기에 가담하고 휘말리게 되는 것이기 때문에, 처음부터 이런 제안에 응하지 않는 것이 중요해요. 브로커나 임대인이 실제 계약하고자 하는 내용과 다른 계약 방식을 제안했을 때 수락 하지 않아야 해요. 실제와 다른 이면계약을 작성하면 전세금안심대출보증에 가입했더라도 공사의 보증 의무가 없어서 공사로부터도 전세금을 돌려받지 못할 수 있어요.
예방 방법 1 - 신분증, 인감 빌려주지 않기
예방 방법 2 - 적법한 계약 절차가 아니면 거절하기

규칙 :
1. 질문이 특정 사기 유형에 해당하지 않는다면 먼저 전세사기 유형 2가지에 대해 답변에서 설명한다.
2. 해당하는 유형별 예방방법을 반드시 답변에 포함한다.

위 내용과 규칙을 참고해서 다음 Question에 500단어 정도로 답변해줘.
Question :  {text}
"""
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# prompt template가 1323이고, 사용자의 질문을 고려해서 2500으로 산정
# llm = ChatOpenAI(temperature=0, max_tokens=2500)
llm = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0, max_tokens=700)

broker_chain = LLMChain(
    llm=llm,
    prompt=chat_prompt
)
def return_broker_chain():
    return broker_chain