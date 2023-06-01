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
너는 사용자의 전세사기 예방을 돕는 챗봇이야. 너는 여러 챗봇 중 "입주한 이후" 시점의 사용자를 담당해
"""
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

human_template="""
입주한 이후 생길 수 있는 전세사기 유형에는 2가지가 있습니다. 아래는 전세사기의 유형과 각 유형에 대한 사례와 예방방법입니다.

1. 미납국세 및 임금채권 우선 변제
사례 : 근저당 금액과 선순위 보증금을 다 계산했음에도 불구하고, 전세금을 돌려받지 못하는 경우가 발생할 수 있어요. 그건 바로 세금을 내지 않는 사람들 때문이지요. 만약 내가 살게 될 집의 임대인이 고액체납자라면, 집이 경매에 넘어갔을 때 보증금 전부를 돌려받지 못할 수도 있어요. 실제로 한국자산관리공사의 자료에 따르면 2016년부터 2020년에 이르기까지, 임대인이 국세를 미납하는 바람에 900명의 가까운 세입자가 총 335억의 보증금을 돌려받지 못했다고 해요. 그런가 하면 사용자가 노동자에게 정당한 임금을 지불하지 않는 경우, 이것이 임금채권이 되어 사용자의 뒤를 따라다니게 되는데요. 이렇게 임금채권이 많은 빚쟁이를 임대인으로서 만나게 되면 그 역시, 보증금을 떼일 우려가 있어요.
예방방법 1 - 미납국세 열람제도를 활용, 미납국세 열람제도를 활용해볼 수 있어요. 임대인 동의가 있다면 확인해볼 수 있죠.
예방방법 2 - 사회보험 완납증명서 또는 납세증명서 확인, 사회보험 완납증명서 또는 납세증명서를 임대인 동의로 확인해볼 수 있어요. 사회보험 미납이 곧 임금 체불은 아니지만, 그 가능성과 위험도를 측정하는 데에는 도움을 줄 거예요. 또한 납세증명서를 통해서 확인해볼 수 있어요.
예방방법 3 - 등기부등본을 확인, 임금을 체불했거나 세금을 체납했을 때 간혹 채권자나 국가가 ‘압류’ 혹은 ‘가압류’와 같은 채권보전조치를 취하는 경우가 있거든요. 따라서 만일 등본을 확인했는데 ‘압류’, ‘가압류’, 강제경매 개시 결정’ 등 임대인의 소유권이 위태로워지는 단어들이 보인다면 반드시 피하세요!
예방방법 4 - 전세보증금반환보증에 가입, 내가 미처 예방할 수 없었던 문제가 생겼을 때, 보증보험을 통해 보증금을 보호받을 수 있어요. 단! 압류, 가압류 등 소유권에 대한 권리침해 상황이 발생한다면 보증보험 가입이 안 돼요. 그러니까 대항력을 갖추며 최대한 빠르게 보증보험에 가입하는 것이 좋아요. 집을 구할 때부터, 전세보증금반환보증에 가입할 수 있는 집을 찾는 것도 좋은 방법이에요. 예를 들어, 여러분이 중소기업취업청년 전월세보증금 대출을 100%로 받으셨다면, 대출 상품 자체가 보증보험에 가입되어 있어서 걱정하지 않으셔도 돼요.

2. 세입자의 전출신고 후 벌어지는 전세사기
사례 : 전세 세입자인 A씨는 어느 날 임대인에게서 전화 한 통을 받았어요. 대출을 받아야 한다고, 잠시 집에서 전출해줄 수 있냐는 임대인의 요청이 있었어요. A씨는 별문제가 없을 거라 판단해 임대인이 대출받는 날 잠깐 전출하고 다시 전입신고를 했죠. 그런데 이후 A씨가 전세자금 대출을 연장하려고 하자 은행에선 A씨의 대출 연장을 거절했어요.
이러한 사기가 벌어지는 이유 1 - 전입신고를 한 후에는 임대인이 무엇을 요구하든 전출신고를 하지 않는 게 좋아요. 전출신고를 하면 그 순간부터 내 대항력이 와장창! 깨지게 되거든요. 내가 이 집의 세입자로 권리를 보장받기 위해서는 대항력을 유지하고 있어야 하는데, 중요한 요건 중 하나인 ‘전입신고’가 해제되는 것이라, 매우 위험해요. 분명 내가 먼저 살고 있었는데, 내가 전출신고한 사이에 들어온 빛이 나보다 우선하게 되는 것이기도 하고요. 이런 경우, 집이 경매에 넘어가면 그 빚을 보전하느라 내 보증금을 돌려받지 못하게 될 수도 있어요.
이러한 사기가 벌어지는 이유 2 - 세입자가 전출신고를 한 사이에 임대인이 집을 필수도 있어요. 이렇게 되면 소유권이 타인에게로 넘어가게 되고, 내 대항력이 깨진 채로 정말 곤란한 일이 벌어질 수 있어요. 새 임대인이 나보고 나가라고 했을 때, 이미 내가 전출신고 후 대항력이 깨진 상태이기 때문에 나의 상황을 보호받는 과정이 몹시 복잡해질 수 있어요.

규칙 :
1. 질문이 특정 사기 유형에 해당하지 않는다면 먼저 전세사기 유형 2가지에 대해 답변에서 설명한다.
2. 해당하는 유형별 예방방법을 반드시 답변에 포함한다.

위 내용과 규칙을 참고해서 다음 Question에 500단어 정도로 답변해줘.
Question :  {text}
"""
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# prompt template가 2020이고, 사용자의 질문을 고려해서 1500으로 산정
# llm = ChatOpenAI(temperature=0, max_tokens=1500)
llm = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0, max_tokens=700)

after_home_chain = LLMChain(
    llm=llm,
    prompt=chat_prompt
)
def return_after_home_chain():
    return after_home_chain

