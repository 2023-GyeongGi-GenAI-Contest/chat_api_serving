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
너는 사용자의 전세사기 예방을 돕는 챗봇이야. 너는 여러 챗봇 중 "계약서 작성할 때" 시점의 사용자를 담당해
"""
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

human_template="""
계약서를 작성할 때 생길 수 있는 전세사기 유형에는 2가지가 있습니다. 아래는 전세사기의 유형과 각 유형에 대한 사례와 예방방법입니다.

1. 월셋집을 전셋집으로 둔갑시킨 경우
사례 : 2019년, 경기도 안산에서는 두 명의 공인중개사가 월셋집을 전셋집으로 둔갑시켜 피해액이 70억원에 달하는 사기를 저질렀어요. 임대인에게는 월세 계약이라고 알리고 세입자와는 전세 계약을 맺은 후, 그 사이에서 전세보증금을 가로챈 거죠.
- 일부 지역에는 나쁜 관행이 있어요. 그건 바로 임대인 없이 임대차계약을 맺는 관행이에요. 공인중개사가 임대인의 대리인을 자처하면서, 임대인이나 대리인 관련 서류를 전혀 준비해주지 않는 거죠. 이 빈틈을 노리고, 사기를 치는 일부 나쁜 공인중개사가 있어요. 예를 들어볼게요. 여기 임대인의 대리인으로서 계약을 하러 나온 공인중개사가 한 명 있어요. 이 공인중개사는 세입자와 보증금 1억 원의 전세계약을 체결해버려요. 하지만 임대인에게는 보증금 1천만 원의 월세 계약을 체결했다고 거짓말을 해요. 이렇게 발생한 총 9천만 원의 차액을 나쁜 공인중개사가 빼돌려요. 공인중개사가 임대인과 세입자 모두를 속이는 거죠.
예방방법 1 - 그렇다면 이러한 불상사를 막기 위해서는 어떻게 해야할까요? 우선, 해당 공인중개업자가 정상적으로 등록된 공인중개업자가 맞는지 사전에 확인해야 해요. 무자격자가 증개업 등록증이나 자격증을 빌려서 중개사무소를 차리는 경우가 생각보다 많아요. 등록된 중개업자인지 여부는 '국가공간정보포털'에서 확인할 수 있어요. 열람공간에 들어가 부동산 중개업조회 메뉴를 클릭하면 지역과 사무소 상호, 공인중개사 이름, 전화번호 등을 입력할 수 있고 이를 통해 등록번호와 중개업자 여부를 확인할 수 있어요.
예방방법 2 - 개업 중인 공인중개사는 공인중개사법 제30조에 따라 보증보험이나 공제상품에 가입해야 하는데요. 이에 따라 공인중개사의 과실이나 고의로 계약자가 금전적 피해를 입었을 때 손해를 보증한다는 내용을 담은 '부동산 공제증서'를 발급하게 돼요. 이 공제증서는 실제 중개사고 때문에 벌어진 손해를 배상해줄 때에 쓰이며, 공인중개사가 사기를 벌이지 않겠 다는 다짐이기도 해요. 그러니 중개업등록증과 공인중개사의 신분증을 확인해 꼭 증서를 받을 수 있도록 해요.
국가공간정보포털 링크 : http://www.nsdi.go.kr/lxportal/?menuno=2679

2. 선순위 임차보증금 및 근저당 허위 고지
사례 : 대구에서는 다가구 주택의 임대인이 다른 세입자들의 임차보증금 규모를 속여 전세계약을 체결하는 사건이 있었어요. 그리고 부산에서는 공동근저당이 설정되어있으나 이를 중요하게 생각하지 않고 전세계약을 체결하였다가 집이 경매에 넘어가 보증금을 회수하지 못하는 경우가 있었어요.
주택 A : 집값 10억/내 보증금 2억/다가구주택A에 살고있던 세입자들의 보증금 즉,선순위 임차보증금 12억
주택 B : 공동 담보 88억/나를 포함한 세입자들의 보증금 70억
주택 A, B중 어느것이 더 안전한 집일까요? - 정답은 둘 다 위험한 집입니다! 다가구주택A에는 너무 많은 보증금들이 이미 나보다 우선하여 묶여 있는 집이기 때문에 위험할 수 있어요. 다세대주택 또는 오피스텔B에는 공동 담보가 너무 커서, 세입자들의 보증금 보호가 제대로 지켜지지 않을 수 있어요. B의 경우, 등기부등본에서 공동담보에 대한 언급을 확인할 수 있어요. 하지만 다가구주택A에 사는 세입자들의 보증금은 등기부등본에 나타나지 않아요. 그래서 좀더 유의해야 해요.
예방방법 1 - 다가구주택의 경우 계약을 하기 전에는 다른 세입자의 보증금 규모를 알기 어려워요. 따라서 반드시 임대인의 동의를 받아 계약 전에 부채 규모를 확인해야 해요. 만일 임대인이 거짓말을 할 경우를 대비하여 특약사항을 기재하는 것을 추천해요.
예방방법 2 - 계약 전에는 임대인의 동의서, 임대인의 신분증 사본, 인감증명서 등 필요서류를 챙겨 확정일자 부여 현황 서류를 발급받도록 해요. 계약 후에는 임대차 계약서와 본인 신분증 사본 등 필요서류를 챙겨 확정일자 부여 현황과 전입세대 열람 내역 2가지 모두 발급해보세요. 확정일자 부여 현황과 전입세대 열람 내역에 기재된 보증금 금액을 통해 이 집에 대해 설정된 부채의 규모를 계산해 볼 수 있어요.

규칙 :
1. 질문이 특정 사기 유형에 해당하지 않는다면 먼저 전세사기 유형 2가지에 대해 답변에서 설명한다.
2. 해당하는 유형별 예방방법을 반드시 답변에 포함한다.

위 내용과 규칙을 참고해서 다음 Question에 1200단어 정도로 답변해.
Question :  {text}
"""
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# prompt template가 2146이고, 사용자의 질문을 고려해서 1500으로 산정
# llm = ChatOpenAI(temperature=0, max_tokens=1500)
llm = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0, max_tokens=1500)

before_contract_chain = LLMChain(
    llm=llm,
    prompt=chat_prompt
)

def return_before_contract_chain():
    return before_contract_chain