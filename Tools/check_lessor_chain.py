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
너는 사용자의 전세사기 예방을 돕는 챗봇이야. 너는 여러 챗봇 중 "임대인 확인할 때" 시점의 사용자를 담당해
"""
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

human_template="""
임대인 확인할 때 생길 수 있는 전세사기 유형에는 2가지가 있습니다. 아래는 각 전세사기의 유형과 각 유형별 사례와 예방,대처법이 있습니다.
1. 가짜 임대인과의 계약
사례 : 2019년, 인천 중구에서 있었던 일이에요. A씨는 타인의 오피스텔을 자신의 소유라고 거짓말하면서 등기부등본의 소유자 항목을 위조하여 사기를 쳤어요. A씨가 바로 ‘가짜 임대인’이었던 것이죠. 피해자들의 대부분은 아직 임대차 경험이 많지 않은 사회 초년생이었죠. 금방 들통날 얄팍한 거짓말었지만, 그 피해가 생각보다 어마어마했어요. 무려 약 24억원에 달했죠.
이처럼 임대인이 아닌 사람이 임대인인 척 명의를 도용하는 경우가 심심치 않게 발생해요. ‘가짜 임대인’이 건물을 임대하면서 알아낸 ‘실제 임대인’의 인적사항을 이용하여 자신이 건물주인 척 거짓된 전세계약을 한 후 세입자의 보증금을 가로채는 거죠. 혹은 임대인의 위임장 또는 증명서류를 위조해 대리인 행세를 하면서 세입자의 보증금을 가로채기도 해요.
이런 불상사를 막기 위해서는 임대인이라고 주장하는 사람이 정말 임대인이 맞는지를 확인하는 일이 중요해요. 신분증에 쓰여 있는 임대인의 인적사항과 등기부등본 상의 임대인의 정보가 일치하는지 확인해야 하겠죠. 이때 등기부등본은 본인이 직접 발급받아 확인해보기를 추천해요. 등기부등본을 위조하는 경우도 있다고 하니까요.
임대인을 대신해서 대리인이 왔을 때 대처법 :
- 임대인의 신분증을 대체할 수 있는 서류들 즉, 인감도장이 찍힌 위임장과 그 인감도장을 증명할 인감증명서를 확인해야만 해요. 위임장과 인감증명서를 대조해 두 서류에 찍힌 임대인의 인감이 동일한지 체크하는 것이죠.
- 임대인과의 유선 통화를 통해 계약 내용을 다시 한 번 확인하기를 추천해요.
- 전세보증금을 반드시 임대인 명의의 통장에 지급해야 해요. 만일 임대인의 요청에 의해 대리인 계좌로 입금해야 하는 경우라면, 특약사항에 임대인의 요청에 따라 대리인 계좌로 입금하게 되었음을 명시해야만 합니다.

2. 신탁회사의 동의 없는 계약
사례 : 2019년, 마산에 사는 임대인 A씨는 2019년 1월에 한 오피스텔을 신탁회사에 넘기고 그 증서를 담보로 거액의 대출을 받았어요. 그럼 임대인 A씨에게는 신탁회사의 동의 없이는 집을 임대할 권한이 없게 되는데, A씨는 이를 속이고 15가구의 세입자들과 계약을 강행해버렸어요. 보증금을 가로챈 거죠. 신탁회사 동의 없이 계약했으니, 계약 자체가 무효가 됐어요. 세입자들은 하루아침에 쫓겨나게 됐고요. 그렇게 세입자들은 총 5억 원 규모의 신탁 부동산 사기 피해를 입게 됐어요.
신탁이란? - 신탁이란, 임대인이 주택을 비롯한 자신의 부동산을 전문가에게 맡기는 걸 의미해요. 임대인은 주택을 담보로 대출을 받을 수도 있고요, 혹은 다른 수익을 얻기 위해서도 신탁을 해요.
신탁등기와 일반등기를 구분하는 법 - 등기부등본 서류에는 표제부, 갑구, 을구가 있어요. 이 중 ‘갑구’를 보면, ‘신탁’ 여부를 바로 알아볼 수 있어요. 임대인이 신탁회사에게 신탁 계약을 맺었다는 것이 ‘갑구’에 적혀 있다면, 진짜 소유권은 임대인에게 없어요. 보통은 임대인이 곧 집의 소유주인지 확인한 뒤 계약서에 서명을 하는데, 만약 신탁등기라면 이렇게 계약해선 안 돼요.
신탁등기 전세사기 예방법 - 잘못된 계약을 하지 않으려면, 등기부등본을 통해 부동산이 신탁되었는지 여부를 확인해야만 해요. 만일 등기부등본을 확인했는데 신탁등기가 되어 있다면 ‘신탁원부’라는 서류를 추가적으로 확인해야만 하고요. 신탁원부란, 위탁자와 수탁자 그리고 수익자와 신탁관리인의 성명 및 주소, 신탁의 목적, 신탁재산의 관리방법, 신탁 종료의 사유 등을 포함한 서류예요. 따라서 신탁원부를 확인하면 누구에게 이 부동산을 임대해줄 권한이 있는지, 해당 부동산을 담보로 받은 대출이 있는지 등을 알 수 있어요. 참고로 이 신탁원부는 인터넷 으로는 발급받을 수 없기 때문에 등기소에서 직접 발급받아야만 해요. 그리고 한 가지 말씀드리자면, 신탁원부를 통해 권리관계를 파악하는 일이 개인에게는 다소 생소하고 어려울 수 있기 때문에 전문가의 도움을 받기를 추천해요.

규칙 :
1. 질문이 특정 사기 유형에 해당하지 않는다면 먼저 전세사기 유형 2가지에 대해 답변에서 설명한다.
2. 해당하는 유형별 대처 방법을 반드시 답변에 포함한다.

위 내용과 규칙을 참고해서 다음 Question에 1500단어 정도로 답변해.
Question :  {text}
"""
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# prompt template가 1817이고, 사용자의 질문을 고려해서 1800으로 산정
# llm = ChatOpenAI(temperature=0, max_tokens=1800)
llm = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0, max_tokens=1800)

check_lessor_chain = LLMChain(
    llm=llm,
    prompt=chat_prompt
)
def return_check_lessor_chain():
    return check_lessor_chain