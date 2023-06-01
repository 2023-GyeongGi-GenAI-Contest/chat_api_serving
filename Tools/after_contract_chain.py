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
너는 사용자의 전세사기 예방을 돕는 챗봇이야. 너는 여러 챗봇 중 "계약한 직후" 시점의 사용자를 담당해
"""
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

human_template="""
계약한 직후 생길 수 있는 전세사기 유형에는 3가지가 있습니다. 아래는 전세사기의 유형과 각 유형에 대한 사례와 예방방법입니다.

1. 전세계약 당일 임대인 변경 및 주택담보대출 실행
사례 : 2021년 4월, A씨는 서울 동작구 상도동의 한 주택에 입주하게 되었어요. 근저당이 없어 권리관계가 깨끗한 집이었죠. A씨는 전입신고와 동시에 확정일자까지 받았 어요. 이쯤 되면 세입자로서 할 수 있는 모든 일을 다 했다고 볼 수 있지요. 그런데 어느 날, 법원으로부터 고지서 한 통을 받게 되었어요. 집이 경매에 넘어가게 되었으니 퇴거해달라는 내용이었죠. 어떻게 된 일일까요? 알고 보니 전입신고 당일, 임대인 B씨가 자산이 거의 없다시피 한 C씨에게 소유권을 넘겼고, C씨는 주택담보대출을 받았어요. 그러다 대출을 갚을 여력이 되지 않았던 C씨는 잠적했고, 집은 경매에 넘어가게 된 거죠.
예방 방법 1 - 전세계약 당일 잔금을 치르기 전에 등기부등본을 확인, 만일 임대인을 변경하는 내용의 등기가 접수되었다면 ‘신청사건 처리 중’으로 명시되기 때문이죠. 이 경우 잔금을 지급하지 말고 어떤 등기가 접수되어 처리중인지 확인해야만 해요. 등기부등본은 이사 당일 뿐 아니라 2-3일 뒤에도 살펴보는 것이 좋아요.
예방 방법 2 - 특약사항을 넣어요, 계약서를 쓸 때, 계약 체결일 다음날까지 소유권 변경, 근저당 설정 등의 행위를 일체 하지 않기로 하는 특약사항을 넣어야 해요.
예방 방법 3 - 불안하다면 전세권을 설정하는 방법도 있어요, 전세권이란, 타인의 부동산을 점유하여 그 부동산의 용도에 따라 사용 및 수익을 얻을 수 있는 권리예요. 쉽게 말해 기존의 임대차계약이 집만 빌리는 것이라고 한다면 전세권 설정은 집과 함께 그 집에 대한 권리관계까지 ‘내’가 차지한다고 볼 수 있어요.

2. 이중계약
사례 : 한 청년 A는 마음에 드는 전세 물건을 찾아 권리관계를 확인하고 임대인을 직접 만나 계약을 체결하였어요. 그러나 이사 당일, A는 자기 집에 다른 세입자 B가 있다는 사실을 알게 되었어요. 임대인이 하나의 주택을 대상으로 2명에게 보증금을 받아 가로챈 것이죠. 이처럼, 임대인이 새로운 임대인에게 보증금을 받은 후 기존 세입자에게 반환하지 않고 잠적하는 사례가 종종 있다고 해요. 하나의 주택을 대상으로 두 사람 이상과 임대차계약을 체결하는 임대인의 목적은 주로 액수가 큰 보증금을 두 배, 세 배로 받는 거죠. 이러한 행위는 그 자체로 불법이므로 이로 인해 누군가 금전적 손해를 입게 된다면 해당 임대인은 배임죄나 사기죄의 명목으로 형사 소송까지 갈 수 있어요.
예방 방법 1 - 깡통주택을 조심해야 해요, 현재까지의 사기 사례를 살펴보면, 보증금이 주택가격과 비슷할수록 이런 사기가 발생할 가능성이 커요. 그래서 보증금을 비롯해 이미 임대인이 집을 담보로 너무 많은 빚을 지고 있다면, 이 빚들의 총합이 주택가격의 80%를 넘는 깡통주택이라면 좀 더 고민해보세요.
예방 방법 2 - 기존 세입자가 이사를 나갔는지 확인하고 잔금을 지급, 임대인이 기존 세입자의 보증금을 돌려줘야 그 사람이 이사를 나갈 텐데, 만약 기존 세입자의 보증금도 제때 돌려주지 않고, 나한테도 보증금도 받아 간 채 잠적하면 안 되니까요. 만약, 임대인과 기존 세입자 모두가 합의하고 내가 보증금을 기존 세입자에게 직접 입금하는 경우가 생길 수도 있어요. 그럴 때는 계약서의 특약사항에 보증금을 기존 세입자의 계좌로 입금한다는 사실을 정확히 적어두어야 해요.

3. 선순위 근저당, 신탁등기 말소 등 특약조건 불이행
사례 : 이사를 앞둔 A씨는 여러 번 집을 보던 도중 마음에 쏙 드는 집을 발견했어요. 곧바로 등기부등본을 떼어 보았죠. 등기부등본을 자세히 살펴보던 A씨는 그 집이 신탁 등기가 되어있는 집이라는 사실을 알게 되었어요. 신탁 부동산은 위험할 소지가 크다는 말을 들었던 A씨는 계약을 고민했어요. 그러자 임대인은 신탁등기를 말소할 테니 걱정하지 말라며 A씨를 안심시켰어요. 결국, 계약서 특약사항에 임대인이 신탁등기를 말소할 것을 적는다는 조건으로 계약하기로 했어요. 마음을 놓은 A씨는 무사히 이사를 마쳤어요. 그런데! 전입신고를 마치고 등기부등본을 떼어 본 A씨는 깜짝 놀랐어요. 아직 신탁등기가 말소되지 않았던 거예요!
예방 방법 - 특약사항에 임대인의 의무를 명확하게 기재하도록 해야 해요. 그러면 근저당권을 말소하지 않을 시, 전세 계약을 취소할 수 있어요. 필요하다면 임대인이 계약서 특약을 지키지 않아서 세입자에 생긴 손해에 대한 배상도 요구해볼 수 있겠죠. 특약을 쓴 뒤, 무엇보다 중요한 것은 임대인이 근저당권 또는 신탁등기를 말소하는지 두 눈으로 직접 확인하는 일이에요. 예를 들어 임대인이 신탁등기를 말소하겠다는 특약을 썼다고 해도, 어차피 신탁등기에서 임대 권한을 가진 것은 신탁회사이기 때문에 유의해야 해요. 아예 계약서에 보증금을 입금할 신탁회사를 정확히 적어두고, 보증금을 신탁회사로 입금하는 게 더 나을 수 있어요.

규칙 :
1. 질문이 특정 사기 유형에 해당하지 않는다면 먼저 전세사기 유형 3가지에 대해 답변에서 설명한다.
2. 해당하는 유형별 예방방법을 반드시 답변에 포함한다.

위 내용과 규칙을 참고해서 다음 Question에 800단어 정도로 답변해줘.
Question :  {text}
"""
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# prompt template가 2587이고, 사용자의 질문을 고려해서 1200으로 산정
# llm = ChatOpenAI(temperature=0, max_tokens=1200)
llm = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0, max_tokens=1200)

after_contract_chain = LLMChain(
    llm=llm,
    prompt=chat_prompt
)
def return_after_contract_chain():
    return after_contract_chain