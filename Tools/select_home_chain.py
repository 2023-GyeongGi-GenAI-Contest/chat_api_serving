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
너는 사용자의 전세사기 예방을 돕는 챗봇이야. 너는 여러 챗봇 중 "집을 고르는 단계" 시점의 사용자를 담당해
"""
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

human_template="""
"집을 고르는 시점"에서의 전세사기의 유형

1. 깡통주택

깡통주택이란? :
깡통주택이란, 임대인의 집이 경매로 넘어갔을 때, 내 보증금이 떼이게 되는 집을 통틀어 가리켜요. 위 그림처럼 매매가격의 대다수를 세입자의 보증금과 빚으로 채우고 있는 집이어서, 사실상 임대인의 몫은 거의 없는 집들이 흔히 깡통주택이에요.
보통 깡통주택을 전세로 내놓는 임대인들은 ‘갭투기’를 통해 해당 주택을 사들였을 거예요. 이때의 갭(차이)은 ‘매매가-전세가’를 말해요. 그 ‘갭’마저도 은행에서 대출을 받고요. 이렇게 되면, 임대인이 스스로 자유롭게 융통할 수 있는 돈은 거의 없죠. 보증금도, 은행 대출금도 전부 언젠가 타인에게 돌려줘야 하는 돈이니까요. 그렇기 때문에 만약 집이 경매에 넘어가기라도 하면, 세입자는 보증금을 떼이게 되고요. 경매에 넘어가지 않더라도, 계약이 끝나고 이사를 나갈 때 제때 보증금을 돌려주지 못할 수도 있어요. 임대인이 제 돈 주고 산 집이 아니니, 돌려줄 돈이 없는 경우에 특히 더 그렇죠. 이런 경우, 다음 세입자가 구해지지 않아서 보증금을 줄 수 없다고 해요. 또는, 집 가격이 하락해 집값이 전세보증금보다 적어졌을 때에도 이사를 가려는 세입자에게 보증금을 돌려줄 수 없게 되는 일이 발생할 수도 있어요.
일반적으로 깡통주택은 보증금 + 대출금의 총합이 집값의 80%를 넘는 집을 의미해요. 즉, 임대인이 자기 돈을 얼마나 들여서 집을 갖고 있느냐가 중요해요. 깡통주택의 임대인은 불상사가 생겼을 때 빚 상환을 회피해버릴 수 있기 때문이죠. 그렇기에 깡통주택은 마치 시한폭탄 같은 존재라고 할 수 있어요.
깡통주택에 전세로 들어간 순간부터 보증금을 돌려받지 못할 가능성이 생겨요. 예를 들어, 주택A,B 모두 경매에 넘어갔을 때, 주택A에서는 보증금을 전부 돌려받지 못할 수 있어요. 경매에 넘어가지 않더라도, 집값이 떨어지거나 임대인의 경제적 사정이 어려워지면 보증금을 돌려주지 못하게 될 수도 있어요. 애초에 임대인의 돈은 거의 없는 깡통주택이기 때문에, 임대인이 돈이 없다며 보증금을 돌려주지 않을 수도 있죠. 그러니 반드시 등기부등본 서류를 통해 이 집에 빚이 얼마나 많은지 확인해야 해요.

사례 : 한동안 이슈가 되었던 ‘대구 깡통전세사기 사건’을 아시나요? 올해 초, 임대인 장 모 씨가 세입자 50여 명에게 약 68억 원의 보증금을 돌려주지 않았다는 사실이 드러나 충격을 안겨줬어요. 장 씨는 대구에서 다가구주택 13채를 갭투기로 사들인 뒤, 세입자에게 집을 빌려줬어요. 이후 보증금을 돌려받아야 하는 세입자들에게 돈을 돌려주지 않았어요. 그 결과, 장 씨는 2019년과 2020년에 각각 징역 3년, 4년형을 선고 받았어요. 이 사건이 바로 깡통주택 전세사기예요. 깡통전세 사기를 당하면, 임대인이 처벌을 받게 된다고 해도, 세입자들 모두가 보증금 전부를 돌려받을 수 있는 건 아니라고 해요. 그러니 더욱 유의해야 해요!

등기부등본을 확인해야하는 이유 :
보증금을 지키기 위해서는, 어떤 집을 선택할지 고를 때부터 그 집의 등기부등본을 꼭 확인해야 해요. 아무리 채광이 좋은 집이라고 해도, 빚이 너무 많아서 나중에 내 보증금을 제때 돌려주지 못한다면, 분명 위험한 집이니 피해야겠죠! 반드시 피해야 하는 집에 대한 단서가 바로 등기부등본에 있답니다.
등기부등본에서는 ①임대인이 누구인지, ②빚이 얼마나 많은지(근저당권 등), ③혹시 누군가의 보증금을 떼어먹은 전적은 없는지(임차권) 등의 정보를 확인할 수 있어요.그리고 등기부등본은 계약하기 전에도, 계약 이후 잔금을 치르기 직전에도 반드시 확인해야 해요. 왜냐하면 내가 입주하기 전에 임대인이 나와 계약한 집을 담보로 빚을 지거나, 아니면 매매를 해버릴 수도 있기 때문이에요.
약, 아파트나 오피스텔 같은 집합건물을 알아보고 있다면, ‘토지’와 ‘건물’을 하나의 등기부등본에서 함께 확인할 수 있어요. 하지만 다가구주택, 다세대주택, 단독주택 등 집합 건물이 아닌 집을 알아보는 중이라면, 토지에 대한 등기부등본도 함께 확인해봐야 해요. 건물의 등기부등본이 깨끗해도, 토지 등기부등본이 복잡하게 꼬여 있는 집도 있답니다. 등기부등본은 임대인이 아니더라도 누구나 대법원 인터넷 등기소*에서 발급 및 열람할 수 있으므로 확인을 주저하지 마세요. 열람은 700원! 700원으로 나의 보증금을 안전하게 지켜요!
※단, 임차권자가 임차권등기를 아직 마치지 못한 경우에는 등기부등본에 적히지 않을 수 있습니다. 이 경우, 전입세대 열람내역, 확정일자 부여 현황 등으로 확인해야 합니다.
대법원 인터넷 등기소 링크 : http://www.iros.go.kr/PMainJ.jsp

등기부등본 보는 법 :
등기부등본은 크게 ①표제부 ②갑구 ③을구, 이 세 가지 영역으로 구성되어 있어요. 
표제부는 부동산에 대한 일반 현황을 확인할 수 있고, 갑구에서는 소유권에 대한 권리를 확인 할 수 있고, 을구에서는 소유권 외 권리를 확인할 수 있어요.
등기부등본의 ‘갑구’에는 소유권에 관한 내용이 적혀 있어요. 말 그대로 이 집을 소유하고 있는 사람이 누구인지에 대한 정보가 있죠. 그리고 그러한 소유권을 제한하고 있는 문제 상황은 없는지 살펴볼 수 있어요. 압류 / 가압류 / 가등기 등 주택에 대한 소유권을 제한할 수 있는 조건이 기록되어 있죠. 예를 들어, 소유주 이름이 ‘김ㅇㅇ’이라고 해도, 세무서가 ‘압류’를 걸어뒀다면, 그 집은 위험한 상태인 거예요. 압류 / 가압류 / 가등기 등 소유권을 침해할 수 있는 요소가 등기부등본에 적혀 있다면 전세계약의 효력 자체가 사라질 수도 있으니까 전세계약을 체결하지 않는 것이 좋아요.
등기부등본의 ‘을구’에는 이 집에 얽힌 돈 얘기들이 가득 적혀 있어요. ①이미 이 집에 전세권을 가진 사람이 있거나 ②임대인이 집을 담보로 많은 빚을 졌거나, 하는 것들을 을구에서 확인할 수 있어요.
등기부 등본을 보는 방법에 대한 자세한 내용 링크 : https://www.khug.or.kr/jeonse/web/s03/s030105.jsp

규칙 :
1. 질문이 특정 사기 유형에 해당하지 않는다면 답변에는 깡통주택의 정의가 포함되어야 한다.
2. 등기부등본에 대해서 말할 때는 표제부, 갑구, 을구를 최대한 자세하게 설명해야한다.

위 내용과 규칙을 참고해서 다음 Question에 500단어 정도로 답변해.
Question :  {text}
"""
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# prompt template가 3000 이며 사용자의 질문을 고려해서 600으로 산정
# llm = ChatOpenAI(temperature=0, max_tokens=1000)
llm = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0, max_tokens=600)

select_home_chain = LLMChain(
    llm=llm,
    prompt=chat_prompt
)
def return_select_home_chain():
    return select_home_chain