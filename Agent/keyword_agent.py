from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate

system_template="""
[키워드 목록]:
- 소재지: 부동산이 위치해 있는 지번주소
- 지목: 토지의 주된 용도에 따라 토지의 종류를 구분하여 지적공부에 등혹한 것을 말합니다. 지목은 전ㆍ답ㆍ과수원ㆍ목장용지ㆍ임야ㆍ광천지ㆍ염전ㆍ대(垈)ㆍ공장용지ㆍ학교용지ㆍ주차장ㆍ주유소용지ㆍ창고용지ㆍ도로ㆍ철도용지ㆍ제방(堤防)ㆍ하천ㆍ구거(溝渠)ㆍ유지(溜池)ㆍ양어장ㆍ수도용지ㆍ공원ㆍ체육용지ㆍ유원지ㆍ종교용지ㆍ사적지ㆍ묘지ㆍ잡종지로 구분하여 총28개로 분류합니다.
- 구조: 건물의 구조. 예: 철근콘크리트조
- 용도: 건물의 용도. 예: 아파트, 상가 등
- 임대할부분: 임대할 공간을 정확히 명시함.
- 보증금: 일정한 채무를 담보하기 위하여 채무자가 채권자에게 미리 교부하는 금전 또는 입찰(入札)·경매(競買)·유상계약에서 계약 이행의 담보로서 납입하는 금전을 말한다.
- 중도금: 계약금을 낸 후 남은 금액 중 일부를 나눠 내는 돈.
- 잔금: 입주전 매매가의 남은 금액.
- 임대인: 임대차 계약에 의하여 임대료를 받고 타인(임차인)에게 물건,부동산을 빌려 주는 사람. 즉, 부동산의 소유주
- 임차인: 임대차 계약에 의하여 임대료를 주고 임대인에게 물건, 부동산을 빌리는 사람.
- 대리인: 대리로서 법률행위를 할 수 있는 지위에 있는 자를 말함.

[부동산 계약서]
{question}


위 [키워드 목록]을 참고하여 해당 [부동산 계약서]에 포함된 키워드를 추출해줘.
name에는 해당 키워드를 작성해줘.
describe에는 해당 키워드에 대한 설명을 작성해줘.

[응답 양식]은 아래 예시와 같은 JSON 형식으로 작성해야 합니다.
[응답 양식] 출력 예시:
{{"keyword":[{{"name":"키워드1", "describe":"키워드1에대한 설명"}}, {{"name":"키워드2", "describe":"키워드2에대한 설명"}}, {{"name":"키워드3", "describe":"키워드3에대한 설명"}}]}}
"""

system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])

llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", max_tokens=12000)
llm_chain = LLMChain(
    llm=llm,
    prompt=chat_prompt
)

def validate_data(data):
    stack = []
    for item in data:
        if item == "[":
            stack.append("[")
        elif item == "{":
            stack.append("{")
        elif item == "]":
            if len(stack) == 0 or stack[-1] != "[":
                return False
            stack.pop()
        elif item == "}":
            if len(stack) == 0 or stack[-1] != "{":
                return False
            stack.pop()

    return len(stack) == 0

def generate_keyword(request):
    res = llm_chain(str(request))['text']
    if validate_data(res) == True:
        return res
    else:
        generate_keyword(request)
