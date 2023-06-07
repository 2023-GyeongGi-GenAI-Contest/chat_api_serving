from langchain import OpenAI

from Tools.after_contract_chain import after_contract_chain
from Tools.after_contract_period_chain import after_contract_period_chain
from Tools.after_home_chain import after_home_chain
from Tools.before_contract_chain import before_contract_chain
from Tools.broker_chain import broker_chain
from Tools.check_lessor_chain import check_lessor_chain
from Tools.fake_chain import fake_llm_chain
from Tools.select_home_chain import select_home_chain

from langchain.agents import Tool
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents import AgentType

custom_tools = [
    Tool(
        name="1.'집 고를때' 시점에 발생하는 전세사기에 대한 답변을 하는 챗봇",
        func=select_home_chain.run,
        description="""
        집을 고르는 시점에 발생하는 전세사기에 대한 답을 해야할 때 유용합니다.
        등기부등본에 대한 질문에 답을 해야할 때 유용합니다.
        표제부, 갑구, 을구와 같은 등기부등본에 대한 정보에 대해 답하는데 유용합니다.
        깡통주택 사기에 대한 질문에 답을 해야할 때 유용합니다.
        """,
        return_direct=True,
    ),
    Tool(
        name="2.'임대인 확인할 때' 시점에 발생하는 전세사기에 대한 답변을 하는 챗봇",
        func=check_lessor_chain.run,
        description="""
        임대인을 확인하는 시점에 발생하는 전세사기에 대한 답을 해야할 때 유용합니다.
        '가짜 임대인 사기' 즉, 실제 임대인이 아닌 사람이 임대인 대신 계약을 하러온 상황에 대한 질문에 답을 해야할 때 유용합니다.
        '신탁회사의 동의 없는 계약' 즉, 임대인이 신탁회사의 동의 없이 나와 계약을 하는 상황에 대한 질문에 답을 해야할 때 유용합니다.
        """,
        return_direct=True,

    ),
    Tool(
        name="3.'계약서 작성할 때' 시점에 발생하는 전세사기에 대한 답변을 하는 챗봇",
        func=before_contract_chain.run,
        description="""
        계약서를 작성하는 시점에 발생하는 전세사기에 대한 답을 해야할 때 유용합니다.
        '월셋집을 전셋집으로 둔갑시킨 경우'에 대한 질문에 답을 해야할 때 유용합니다.
        '선순위 임차보증금 및 근저당 허위 고지' 즉, 다른 세입자들의 임차보증금 규모를 속여 전세계약을 체결한 경우에 대한 질문에 답을 해야할 때 유용합니다.
        """,
        return_direct=True,

    ),
    Tool(
        name="4.'계약한 직후' 시점에 발생하는 전세사기에 대한 답변을 하는 챗봇",
        func=after_contract_chain.run,
        description="""
        계약한 직후 시점에 발생하는 전세사기에 대한 답을 해야할 때 유용합니다.
        '전세계약 당일 임대인 변경 및 주택담보대출 실행' 유형의 사기에 대한 질문에 답을 해야할 때 유용합니다.
        '이중계약' 유형의 사기에 대한 질문에 답을 해야할 때 유용합니다.
        '선순위 근저당, 신탁등기 말소 등 특약조건 불이행' 즉, 계약서에 명시된 특약조건을 불이행하는 유형의 사기에 대한 질문에 답을 해야할 때 유용합니다.
        """,
        return_direct=True,

    ),
    Tool(
        name="5.'입주한 이후' 시점에 발생하는 전세사기에 대한 답변을 하는 챗봇",
        func=after_home_chain.run,
        description="""
        입주한 이후 시점에 발생하는 전세사기에 대한 답을 해야할 때 유용합니다.
        '미납국세 및 임금채권 우선 변제' 즉, 임대인이 세금을 내지 않아 집이 경매에 넘어가도 보증금을 돌려받지 못하는 경우의 사기에대한 답을 해야할 때 유용합니다.
        '세입자의 전출신고 후 벌어지는 전세사기' 즉, 임대인이 갑자기 잠시 전출신고를 해줄 수 있냐는 요구를 하는 유형의 사기에대한 답을 해야할 때 유용합니다.
        """,
        return_direct=True,

    ),
    Tool(
        name="6.'계약기간이 끝난 후' 시점에 발생하는 전세사기에 대한 답변을 하는 챗봇",
        func=after_contract_period_chain.run,
        description="""
        계약기간이 끝난 후 시점에 발생하는 전세사기에 대한 답을 해야할 때 유용합니다.
        즉, 임대인이 계약기간이 끝났는데 보증금을 돌려주지 않는 경우에 대한 답을 해야할 때 유용합니다.
        """,
        return_direct=True,

    ),
    Tool(
        name="7.'명의도용 대출 사기', '브로커를 통한 전세보증 사기'에 대한 답변을 하는 챗봇",
        func=broker_chain.run,
        description="'명의도용 대출 사기', '브로커를 통한 전세보증 사기'에 대한 답을 해야할 때 유용합니다.",
        return_direct=True,

    ),
    # Tool(
    #     name="추가정보를 요청하는 챗봇",
    #     func=fake_llm_chain.run,
    #     description="질문에 특정 시점에대한 명시가 되어있지 않은 경우 이 도구를 사용합니다.",
    #     return_direct=True,
    #
    # ),
]
memory = ConversationBufferWindowMemory(memory_key="chat_history", k=1, return_messages=True)

llm=ChatOpenAI(temperature=0, max_tokens=1000)
agent_chain = initialize_agent(custom_tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, memory=memory, verbose=True)


def get_reply(input):
    return agent_chain.run(input + ' Please summarize it in 300 words.')