from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import re
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

from langchain.agents import Tool
from Tools.after_contract_chain import after_contract_chain
from Tools.after_contract_period_chain import after_contract_period_chain
from Tools.after_home_chain import after_home_chain
from Tools.before_contract_chain import before_contract_chain
from Tools.broker_chain import broker_chain
from Tools.check_lessor_chain import check_lessor_chain
from Tools.fake_chain import fake_llm_chain
from Tools.select_home_chain import select_home_chain


##########################################################
#### 이전의 legacy agent 코드 입니다. base_agent를 사용하세요 ####
##########################################################


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

docs = [Document(page_content=t.description, metadata={"index": i}) for i, t in enumerate(custom_tools)]

vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())

retriever = vector_store.as_retriever(k=5)

def get_tools(query):
    docs = retriever.get_relevant_documents(query)
    return [custom_tools[d.metadata["index"]] for d in docs]

# Set up the base template
template = """Answer the following questions as best you can, but speaking as a pirate might speak. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to speak as a pirate when giving your final answer. Use lots of "Arg"s

Question: {input}
{agent_scratchpad}"""

from typing import Callable


# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    ############## NEW ######################
    # The list of tools available
    tools_getter: Callable

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        ############## NEW ######################
        tools = self.tools_getter(kwargs["input"])
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in tools])
        return self.template.format(**kwargs)

prompt = CustomPromptTemplate(
    template=template,
    tools_getter=get_tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"]
)


class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

output_parser = CustomOutputParser()
# 마지막에 Agent를 사용할 때는 gpt3을 사용하자.
llm = OpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0, max_tokens=1000)

# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)

tools = get_tools("나는 현재 계약기간이 끝난 시점이야, 내가 유의해야할 전세사기의 유형에는 어떤 것이 있을까?")
tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names
)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

def get_reply(input):
    return agent_executor.run(input + ' Please summarize it in 500 words.')