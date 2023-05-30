from langchain.memory import ConversationSummaryMemory, ChatMessageHistory, ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from db_model.db_redis import get_chat

def build_memory(session_id):
    # history = ChatMessageHistory()

    decoded_value = get_chat(session_id)

    # 요약 메모리 (장기기억 가능)
    # if decoded_value != None:
    #     for i in range(len(decoded_value['UserChat'])):
    #         history.add_user_message(decoded_value['UserChat'][i]['chatmessage'])
    #
    #     for i in range(len(decoded_value['AIChat'])):
    #         history.add_ai_message(decoded_value['AIChat'][i]['chatmessage'])
    #
    # memory = ConversationSummaryMemory.from_messages(llm=ChatOpenAI(temperature=0), memory_key="chat_history",
    #                                                  chat_memory=history, return_messages=True)

    memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history", return_messages=True)

    if decoded_value != None:
        for i in range(len(decoded_value['UserChat'])):
            memory.save_context({"input": decoded_value['UserChat'][i]['chatmessage']}, {"ouput": decoded_value['AIChat'][i]['chatmessage']})

    return memory




