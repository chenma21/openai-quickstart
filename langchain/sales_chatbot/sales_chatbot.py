import gradio as gr

from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


def initialize_sales_bot(vector_store_dir: str="real_estates_sale"):
    db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings(base_url="https://api.xiaoai.plus/v1",api_key="sk-IrIgzFee3yL0i2ccD5Bb6820566843EcBcDd1b94B034D01a"),allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model_name="gpt-4", temperature=0, base_url="https://api.xiaoai.plus/v1",api_key="sk-IrIgzFee3yL0i2ccD5Bb6820566843EcBcDd1b94B034D01a")
    
    global SALES_BOT    
    SALES_BOT = RetrievalQA.from_chain_type(llm,
                                           retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                     search_kwargs={"score_threshold": 0.8}))
    # 返回向量数据库的检索结果
    SALES_BOT.return_source_documents = True

    return SALES_BOT

def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")

    # TODO: 从命令行参数中获取
    enable_chat = True

    # 將歷史對話格式化為字符串
    formatted_history = "\n".join([f"客戶: {h[0]}\n銷售助理: {h[1]}" for h in history])

    ans = SALES_BOT({"query": message, "chat_history": formatted_history})
    
    if ans["source_documents"] or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        if len(ans["source_documents"]) == 0:
            template = """
            你是一個專業的車輛銷售助理。你的回答應該自然、友好，並且要有連貫性。請記住以下幾點：
            1. 總是要理解並回應客戶最新的問題，同時考慮之前的對話內容。
            2. 如果客戶只提供簡短的回答，試著根據上下文推斷他們的意思。
            3. 避免重複提問，除非真的需要澄清。
            4. 用自然的語氣交談，就像真人一樣。避免過於正式或機械的表達。
            5. 如果不確定，可以做出合理的假設，並在回答中體現出來。
            6. 不要提及你是一個大模型的概念以及語氣，想象你是一個專業的4S店助理。

            以下是之前的對話：
            {history}

            客戶的最新回答是：{question}

            請給出一個自然、連貫的回覆，要像真人銷售助理一樣：
            """
            llm = ChatOpenAI(model_name="gpt-4", temperature=0, base_url="https://api.xiaoai.plus/v1",api_key="sk-IrIgzFee3yL0i2ccD5Bb6820566843EcBcDd1b94B034D01a")

            prompt = PromptTemplate(template=template, input_variables=["history", "question"])
            chain = LLMChain(llm=llm, prompt=prompt)

            response = chain.run(history=formatted_history, question=message)
            
            return response    
        else:
            return ans["result"]
    else:
        return "這個問題我要問問領導~"
    

def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="汽車销售",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")

if __name__ == "__main__":
    # 初始化房产销售机器人
    initialize_sales_bot()
    # 启动 Gradio 服务
    launch_gradio()
