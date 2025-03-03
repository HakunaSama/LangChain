###用langchain构建一个聊天机器人

#安装langchain:conda install langchain -c conda-forge

# 确保设置环境变量让LangSmith开始记录跟踪
# export LANGCHAIN_TRACING_V2="true"
# export LANGCHAIN_API_KEY="..."
# 在jupyternotebook中也可以：
import getpass
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()

#### 单独使用语言模型。以openai为例：pip install -qU langchain-openai
import getpass
import os
os.environ["OPENAI_API_KEY"] = getpass.getpass()
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-3.5-turbo")
# 我们已经获取到了模型的运行接口，现在来使用它试一下
from langchain_core.messages import HumanMessage
model.invoke([HumanMessage(content="Hi! I'm Bob")])#为什么使用HumanMessage呢？因为可能还会有一些其他类型的Message，先不要在意
#返回值如下
#AIMessage(content='Hello Bob! How can I assist you today?', 
          #response_metadata={'token_usage': {'completion_tokens': 10
                                             #'prompt_tokens': 12, 
                                             #'total_tokens': 22}, 
                              #'model_name': 'gpt-4o-mini', 
                              #'system_fingerprint': None, 
                              #'finish_reason': 'stop', 
                              #'logprobs': None}, 
          #id='run-d939617f-0c3b-45e9-a93f-13dafecbd4b5-0', 
          #usage_metadata={'input_tokens': 12, 
                          #'output_tokens': 10, 
                          #'total_tokens': 22})
                          
#### 但是现在模型是没有状态概念的，也就是没有记忆，每次都是一次新对话
# 比如继续问一个后续的问题
model.invoke([HumanMessage(content="What's my name?")])
# 返回值如下
#AIMessage(content="I'm sorry, I don't have access to personal information unless you provide it to me. How may I assist you today?", 
          #response_metadata={'token_usage': {'completion_tokens': 26, 
                                             #'prompt_tokens': 12, 
                                             #'total_tokens': 38}, 
                             #'model_name': 'gpt-4o-mini', 
                             #'system_fingerprint': None, 
                             #'finish_reason': 'stop', 
                             #'logprobs': None}, 
          #id='run-47bc8c20-af7b-4fd2-9345-f0e9fdf18ce3-0', 
          #usage_metadata={'input_tokens': 12, 'output_tokens': 26, 'total_tokens': 38})
          
#### 为了解决这个问题，需要将整个对话历史传递给模型，来尝试一下，看起来效果还不错
from langchain_core.messages import AIMessage
model.invoke(
    [
        HumanMessage(content="Hi! I'm Bob"),
        AIMessage(content="Hello Bob! How can I assist you today?"),
        HumanMessage(content="What's my name?"),
    ]
)
#AIMessage(content='Your name is Bob. How can I help you, Bob?', response_metadata={'token_usage': {'completion_tokens': 13, 'prompt_tokens': 35, 'total_tokens': 48}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-9f90291b-4df9-41dc-9ecf-1ee1081f4490-0', usage_metadata={'input_tokens': 35, 'output_tokens': 13, 'total_tokens': 48})

#### 这些是支撑聊天机器人对话的基本概念，接下来可以使用消息历史类来包装我们的模型，
#使其具有状态，能够跟踪模型的输入输出，并存储下来，未来的交互将加载这些消息，并且一并传递给模型链

######################################### 消息历史 #########################################
#### 首先要安装langchain-community模块：pip install langchain_community
# 从 langchain_core.chat_history 模块中导入基础聊天记录类和基于内存的聊天记录实现
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)# 从 langchain_core.runnables.history 模块中导入支持聊天记录功能的 Runnable 类
from langchain_core.runnables.history import RunnableWithMessageHistory
store = {}# 定义一个全局字典，用于存储不同会话(session)的聊天记录
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    根据会话的唯一标识符 session_id 获取对应的聊天记录。
    如果不存在，则创建一个新的 InMemoryChatMessageHistory 实例，并存入全局 store 字典中。
    返回值为 BaseChatMessageHistory 类型的实例。
    """
    if session_id not in store:# 如果全局字典中不存在该 session_id 的聊天记录，则初始化一个新的内存聊天记录
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]# 返回对应 session_id 的聊天记录实例
# 此处将模型包装成支持聊天记录功能的 Runnable 对象，以后我们就用这个对象来进行操作了
# get_session_history 用于动态获取或创建与每个会话关联的聊天记录
with_message_history = RunnableWithMessageHistory(model, get_session_history)
# 我们现在需要创建一个config，每次将它传递给RunnableWithMessageHistory这个包装模型
config = {"configurable": {"session_id": "abc2"}}#代表我们想在abc2这个会话下进行输入
response = with_message_history.invoke(#用包装好的模型进行调用
    [HumanMessage(content="Hi! I'm Bob")],#为什么使用HumanMessage呢？
    config=config,
)
response.content#输出'Hi Bob! How can I assist you today?'
response = with_message_history.invoke(#再调用一次，看看能不能记住我们之前说的
    [HumanMessage(content="What's my name?")],
    config=config,
)
response.content#输出'Your name is Bob. How can I help you today, Bob?'

#### 非常好，现在我们的包装模型可以记住我们之前的聊天记录了，如果我们使用不同的config设置不同的会话，就可以开启新的对话了
config = {"configurable": {"session_id": "abc3"}}
response = with_message_history.invoke(
    [HumanMessage(content="What's my name?")],
    config=config,
)
response.content#输出"I'm sorry, I cannot determine your name as I am an AI assistant and do not have access to that information."
#但是可以继续之前的会话
config = {"configurable": {"session_id": "abc2"}}
response = with_message_history.invoke(
    [HumanMessage(content="What's my name?")],
    config=config,
)
response.content#输出'Your name is Bob. How can I assist you today, Bob?'

######################################### 提示词模板 #########################################
#### 提示词模板帮助我们把原始用户输入信息转换成大语言模型更好处理的格式，
# 本来原始用户输入的只是一个消息，现在我们来让这个消息变的更加复杂一点，
# 首先添加一个带有一些自定义指令的系统消息（但仍然将消息作为输入）
# 然后添加除了消息之外的更多输入

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# 使用 from_messages 方法根据消息列表创建一个聊天提示模板 (ChatPromptTemplate)
# 该模板包含两部分：
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",#一个系统消息，用于设置助手的角色和行为准则
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        #一个消息占位符 (MessagesPlaceholder)，会把"messages"对应的输入对象（比如HumanMessage）解析拼接到prompt里
        MessagesPlaceholder(variable_name="messages"),
    ]
)
# 模板做好了之后，我们把这个模板放入链的最前端
chain = prompt | model# 使用管道运算符 (|) 将提示模板 (prompt) 与模型 (model) 组合，形成一个处理链 (chain)

#### 好了，我们现在有了这个带有提示词模板的模型链，来调用一下
# 请注意，我们现在的输入是一个包含messages键的字典，其中包含消息列表，而不是直接传递消息列表
# 因为这个输入是给提示词模板用的，因为这个输入是给提示词模板用的，提示词模板会把这个字典拼接到输入中给模型
response = chain.invoke({"messages": [HumanMessage(content="hi! I'm bob")]})
response.content# 输出'Hello Bob! How can I assist you today?'
# 我们也可以把这个模型链也做成包装模型，像之前一样实现聊天记录的功能
with_message_history = RunnableWithMessageHistory(chain, get_session_history)
config = {"configurable": {"session_id": "abc5"}}#配置一下会话id
# HumanMessage(content="Hi! I'm Jim")经过prompt将会和提示词模板中的其他提示词拼接
# 但是这里为什么不需要使用字典了呢？不使用字典，prompt怎么知道这个HumanMessage对象需要放入哪个消息占位符里
response = with_message_history.invoke(
    [HumanMessage(content="Hi! I'm Jim")],
    config=config,
)
response.content#输出'Hello, Jim! How can I assist you today?'
response = with_message_history.invoke(
    [HumanMessage(content="What's my name?")],
    config=config,
)
response.content#输出'Your name is Jim.'

#### 我们已经学会了基本的提示词模板的使用，现在让我们的提示词变得更加复杂一点
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
chain = prompt | model
# 请注意我们在提示中添加了一个输入：language，看看这样我们应该怎么调用链并传入我们的输入
response = chain.invoke(
    {"messages": [HumanMessage(content="hi! I'm bob")], "language": "Spanish"}
)
response.content#输出'¡Hola, Bob! ¿En qué puedo ayudarte hoy?'
#可以看到，我们的输入是一个字典，key：messages的value是一个HumanMessage对象，key：language的value是"Spanish"
#prompt模块将会把这个字典的value替换掉对应的占位符

####现在我们把这个更加复杂的链封装在一个消息历史类中，也就是之前说的包装模型
# 这次，由于输入中有多个键，我们需要指定正确的键来保存聊天历史，其他键的内容是不需要保存到聊天记录的
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    #input_messages_key 指定从输入字典中哪个键提取聊天消息（此处为 "messages"）
    #为什么需要指定呢，因为其他的键，也就是language的内容是不需要保存到聊天记录里的
    input_messages_key="messages",
)
config = {"configurable": {"session_id": "abc11"}}
response = with_message_history.invoke(#传入一个包含多个键的输入字典：
    # - "messages" 键：包含一个 HumanMessage 对象列表，代表用户的消息（这里是 "hi! I'm todd"）
    # - "language" 键：指定希望生成的响应语言（这里为西班牙语）
    {"messages": [HumanMessage(content="hi! I'm todd")], "language": "Spanish"},
    config=config,
)
#对比一下之前的invoke不需要指定键，因为prompt只有一个占位符
#response = with_message_history.invoke(
#    [HumanMessage(content="Hi! I'm Jim")],
#    config=config,
#)
response.content#输出'¡Hola Todd! ¿En qué puedo ayudarte hoy?'
response = with_message_history.invoke(
    {"messages": [HumanMessage(content="whats my name?")], "language": "Spanish"},
    config=config,
)
response.content#输出'Tu nombre es Todd.'

######################################### 管理对话历史 #########################################
####对于聊天机器人而言，需要管理对话历史，如果不管理，消息列表将会无限增长，溢出上下文窗口，
# 因此需要对传入消息的大小进行限制，
# 需要注意的一点是，需要在 提示模板之前 但是 消息历史加载之后 的消息上执行此操作
# 我们可以通过在提示前添加一个简单的步骤，适当地修改 messages 键，然后将该新链封装在消息历史类中来实现。
# LangChain 提供了一些内置的助手来 管理消息列表。在这种情况下，我们将使用 trim_messages 助手来减少我们发送给模型的消息数量。
#修剪器允许我们指定希望保留的令牌数量，以及其他参数，例如是否希望始终保留系统消息以及是否允许部分消息：
from langchain_core.messages import SystemMessage, trim_messages
# 创建一个消息修剪器 (trimmer) 对象，用于控制消息列表的最大 token 数量
trimmer = trim_messages(
    max_tokens=65,# 修剪后的消息列表中总共允许的最大 tokens 数量为 65
    strategy="last",# 采用保留最后几条消息的策略进行修剪
    token_counter=model,# 使用 model 作为计数 token 的函数或对象
    include_system=True,# 包括系统消息在内进行 token 计算
    allow_partial=False,# 不允许部分消息被保留，修剪时要么完全保留，要么删除
    start_on="human",# 从第一条人类消息开始考虑修剪逻辑
)
# 构建一个包含不同角色消息的对话历史列表
messages = [
    SystemMessage(content="you're a good assistant"),   # 系统消息，用于设定角色或指令
    HumanMessage(content="hi! I'm bob"),                # 人类消息
    AIMessage(content="hi!"),                           # AI 消息
    HumanMessage(content="I like vanilla ice cream"),   # 人类消息
    AIMessage(content="nice"),                          # AI 消息
    HumanMessage(content="whats 2 + 2"),                # 人类消息
    AIMessage(content="4"),                             # AI 消息
    HumanMessage(content="thanks"),                     # 人类消息
    AIMessage(content="no problem!"),                   # AI 消息
    HumanMessage(content="having fun?"),                # 人类消息
    AIMessage(content="yes!"),                          # AI 消息
]
# 调用修剪器，将消息列表修剪为符合 max_tokens 限制的子列表
# 返回的结果将只包含总 tokens 数不超过 65 的消息
trimmer.invoke(messages)
#输出
#[SystemMessage(content="you're a good assistant"),
# HumanMessage(content='whats 2 + 2'),
# AIMessage(content='4'),
# HumanMessage(content='thanks'),
# AIMessage(content='no problem!'),
# HumanMessage(content='having fun?'),
# AIMessage(content='yes!')]

#### 这个修剪器我们已经创建好了，想要在我们的链中使用它，只需要在将message输入传递给prompt之前运行修剪器即可
from operator import itemgetter

from langchain_core.runnables import RunnablePassthrough

chain = (
    # RunnablePassthrough.assign 部分：
        # 使用 itemgetter("messages") 从输入字典中提取 "messages" 键对应的值
        # 将提取到的消息列表通过管道操作符传给 trimmer 进行消息修剪处理
        # 结果会被分配给后续环节中名为 "messages" 的输入键
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | model
)

response = chain.invoke(
    {
        #"messages": 原始消息列表 messages 加上一个额外的 HumanMessage，
        #内容为 "what's my name?"，以扩展对话历史
        "messages": messages + [HumanMessage(content="what's my name?")],
        "language": "English",
    }
)
response.content# 输出："I'm sorry, but I don't have access to your personal information. How can I assist you today?"
# 但是我们询问最近几条消息中的信息，它会记住
response = chain.invoke(
    {
        "messages": messages + [HumanMessage(content="what math problem did i ask")],
        "language": "English",
    }
)
response.content#输出 'You asked "what\'s 2 + 2?"'

#### 现在让我们把它包装在 消息历史 中
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)
config = {"configurable": {"session_id": "abc20"}}
# 来调用一下！
response = with_message_history.invoke(
    {
        "messages": messages + [HumanMessage(content="whats my name?")],
        "language": "English",
    },
    config=config,
)
response.content#输出 "I'm sorry, I don't have access to that information. How can I assist you today?"

######################################### 流式处理 #########################################
# 现在我们有了，能够记住聊天记录，并且会设置最大长度进行裁剪记录，并且有提示词模块的聊天机器人了
# 但是对于聊天机器人来说，非常重要的一个体验是流式处理，实际上，我们只需要让所有链都暴露一个.stream方法即可
# 使用消息历史的链也不例外，可以简单的使用该方法获取流式响应
config = {"configurable": {"session_id": "abc15"}}
for r in with_message_history.stream(
    {
        "messages": [HumanMessage(content="hi! I'm todd. tell me a joke")],
        "language": "English",
    },
    config=config,
):
    print(r.content, end="|")
    #输出：|Hi| Todd|!| Sure|,| here|'s| a| joke| for| you|:| Why| couldn|'t| the| bicycle| find| its| way| home|?| Because| it| lost| its| bearings|!| 😄||
    
    
    
######################################### 对话式RAG #########################################
# 好了，我们已经学会了构建一个简单的聊天机器人，但是在许多问答应用中，用户想要进行反复问话，
# 这意味着我们需要某种形式的"记忆"来记录过去的问题和答案，并能够将这些信息融入当前思考的逻辑
# 现在我们重点关注 添加用于整合历史消息的逻辑
# 将有两种方法：
# 链接，在其中我们始终执行检索步骤
# 代理，在其中我们给予llm自由决定是否，以及如何执行检索步骤（或多个步骤）
# 对于外部知识源，我们将使用LilianWeng的同一篇LLM Powered Autonomous Agents博客文章，来自RAG教程

####设置
# 在这个例子中，我们使用openai嵌入和chroma向量嵌入，但是这里展示的所有内容，适用于任何嵌入和向量存储或检索器
# 使用以下软件包
# 使用该命令可以隐藏正常的输出信息，但如果代码出错，错误信息仍会显示在 Notebook 中。
%%capture --no-stderr
%pip install --upgrade --quiet  langchain langchain-community langchainhub langchain-chroma beautifulsoup4
# 我们需要设置环境变量 OPENAI_API_KEY，可以直接设置或从 .env 文件中加载
import getpass
import os
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass()
# import dotenv
# dotenv.load_dotenv()
# 如果您确实想使用 LangSmith检查您的链或代理内部究竟发生了，请确保设置您的环境变量以开始记录跟踪：
os.environ["LANGCHAIN_TRACING_V2"] = "true"
if not os.environ.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
    
####链
#先来回顾以下我们之前做的聊天机器人
pip install -qU langchain-openai#记得安装以下

import getpass
import os
os.environ["OPENAI_API_KEY"] = getpass.getpass()
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")#创建llm操作对象

import bs4
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 1.加载、分块并索引博客的内容以创建一个检索器。
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),# 使用 WebBaseLoader 从指定的 URL 加载网页内容。
    # bs_kwargs 参数用于传递 BeautifulSoup 的参数，这里通过 SoupStrainer 指定只解析包含特定 class 的 HTML 元素，
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")# 这些元素包括 "post-content"、"post-title" 和 "post-header"。
        )
    ),
)

docs = loader.load()# 调用 load 方法加载网页内容，返回文档列表。只加载博客中的正文内容、标题以及页眉部分，而忽略其它无关内容。

# 创建一个 RecursiveCharacterTextSplitter，用于将文档分块。
#   chunk_size=1000 表示每个块的最大字符数，
#   chunk_overlap=200 表示相邻块之间有 200 个字符的重叠，确保上下文连续性。
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# 这一步会遍历 docs 中的每个文档，根据设定的字符数限制将其拆分成多个较小的文本块，
# 每个块都是一个独立的文档片段，存储在 splits 列表中。
splits = text_splitter.split_documents(docs)#进行分块，生成分块后的文档列表。

# 使用 Chroma.from_documents 方法将分块后的文档列表转换为向量存储（vectorstore）
# 使用 OpenAIEmbeddings() 对每个文本块生成向量嵌入，实现文本的数值化表示，
 # 以便后续通过向量相似度来快速检索相关内容。
## 注意：每个文档块（chunk）会被整体转换成一个向量，而不是对文档中的每个 token 单独进行嵌入。
## 也就是说，整个文本块作为一个输入被传递给 OpenAIEmbeddings 模型，然后生成一个向量来代表这个块的语义信息。
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
# 将向量存储转换为检索器对象，这样我们就能根据用户查询从存储的向量中找出最相关的文档块。
#   输入：通常是一个查询字符串（例如用户的问题或搜索关键字）。在内部，retriever 会对这个查询进行向量化，然后与向量存储中的所有嵌入向量进行相似性比较。
#   输出：一个文档对象列表，每个文档对象通常包含文本内容和可能的元数据。这些文档块按照与查询的相关性排序，最相关的排在最前面。
retriever = vectorstore.as_retriever()

# 2. 将检索器整合到问答链中。
#       定义系统提示（system_prompt），里面有上下文占位符context：
#       - 指明助手角色为问答助手。
#       - 指示使用检索到的上下文（{context}）来回答问题。
#       - 若不清楚答案，则直接说明不知道。
#       - 限制回答最多使用三句话，并保持简洁。
system_prompt = (
    "你是一个问答任务助手"
    "使用以下检索的文档进行回答问题"
    "如果你不知道答案，就说你不知道"
    "限制回答最多使用三句话，并保持简洁。"
    "\n\n"
    "{context}"#检索到的上下文占位符
)

# 使用 ChatPromptTemplate.from_messages 方法创建对话提示模板，现在这个模板中还差context没有填入
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# 创建一个文档问答链，input -(prompt)-> 带有context占位符的newinput -(llm)-> output
# create_stuff_documents_chain 文档填充链创建函数
# 这里比较奇怪的事情是为什么llm写在propmt前面了，真实的处理流程是先propmt后llm
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# 创建一个检索问答链，将检索器（retriever）与问答链整合在一起，
# input -(retriever)-> context+input -(prompt)-> 带有context的newInput -(llm)-> output
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

#### 好了，最终的rag_chain我们知道了它的流程是input -(retriever)-> context+input -(prompt)-> 带有context的newInput -(llm)-> output

response = rag_chain.invoke({"input": "什么是任务分解？?"})
response["answer"]
#输出  "任务分解是指将复杂任务拆分成更小、更简单的步骤，从而使任务更易于代理或模型管理。
#       这个过程有助于引导代理完成实现总体任务所需的各个子目标。可以使用诸如链式思维（Chain of Thought）和
#       树状思维（Tree of Thoughts）等不同技术，将任务分解为逐步执行的流程，
#       从而提升模型的性能，并加深对模型思考过程的理解。"

####添加聊天历史
# 我们现在构建的这个链可以使用输入来检索相关文档，但是在现实中，用户的意图可能需要对话中的上下文才能理解
# 比如：人类: "什么是任务分解？"
#       AI: "任务分解涉及将复杂任务分解为更小、更简单的步骤，以便使代理或模型更易于管理。"
#       人类: "它常见的做法有哪些？"
# 为了回答"它常见的做法有哪些？"这个问题，我们希望模型理解它指的是"任务分解"
# 这个问题虽然大模型本身是可以理解一点，但是我们还是希望可以在我们这里进行一些优化
# 为了解决这个问题，我们从两个方面考虑
#   1.提示词:希望可以更新提示词，把我们的一些历史消息，也放到提示词里
#   2.把问题进行上下文融合，添加一个子链，它能够获取用户的问题，并且通过聊天历史的上下文，来重新表述这个问题
#       这可以简单理解为构建一个新的"历史感知"检测器，
#       之前我们有：查询->retriever
#       现在我们想要：（查询，聊天历史）->llm->重新表达的查询->检索器

##问题 上下文 化
# 我们使用一个提示词，其中包含名为"chat_history"的MessagesPlaceholder 变量。然后用它将消息列表插入提示词中，
# 这些历史消息将会在系统消息（system提示词）
# 我们利用一个辅助函数 create_history_aware_retriever（历史感知检索器）来处理这一步，这个函数管理chat_history为空的情况，否则按顺序应用提示词|大模型|输出解析器（）|检索器
# create_history_aware_retriever构建一个接受输入和聊天历史作为输入的链，并具有与检索器相同的输出模式
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

#先来定义一个系统提示词
contextualize_q_system_prompt = (
    "给定一段聊天记录以及最新的用户问题 "
    "该问题可能引用了chat history中的context, "
    "将其重新表述为一个独立的问题，使其在没有上下文的情况下也能被理解。 "
    "不要回答这个问题，只在必要时对其进行重新表述，否则原样返回。"
)
#来定义上下文感知提示词模板，由上下文感知系统提示词+chat_history+input
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
#创建历史感知检索器，由大模型、检索器和上下文感知提示词构成
# 当调用了create_history_aware_retriever 函数后，返回的history_aware_retriever 对象就内置了将输入数据中的chat_history 自动填入contextualize_q_prompt 占位符的逻辑，
# 这里的llm是为了和已经填充好了chat_history的contextualize_q_prompt将用户的问题结合聊天历史（还有模板啦）和当前输入重新表述出一个独立问题
# 然后将这个独立问题交给外部文档检索器进行检索
#### 简单点说输入一段数据，数据中的聊天历史会被填入到contextualize_q_prompt的chat_history中，llm利用这个模板生成新问题，新问题将调用retriever查找相关文档或信息。
# input->融合了历史记录和外部检索的newInput->llm
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

#### 这条链将输入查询的重述（结合了历史记录的）添加到我们的检索器之前
# 但是这里的retriever不是我们之前创建的外部知识的检索器么
# 现在我们可以构建完整的问答链，我们要把之前的检索器替换为我们新的history_aware_retriever
# 我们还是使用之前的create_stuff_documents_chain（创建文档填充链函数）来创建一个问答链，
# 输入键为context、chat_history和input，它接受检索到的上下文以及对话历史和查询以生成答案。
# 使用create_retrieval_chain构建最终的rag_chain，该链按顺序应用history_aware_retriever（历史感知检索器）和question_answer_chain（问答链）
# 保留中间输出，例如检索到的上下文以便于使用，它的输入键为input和chat_history，输出中包括input、chat_history、context 和 answer
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),# 聊天历史占位符：MessagesPlaceholder("chat_history") 会在执行时填充之前的对话记录，以便模型了解上下文信息。
        ("human", "{input}"),
    ]
)#这个提示把用户输入、系统指令、历史对话放在一起

#输入->qa_prompt->llm
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
#输入->history_aware_retriever融合历史消息与检索生成新问题->qa_prompt->llm
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

