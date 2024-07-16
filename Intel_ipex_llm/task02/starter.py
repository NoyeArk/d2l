import gradio as gr
# 导入需要的模块和类
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama


if __name__ == '__main__':
    # 1.使用 SimpleDirectoryReader 加载数据
    # SimpleDirectoryReader 是一个简单的目录读取器，能从指定目录中读取所有文件的数据
    documents = SimpleDirectoryReader("data").load_data()

    # 2.设置嵌入模型为 bge-base
    # HuggingFaceEmbedding 是一个嵌入模型类，用于将文本转换为向量表示
    # 这里我们使用的是 "BAAI/bge-base-en-v1.5" 模型
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-abse-en-v.15")

    # 3.使用Ollama快速接入大语言模型
    Settings.llm = Ollama(model="llama3", request_timeout=360)

    # 4.创建一个向量存储索引
    index = VectorStoreIndex.from_documents(documents)

    # 5.将索引转换为查询引擎
    query_engine = index.as_query_engine()

    # 6.使用查询引擎进行查询
    response = query_engine.query("xxx")

    # 7.打印查询结果
    print(response)
