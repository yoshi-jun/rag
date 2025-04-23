# =============================================================================
# BSD 2-Clause License

# Copyright (c) 2025, yoshi-jun
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# =============================================================================

##### For each LLMs lib ####
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
# not supported conda
from langchain_ollama import ChatOllama
# installable by conda
# from langchain_community.chat_models import ChatOllama 
# from langchain_community.embeddings import OllamaEmbeddings

#### for loading PDF Table ####
from llama_index.readers.pdf_table import PDFTableReader
#### For load json files ####
import json
from pathlib import Path
from pprint import pprint

#### for text loader ####
from langchain_community.document_loaders import TextLoader

### fro embedding text and store it ####
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

#### For Make Prompt ####
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate

#==============================================================================
def GetLlmModel(model="llama3.1"):
    llm = ChatOllama(model=model, streaming=True)
    return llm

#------------------------------------------------------------------------------
def LoadPDFTable(f_name):
    reader = PDFTableReader()
    documents = reader.load_data(file=f_name)

    return documents

#------------------------------------------------------------------------------
def Embedding(docs, model="llama3.1"):

    embededings = OllamaEmbeddings(model=model)

    for doc in docs:
        vecs = InMemoryVectorStore.from_texts(
            [doc.text],
            embedding=embededings
        )

    return vecs

#------------------------------------------------------------------------------
def MakeQuestion(mes, fname):
    hmes = HumanMessage(content=mes)

    temp = PromptTemplate.from_template(
        "次に表示されるテキストから問題を出題してください。"
        "問題を出題する形式は次のとおりです。"
        "1. ①や②, ③, ④, ⑤の後の単語を問う問題であること"
        "一度に出題する量は【】の部分から次の【】までの間を問題として出題すること"
        "テキストは次のとおりです。"
        "{question}")

    # load QA
    loder = TextLoader(fname)
    question = loder.load()
    print(question)
    prompt = temp.invoke({"question": question})
    print(prompt)
    smes = SystemMessage(content=prompt)

    return smes, hmes

###############################################################################
def main():
    # load to QA
    f_name = ".QA"

    # Define the llm
    llm = GetLlmModel()

    while True:
        print("問題を出題します。")

        mes = "問題を出題してください"

        smes, hmes = MakeQuestion(mes,f_name)

        for chunk in llm.stream([smes, hmes]):
            if chunk.content:
                print(chunk.content, end="", flush=True)        

###############################################################################
if __name__=="__main__":
    main()