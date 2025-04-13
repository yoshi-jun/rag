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

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# not supported conda
from langchain_ollama import ChatOllama, OllamaEmbeddings

# installable by conda
# from langchain_community.chat_models import ChatOllama 
# from langchain_community.embeddings import OllamaEmbeddings

from langchain.schema import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate

import json
from pathlib import Path
from pprint import pprint

#==============================================================================
def GetLlmModel(mode):
    llm = ChatOllama(model = "llama3", streaming=True)
    return llm

#------------------------------------------------------------------------------
def LoadJSON(f_name):
    with open(f_name, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data

#------------------------------------------------------------------------------
def Embedding(doc):
    embeded = OllamaEmbeddings(model="llama3.1")

    vec = embeded.embed_query(doc)

    return vec
###############################################################################
def main():
    # load to json file
    doc = LoadJSON("subject/subtitle.json")
    
    # transfrom text to vec
    print(doc.values())
    

    # print(vec)

if __name__=="__main__":
    main()