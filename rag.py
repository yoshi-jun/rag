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
# from langchain_ollama import ChatOllama   # not supported conda
from langchain_community.chat_models import ChatOllama # installable by conda

from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import JSONLoader
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate

import json
from pathlib import Path
from pprint import pprint

#==============================================================================
def GetLlmModel(mode):
    # Get llm models 
    if mode.lower() == "gpt-4o-mini":
        llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)

    elif mode.lower() == "gemini-1.5-pro" or mode is None:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", 
                                     temperature=0.7, streaming=True)
    
    elif mode.lower() == "llama3":
        llm = ChatOllama(model = "llama4",streaming=True)
    
    else:
        llm = None
    
    return llm

#------------------------------------------------------------------------------
def LoadJSON(f_name):
    # transform context to vector

    data = json.loads(f_name)

    return data


###############################################################################
def main():
    doc = LoadJSON("subtitle.json")
    pprint(doc)

if __name__=="__main__":
    main()