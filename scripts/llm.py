from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from scripts.db_manager import db_manager


class Llm :
    MODEL_NAME = "llama3-70b-8192"

    def __init__(self) :
        self.model = ChatGroq(
            model_name= self.MODEL_NAME,
            temperature=1,
            max_tokens=1024,
            )
        
        self.chat_history = list()
        self.llm = self.prompt_structure()
        self.chain = self.retrieval_chain()


    def prompt_structure(self) :
        system = f"""You are a helpful assistant. Explain in simple consise statements.
            generate a list of important areas or considerations for this subject
            and for each area or consideration listed, generate a brief description of it"""
        human = "{text}"
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
        
        llm = prompt | self.model
        
        return llm
    
    
    def invoke_from_base_llm(self, prompt) :
        return self.llm.invoke({"text": prompt}).content
    
    
    def retrieval_chain(self) :
        chain = ConversationalRetrievalChain.from_llm(self.llm, db_manager.db.as_retriever(), return_source_documents=True)
        
        return chain
    
    
    def invoke_rag(self, prompt) :
        response = self.chain({"question": prompt, "chat_history": self.chat_history})["answer"]
        self.chat_history.append((prompt, response))
        
        return response
    
# db_manager = DbManager()