import os
import requests
import urllib.parse

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from datetime import datetime
from zoneinfo import ZoneInfo
from twilio.rest import Client

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
        self.classification_llm = self.class_prompt_structure()
        self.rag = self.rag_prompt_structure()
        self.chain = self.retrieval_chain()
        
        self.phone_book = {
            "Abel": "+919605730268",
            "Savio": "+919526664904",
            "Sarat": "+919633538760",
            "Jovan": "+919961109777"
        }
        
        
    def class_prompt_structure(self) :
        # classification_template = PromptTemplate.from_template(
        #     """You're job is to classify queries. Given a user question below, classify it
        #     as belonging to either "Get_Time", "Get_Stock_Price" or "Miscellaneous". 

        #     <If the user query is about the current time in any country, then classify the question as "Get_Time".
        #     The response should be of the format "Get_Time,IANA" where IANA is a valid time zone from
        #     the IANA Time Zone Database like "America/New_York" or "Europe/London". If the country isn't mentioned
        #     assume it to be India>
        #     <If the user query is about the stock price of a share of any company in the market, then classify the question
        #     as "Get_Stock_Price". The Respose should be of the format "Get_Stock_Price,Symbol" where Symbol
        #     is the Ticker symbol of the concerned company.>
        #     <If the user query is about any other subject or topic, classify the question as "Miscellaneous">

        #     <question>
        #     {question}
        #     </question>

        #     Classification:
        #     """
        #     )
        classification_template = PromptTemplate.from_template(
            """You're job is to classify queries. Given a user question below, classify it
            as belonging to either "Get_Time", "Get_Stock_Price", "Send_Whatsapp_Message" or "Miscellaneous". 

            <If the user query is about the current time in any country, then classify the question as "Get_Time".
            The response should be of the format "Get_Time,IANA" where IANA is a valid time zone from
            the IANA Time Zone Database like "America/New_York" or "Europe/London". If the country isn't mentioned
            assume it to be India>
            
            <If the user query is about the stock price of a share of any company in the market, then classify the question
            as "Get_Stock_Price". The Respose should be of the format "Get_Stock_Price,Symbol" where Symbol
            is the Ticker symbol of the concerned company.>
            
            <If the user query is about sending a whatsapp message to a particular user, then classify the question as "Send_Whatsapp_Message". 
            The response should be in the format "Send_Whatsapp_Message,Name,Message" where Name is the name of the person mentioned and
            Message is the message to be sent that is taken from the user's query word for word.>
            
            <If the user query is about any other subject or topic, classify the question as "Miscellaneous">

            <question>
            {question}
            </question>

            Classification:
            """
            )

        classification_chain = (
            classification_template
            | self.model
            | StrOutputParser()
        )
        
        return classification_chain


    def get_time(self, timezone) :
        time_now = datetime.now(ZoneInfo(timezone))
        return time_now.strftime("%H:%M")
    
    
    def get_stock_price(self, ticker_symbl) :
        sym = urllib.parse.quote_plus(ticker_symbl)
        try:
            api_key = os.environ.get("STOCK_API_KEY")
            url_real_time = f"https://fmpcloud.io/api/v3/quote/{sym}?apikey={api_key}"
            price = requests.get(url_real_time)
            price.raise_for_status()
        except requests.RequestException:
            return None

        # Parse response
        try:
            price = price.json()
            return {
                "price": price[0]['price'],
                "name": price[0]['name']
            }
        except (KeyError, TypeError, ValueError) :
                return None
            
    
    def send_whatsapp_mssg(self, mssg, to_number) :
        client = Client(os.environ.get("TwilioAccountSid"), os.environ.get("TwilioAuthToken"))

        from_whatsapp_number = "whatsapp:+14155238886"
        to_whatsapp_number = f"whatsapp:{to_number}"

        mssg = client.messages.create(
            body=mssg,
            from_=from_whatsapp_number,
            to=to_whatsapp_number
        )


    def rag_prompt_structure(self) :
        system = f"""You are a helpful assistant. Explain in simple consise statements.
            Theres no need to mention the context if the query is unrelated.
            If the query is unrelated to the provided context, you many go out of context to
            respond to the query. If you don't know the answer, just say that you don't know.
            Generate a list of important areas or considerations for this subject
            and for each area or consideration listed, generate a brief description of it"""
        human = "{text}"
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
        
        llm = prompt | self.model
        
        return llm
    
    
    def route(self, query) :
        print("Here in route")
        route_class = self.classification_llm.invoke({"question": query}).split(",")
        if route_class[0] == "Get_Time" :
            print("here in get time")
            return "The time is: " + self.get_time(route_class[1].strip())

        elif route_class[0] == "Get_Stock_Price" :
            response = self.get_stock_price(route_class[1].strip())
            return f"The price of {response['name']}'s stock is currently USD {response['price']}"
        
        elif route_class[0] == "Send_Whatsapp_Message" :
            print("Here")
            name = route_class[1].strip().capitalize()
            number = self.phone_book[name]
            print(f"number: {number}")
            mssg = route_class[2].strip()
            self.send_whatsapp_mssg(mssg, number)
            
            return f"Message {mssg} sent successfully to {name}"

        else :
            return self.invoke_rag(query)

    
    
    def invoke_from_base_llm(self, prompt) :
        return self.rag.invoke({"text": prompt}).content
    
    
    def retrieval_chain(self) :
        chain = ConversationalRetrievalChain.from_llm(self.rag, db_manager.db.as_retriever(), return_source_documents=True)
        
        return chain
    
    
    def invoke_rag(self, prompt) :
        response = self.chain({"question": prompt, "chat_history": self.chat_history})["answer"]
        self.chat_history.append((prompt, response))
        
        return response
    
# db_manager = DbManager()