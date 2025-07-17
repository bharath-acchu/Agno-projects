from agno.agent import Agent 
from agno.models.google import Gemini
from agno.tools.yfinance import YFinanceTools
from dotenv import load_dotenv

#load .env files
load_dotenv()


#define the llm
def define_llm():
        return Gemini(id="gemini-2.0-flash", temperature=0.0)

def main():
        llm = define_llm()
        while True:
                
                query = input("Enter the query:\n")
                if "exit" in query.lower():
                        print("Bye....")
                        break
        
                

                stockAgent = Agent(
                    model=llm,
                    tools=[YFinanceTools(stock_price=True)],
                    instructions="You are knowledgable financial advisor, make sure you provide the correct information using available tools to you.If you are not sure about the answer , respond back as you are not aware of it.",
                    markdown=False,
                    show_tool_calls=False
                    )

                #stockAgent.print_response("What is the stock price of AAPL?", stream=False)
                resp = stockAgent.run(query)
                print("Resposne we got:", resp.content)

if __name__ == "__main__":
        main()