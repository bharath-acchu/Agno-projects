from agno.agent import Agent
from agno.models.google import Gemini
from dotenv import load_dotenv
from agno.team.team import Team
load_dotenv()
from agno.tools.yfinance import YFinanceTools
from pydantic import BaseModel
import json


#define the structure of the output using pydantic
class StockAnalaysis(BaseModel): 
    symbol:str
    stockPrice:float
    company_name:str
    analysis: str

class CompanyAnalysis(BaseModel):
    companay_name:str
    analysis:str


#agent1: Stock searcher
stock_searcher = Agent(
    name="Stock Searcher",
    model = Gemini(id="gemini-2.0-flash"),
    response_model=StockAnalaysis, # for the structured output
    role="Searches for information on stocks and provides price analysis.",
    tools=[
        YFinanceTools(
            stock_price =True,
            analyst_recommendations=True
        )
    ]
    )

#agent2: Comapny info searcher w.r.t stocks
company_info_agent = Agent(
    name="Company Info Searcher",
    model=Gemini(id="gemini-2.0-flash"),
    role="Searches for information about companies and recent news .",
    response_model=CompanyAnalysis,
    tools=[
        YFinanceTools(
            stock_price=False,
            company_info=True,
            company_news=True,
        )
    ],
)

def defineTeam():
    team = Team(
        name="Stock Research Team",
        mode="route", # mode is route : it will route to the dedicated agent for the work
        model=Gemini(id="gemini-2.0-flash"),
        members=[stock_searcher,company_info_agent],
        markdown=True
    )
    return team

def main():
    print("Team Lead is on strike !!")
    TeamLeader = defineTeam()
    while True:
        query = input("Enter the question for your stock reseracher\n")
        if "exit" in query.lower() or "end" in query.lower() or "quit" in query.lower():
            print("Thanks, Bye..Exitting...")
            break
        try:
            resp = TeamLeader.run(query)
            print(resp.content.model_dump_json(indent=2)) ## if you want in json format
            #TeamLeader.print_response(query) ## to see which agent/tool call
        except ModuleNotFoundError as e:
            if "503" in e:
                print("Model is busy BRUH!! try after sometime")
            else:
                raise
if __name__ == "__main__":
    main()

    



