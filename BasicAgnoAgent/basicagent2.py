from agno.agent import Agent
from agno.models.google import Gemini
from dotenv import load_dotenv
from agno.tools.duckduckgo import DuckDuckGoTools

load_dotenv()

def define_llm():
    return Gemini(id="gemini-2.0-flash", temperature=0.5)

def main():
    llm = define_llm()
    Researchagent = Agent(
            model=llm,
            tools=[DuckDuckGoTools(search=True,fixed_max_results=3)],
            instructions="You are a deligent research assisstant who is very skilled at gathering, collecting "
            "and summarizing the information",
            markdown=True
        )

    while True:
        query  = input("Enter the search querry\n")

        if "exit" in query.lower() or "end" in query.lower():
            print("Exiting ....")
            break

        resp = Researchagent.run(query)
        print(resp.content)

if __name__ == "__main__":
    main()

