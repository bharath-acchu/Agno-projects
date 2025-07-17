from agno.agent import Agent
from agno.models.google import Gemini
from agno.team.team import Team
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.reasoning import ReasoningTools
from agno.tools.yfinance import YFinanceTools
from dotenv import load_dotenv


load_dotenv()

web_agent = Agent(
    name="Web Search Agent",
    role="Handle web search requests and general research",
    model=Gemini(id="gemini-2.0-flash"),
    tools=[DuckDuckGoTools()],
    instructions="Always include sources",
    add_datetime_to_instructions=True,
)

finance_agent = Agent(
    name="Finance Agent",
    role="Handle financial data requests and market analysis",
    model=Gemini(id="gemini-2.0-flash"),
    tools=[YFinanceTools(stock_price=True, stock_fundamentals=True,analyst_recommendations=True, company_info=True)],
    instructions=[
        "Use tables to display stock prices, fundamentals (P/E, Market Cap), and recommendations.",
        "Clearly state the company name and ticker symbol.",
        "Focus on delivering actionable financial insights.",
    ],
    add_datetime_to_instructions=True,
)

reasoning_finance_team = Team(
    name="Reasoning Finance Team",
    mode="coordinate",
    model=Gemini(id="gemini-2.0-flash"),
    members=[web_agent, finance_agent],
    tools=[ReasoningTools(add_instructions=True)],
    instructions=[
        "Collaborate to provide comprehensive financial and investment insights",
        "Consider both fundamental analysis and market sentiment",
        "Use tables and charts to display data clearly and professionally",
        "Present findings in a structured, easy-to-follow format",
        "Only output the final consolidated analysis, not individual agent responses",
    ],
    markdown=True,
    show_members_responses=True,
    enable_agentic_context=True,
    add_datetime_to_instructions=True,
    success_criteria="The team has provided a complete financial analysis with data, visualizations, risk assessment, and actionable investment recommendations supported by quantitative analysis and market research.",
)

if __name__ == "__main__":
    try:

        reasoning_finance_team.print_response("""Compare the tech sector giants (AAPL, GOOGL, MSFT) performance:
            1. Get financial data for all three companies
            2. Analyze recent news affecting the tech sector
            3. Calculate comparative metrics and correlations
            4. Recommend portfolio allocation weights""",
            stream=False,
            show_full_reasoning=True,
            stream_intermediate_steps=True,
        )
    except ModuleNotFoundError as e:
        if "503" in e:
            print("Gemini model is currently overloaded.Please try again later")
        else:
            raise
