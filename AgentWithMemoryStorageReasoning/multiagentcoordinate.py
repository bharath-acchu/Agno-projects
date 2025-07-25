from agno.agent import Agent
from agno.models.google import Gemini
from dotenv import load_dotenv
from agno.team.team import Team
load_dotenv()
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.newspaper4k import Newspaper4kTools

#agent1: aearch agent
searcher = Agent(
    name="Searcher",
    role="Searches the top URLs for a topic",
    model=Gemini(id="gemini-2.0-flash"), ## if not specified by default it will call OprnAI model
    instructions=[
        "Given a topic, first generate a list of 3 search terms related to that topic.",
        "For each search term, search the web and analyze the results.Return the 10 most relevant URLs to the topic.",
        "You are writing for the New York Times, so the quality of the sources is important.",
    ],
    tools=[DuckDuckGoTools(fixed_max_results=3)],
    add_datetime_to_instructions=True,
)

#agent2: Comapny info searcher w.r.t stocks
writer = Agent(
    name="Writer",
    role="Writes a high-quality article",
    model=Gemini(id="gemini-2.0-flash"), ## if not specified by default it will call OprnAI model
    description=(
        "You are a senior writer for the New York Times. Given a topic and a list of URLs, "
        "your goal is to write a high-quality NYT-worthy article on the topic."
    ),
    instructions=[
        "First read all urls using `read_article`."
        "Then write a high-quality NYT-worthy article on the topic."
        "The article should be well-structured, informative, engaging and catchy.",
        "Ensure the length is at least as long as a NYT cover story -- at a minimum, 5 paragraphs.",
        "Ensure you provide a nuanced and balanced opinion, quoting facts where possible.",
        "Focus on clarity, coherence, and overall quality.",
        "Never make up facts or plagiarize. Always provide proper attribution.",
        "Remember: you are writing for the New York Times, so the quality of the article is important.",
    ],
    tools=[Newspaper4kTools()],
    add_datetime_to_instructions=True,
)

def defineTeam():
    editor = Team(
    name="Editor",
    mode="coordinate",
    model=Gemini(id="gemini-2.0-flash"),
    members=[searcher, writer],
    description="You are a senior NYT editor. Given a topic, your goal is to write a NYT worthy article.",
    instructions=[
        "First ask the search journalist to search for the most relevant URLs for that topic.",
        "Then ask the writer to get an engaging draft of the article.",
        "Edit, proofread, and refine the article to ensure it meets the high standards of the New York Times.",
        "The article should be extremely articulate and well written. "
        "Focus on clarity, coherence, and overall quality.",
        "Remember: you are the final gatekeeper before the article is published, so make sure the article is perfect.",
    ],
    add_datetime_to_instructions=True,
    add_member_tools_to_system_message=False,  # This can be tried to make the agent more consistently get the transfer tool call correct
    enable_agentic_context=True,  # Allow the agent to maintain a shared context and send that to members.
    share_member_interactions=True,  # Share all member responses with subsequent member requests.
    show_members_responses=True,
    markdown=True,
    )
    return editor

def main():
    print("Team Lead is on strike !!")
    ContentEditor = defineTeam()
    while True:
        query = input("Enter the topic for your content to the news\n")
        if "exit" in query.lower() or "end" in query.lower() or "quit" in query.lower():
            print("Thanks, Bye..Exitting...")
            break
        try:
            resp = ContentEditor.run(query)
            print(resp.content)
            #ContentEditor.print_response(query) ## to see which agent/tool call
        except ModuleNotFoundError as e:
            if "503" in e:
                print("Model is busy BRUH!! try after sometime")
            else:
                raise
if __name__ == "__main__":
    main()

    



