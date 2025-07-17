from agno.agent import Agent
from agno.models.google import Gemini
from agno.embedder.google import GeminiEmbedder
from agno.knowledge.pdf import PDFKnowledgeBase,PDFReader
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.storage.sqlite import SqliteStorage
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.memory.v2.memory import Memory
from agno.tools.reasoning import ReasoningTools
from dotenv import load_dotenv
from agno.tools.duckduckgo import DuckDuckGoTools

load_dotenv()

knowledge = PDFKnowledgeBase(
    path = "data/llm.pdf",
    vector_db=LanceDb(
        uri="tmp/lancedb",
        table_name="test_docs",
        search_type=SearchType.hybrid,
        # Use Gemini for embeddings
        embedder=GeminiEmbedder(id="models/text-embedding-004"),
    ),
    reader=PDFReader(chunk=True),
)
memory = Memory(
    # Use any model for creating and managing memories
    model=Gemini(id="gemini-2.0-flash"),
    # Store memories in a SQLite database
    db=SqliteMemoryDb(table_name="user_memories", db_file="tmp/agent.db"),
    # We disable deletion by default, enable it if needed
    delete_memories=True,
    clear_memories=True,
)

# Store agent sessions in a SQLite database
storage = SqliteStorage(table_name="agent_sessions", db_file="tmp/agent.db")


def create_agent():
    agent = Agent(
            name="Agno Assist",
            model=Gemini(id="gemini-2.0-flash", temperature=0.0),
            tools=[
                ReasoningTools(add_instructions=True),
                DuckDuckGoTools()
            ],
            # User ID for storing memories, `default` if not provided
            user_id="acchu",
            instructions=[
                "Search your knowledge before answering the question.If now found in your knowledge use the tools and answer",
                "Only include the output in your response. No other text.",
            ],
            knowledge=knowledge,
            memory=memory,
            # Let the Agent manage its memories
            enable_agentic_memory=True,
            storage=storage,
            add_datetime_to_instructions=True,
            # Add the chat history to the messages
            add_history_to_messages=True,
            # Number of history runs
            num_history_runs=3,
            markdown=True,
        )
    return agent


def main():
    agent = create_agent()
    #recreate can be set true if we want to recreate
    agent.knowledge.load(recreate=False)
    while True:
        query = input("Ask question...\n")

        if "exit" in query.lower() or "end" in query.lower():
            print("Exiting...")
            break

       # resp = agent.run(query)
       # print(resp.content)
        agent.print_response(query,stream=True)
        

if __name__ == "__main__":
    main()
    

