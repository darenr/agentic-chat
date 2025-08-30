from pydantic_ai import Agent, UrlContextTool
from rich import print
from rich.markdown import Markdown
from rich.console import Console
import sys
import os
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.models.google import GoogleModel


model = GoogleModel(
    "gemini-2.0-flash", provider=GoogleProvider(api_key=os.getenv("GEMINI_API_KEY"))
)


agent = Agent(
    model,
    builtin_tools=[UrlContextTool()],
)


def run_agent(q: str):
    result = agent.run_sync(q)
    return result.output


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} '<your question here>'")
        sys.exit(1)

    console = Console()

    answer = run_agent(sys.argv[1])
    console.print(Markdown(answer))
