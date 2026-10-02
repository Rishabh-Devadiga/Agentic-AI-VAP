from mcp.server.fastmcp import FastMCP
from ddgs import DDGS

mcp = FastMCP("LocalWebSearch")


@mcp.tool()
def search_web(query: str) -> str:
    """Search the web for up-to-date information and facts."""

    try:
        with DDGS() as ddgs:
            results = list(
                ddgs.text(
                    query,
                    max_results=2
                )
            )

        if not results:
            return "No results found."

        return "\n".join(
            f"- {r['title']}: {r['body']}"
            for r in results
        )

    except Exception as e:
        return f"Search error: {e}"


if __name__ == "__main__":
    mcp.run(transport="stdio")