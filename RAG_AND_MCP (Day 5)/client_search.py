import asyncio
import ollama

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def run_local_mcp():

    # 1. Define server startup options
    server_params = StdioServerParameters(
        command="python",
        args=["search_server.py"]
    )

    # 2. Connect to MCP Server over stdio
    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            await session.initialize()

            # 3. Discover available tools
            mcp_tools = await session.list_tools()

            # Convert MCP tools to Ollama format
            ollama_tools = []

            for tool in mcp_tools.tools:

                ollama_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                })

            user_query = "What is the latest score or news about Real Madrid?"

            print(f"User Query: {user_query}\n")

            # 4. Ask local LLM
            response = ollama.chat(
                model="qwen2.5:7b",
                messages=[
                    {
                        "role": "user",
                        "content": user_query
                    }
                ],
                tools=ollama_tools
            )

            # 5. Check whether LLM requested a tool
            message = response["message"]

            if message.get("tool_calls"):

                for tool_call in message["tool_calls"]:

                    fn_name = tool_call["function"]["name"]
                    fn_args = tool_call["function"]["arguments"]

                    print(
                        f"--> [Local LLM called MCP Tool]: "
                        f"{fn_name}({fn_args})"
                    )

                    # Execute MCP tool
                    result = await session.call_tool(
                        fn_name,
                        fn_args
                    )

                    tool_output = result.content[0].text

                    print(
                        f"--> [MCP Server Output]:\n"
                        f"{tool_output}\n"
                    )

                    # 6. Give search result back to LLM
                    final_response = ollama.chat(
                        model="llama3.2",
                        messages=[
                            {
                                "role": "user",
                                "content": user_query
                            },
                            message,
                            {
                                "role": "tool",
                                "content": tool_output
                            }
                        ]
                    )

                    print(
                        f"Final Answer:\n"
                        f"{final_response['message']['content']}"
                    )

            else:
                print(
                    f"Direct Response:\n"
                    f"{message['content']}"
                )


if __name__ == "__main__":
    asyncio.run(run_local_mcp())