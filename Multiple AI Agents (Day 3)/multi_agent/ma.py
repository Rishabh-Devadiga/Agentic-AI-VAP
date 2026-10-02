import json
import subprocess
import psutil
import os

from groq import Groq


# ==========================================
# 0. GROQ CONFIGURATION
# ==========================================

MODEL_NAME = "openai/gpt-oss-20b"

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)


# ==========================================
# 1. DEFINE AGENT TOOLS
# ==========================================

def get_system_metrics() -> str:
    """Returns local host system resource utilization."""

    metrics = {
        "cpu_usage_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage("C:\\").percent
    }

    return json.dumps(metrics)


def ping_host(hostname: str) -> str:
    """
    Pings a network host to evaluate connectivity.
    """

    try:

        # Windows uses -n instead of -c
        res = subprocess.check_output(
            ["ping", "-n", "2", hostname],
            stderr=subprocess.STDOUT,
            text=True
        )

        return json.dumps({
            "status": "success",
            "raw_output": res.strip()
        })

    except Exception as err:

        return json.dumps({
            "status": "error",
            "message": str(err)
        })


# ==========================================
# 2. TOOL MAPS
# ==========================================

SYSTEM_TOOLS_MAP = {
    "get_system_metrics": get_system_metrics
}

NETWORK_TOOLS_MAP = {
    "ping_host": ping_host
}


# ==========================================
# 3. TOOL SCHEMAS
# ==========================================

SYSTEM_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_system_metrics",
            "description": (
                "Get current CPU, RAM, and Disk "
                "usage of the local computer."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]


NETWORK_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "ping_host",
            "description": (
                "Ping a hostname or IP address "
                "to check network connectivity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "hostname": {
                        "type": "string",
                        "description": (
                            "Hostname or IP address "
                            "to ping."
                        )
                    }
                },
                "required": ["hostname"]
            }
        }
    }
]


# ==========================================
# 4. GENERIC AGENT RUNNER
# ==========================================

def run_agent(
    agent_name: str,
    system_prompt: str,
    user_query: str,
    tools_schema: list,
    tools_map: dict
):

    print(f"\n--- [{agent_name}] Activated ---")

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

    # ======================================
    # MODEL → TOOL DECISION
    # ======================================

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=tools_schema,
        tool_choice="auto"
    )

    message = response.choices[0].message

    # Add assistant response to conversation
    messages.append(message)

    # ======================================
    # CHECK FOR TOOL CALLS
    # ======================================

    if not message.tool_calls:

        print(f"\n[{agent_name} Response]:")
        print(message.content)

        return

    # ======================================
    # EXECUTE ALL TOOL CALLS
    # ======================================

    for tool_call in message.tool_calls:

        function_name = tool_call.function.name

        # Groq returns arguments as JSON string
        function_args = json.loads(
            tool_call.function.arguments
        )

        print(
            f"\n  └─ Executing Tool: "
            f"`{function_name}`"
        )

        print(
            f"     Arguments: {function_args}"
        )

        # Find Python function
        function_to_call = tools_map.get(function_name)

        if function_to_call is None:

            result = json.dumps({
                "error": f"Unknown tool: {function_name}"
            })

        else:

            try:

                result = function_to_call(
                    **function_args
                )

            except Exception as err:

                result = json.dumps({
                    "error": str(err)
                })

        print(
            f"     Result: {result}"
        )

        # ==================================
        # SEND TOOL RESULT BACK TO GROQ
        # ==================================

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": function_name,
            "content": str(result)
        })

    # ======================================
    # FINAL MODEL RESPONSE
    # ======================================

    final_response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tool_choice="none"
    )

    print(f"\n[{agent_name} Final Answer]:")

    print(
        final_response.choices[0].message.content
    )


# ==========================================
# 5. ORCHESTRATOR / ROUTER
# ==========================================

def orchestrate_query(user_query: str):

    print("\n==========================================")
    print(f'USER QUERY: "{user_query}"')
    print("==========================================")

    # ======================================
    # ROUTER PROMPT
    # ======================================

    router_prompt = f"""
You are a query router.

Analyze the user's query and respond with
ONLY ONE WORD.

SYSTEM

Choose SYSTEM if the user asks about:

- CPU
- RAM
- memory usage
- disk usage
- system hardware
- computer resources
- system performance

NETWORK

Choose NETWORK if the user asks about:

- ping
- network connectivity
- internet connectivity
- latency
- connection
- whether a host is reachable

UNKNOWN

Choose UNKNOWN for anything else.

User query:

{user_query}
"""

    # ======================================
    # ROUTER CALL
    # ======================================

    route_response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": router_prompt
            }
        ],
        temperature=0
    )

    decision = (
        route_response
        .choices[0]
        .message
        .content
        .strip()
        .upper()
    )

    print(
        f"\n[Router Decision]: {decision}"
    )

    # ======================================
    # SYSTEM AGENT
    # ======================================

    if "SYSTEM" in decision:

        run_agent(
            agent_name="System Health Agent",

            system_prompt=(
                "You are a system administration "
                "agent. Use the available system "
                "metrics tool to retrieve CPU, "
                "RAM, and Disk usage."
            ),

            user_query=user_query,

            tools_schema=SYSTEM_TOOLS_SCHEMA,

            tools_map=SYSTEM_TOOLS_MAP
        )

    # ======================================
    # NETWORK AGENT
    # ======================================

    elif "NETWORK" in decision:

        run_agent(
            agent_name="Network Agent",

            system_prompt=(
                "You are a network diagnostic agent. "
                "Use the ping tool to check whether "
                "the requested host is reachable."
            ),

            user_query=user_query,

            tools_schema=NETWORK_TOOLS_SCHEMA,

            tools_map=NETWORK_TOOLS_MAP
        )

    # ======================================
    # UNKNOWN
    # ======================================

    else:

        print(
            "\n[Router]: Request does not match "
            "the available agent domains."
        )


# ==========================================
# 6. MAIN PROGRAM
# ==========================================

if __name__ == "__main__":

    print("==========================================")
    print("        GROQ MULTI-AGENT SYSTEM")
    print("==========================================")

    print("Model:", MODEL_NAME)

    print("\nAvailable agents:")

    print("  - System Health Agent")
    print("  - Network Agent")

    print("\nType 'exit' to quit.")

    while True:

        user_query = input(
            "\nEnter your query: "
        ).strip()

        if user_query.lower() == "exit":

            print("Exiting...")

            break

        if not user_query:

            continue

        orchestrate_query(user_query)