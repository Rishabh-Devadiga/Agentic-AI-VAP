import os
import json
import subprocess
import psutil
import warnings

# Suppress the Python 3.14 Pydantic V1 warning to keep the console clean
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core.utils.pydantic")

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# The new graph-based runtime import (LangChain v1.0+)
from langchain.agents import create_agent

# ==========================================
# 0. GROQ CONFIGURATION
# ==========================================

MODEL_NAME = "qwen/qwen3.8-27b"

llm = ChatGroq(
    model=MODEL_NAME,
    temperature=0,
    api_key=os.environ.get("GROQ_API_KEY")
)


# ==========================================
# 1. DEFINE AGENT TOOLS
# ==========================================

@tool
def get_system_metrics() -> str:
    """Get current CPU, RAM, and Disk usage of the local computer."""
    metrics = {
        "cpu_usage_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage("C:\\").percent
    }
    return json.dumps(metrics)

@tool
def ping_host(hostname: str) -> str:
    """Ping a hostname or IP address to check network connectivity."""
    try:
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
# 2. AGENT FACTORY
# ==========================================

# We now initialize the agents purely with the LLM and their tools. 
# We'll handle the system prompts during execution to avoid argument errors.
system_agent = create_agent(model=llm, tools=[get_system_metrics])
network_agent = create_agent(model=llm, tools=[ping_host])


# ==========================================
# 3. ROUTER SETUP
# ==========================================

router_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a query router. Analyze the user's query and respond with ONLY ONE WORD.

Choose SYSTEM if the user asks about:
- CPU, RAM, memory usage, disk usage, system hardware, computer resources, system performance

Choose NETWORK if the user asks about:
- ping, network connectivity, internet connectivity, latency, connection, whether a host is reachable

Choose UNKNOWN for anything else."""),
    ("human", "{query}")
])

router_chain = router_prompt | llm | StrOutputParser()


# ==========================================
# 4. ORCHESTRATOR
# ==========================================

def orchestrate_query(user_query: str):
    print("\n==========================================")
    print(f'USER QUERY: "{user_query}"')
    print("==========================================")

    # 1. Router makes a decision
    decision = router_chain.invoke({"query": user_query}).strip().upper()
    print(f"\n[Router Decision]: {decision}\n")

    # 2. Route to the correct agent
    if "SYSTEM" in decision:
        print("--- [System Health Agent] Activated ---\n")
        
        system_prompt = (
            "You are a system administration agent. Use the available system "
            "metrics tool to retrieve CPU, RAM, and Disk usage."
        )
        
        # Pass the system prompt directly into the message array
        result = system_agent.invoke({
            "messages": [
                ("system", system_prompt),
                ("user", user_query)
            ]
        })
        
        print(result["messages"][-1].content)
        
    elif "NETWORK" in decision:
        print("--- [Network Agent] Activated ---\n")
        
        network_prompt = (
            "You are a network diagnostic agent. Use the ping tool to check "
            "whether the requested host is reachable."
        )
        
        result = network_agent.invoke({
            "messages": [
                ("system", network_prompt),
                ("user", user_query)
            ]
        })
        
        print(result["messages"][-1].content)
        
    else:
        print("[Router]: Request does not match the available agent domains.")


# ==========================================
# 5. MAIN PROGRAM
# ==========================================

if __name__ == "__main__":
    print("==========================================")
    print("        LANGCHAIN MULTI-AGENT SYSTEM")
    print("==========================================")
    print("Model:", MODEL_NAME)
    print("\nAvailable agents:")
    print("  - System Health Agent")
    print("  - Network Agent")
    print("\nType 'exit' to quit.")

    while True:
        query = input("\nEnter your query: ").strip()
        
        if query.lower() == "exit":
            print("Exiting...")
            break
            
        if not query:
            continue
            
        orchestrate_query(query)