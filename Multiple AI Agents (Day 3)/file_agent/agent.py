from pathlib import Path
import ollama

MODEL = "gpt-oss:20b"

def echo_fun(prompt: str) -> str:
    return prompt

def function1(prompt: str) -> str: 
    return prompt.upper()

def function2(): {}
def function3(): {}



TOOLS = [echo_fun, function1, function2, function3]

SYSTEM_PROMPT = """
       You are a model that likes to eat dosa and vadapav as well.
       Your job is to read tools and print the messages. 
"""

def run_agent():
    messages = [{"role" : "system", "content": SYSTEM_PROMPT}]
    print("Local  GPT-OSS echo Agent")
    print("Type 'exit' to quit")
    while True:
        prompt = input("\nmessage me for echoing ...: ").strip()
        if prompt.lower() == "exit":
            break
        if not prompt:
            continue

        messages.append({"role": "user", "content": prompt})

        while True:
            response = ollama.chat(
                model = MODEL,
                messages = messages,
                tools = TOOLS
            )

            messages.append(response.message)

            if not response.message.tool_calls:
                print("\nAgent:", response.message.content)
                break

            for call in response.message.tool_calls:
                name = call.function.name
                args = call.function.arguments

                tool = next(
                    (t for t in TOOLS if t.__name__ == name),
                    None
                )

                if tool is None:
                    result = "Error: Unknown tool."
                else:
                    try:
                        result = tool(**args)
                    except Exception as exc:
                        result = f"Error: {exc}"

                print(f"\n[Tool: {name}]")
                print(result)

                messages.append({
                    "role": "tool",
                    "tool_name": name,
                    "content": str(result),
                })
if __name__ == "__main__":
    run_agent()


