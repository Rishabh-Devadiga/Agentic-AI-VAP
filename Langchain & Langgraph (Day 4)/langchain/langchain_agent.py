from langchain_groq import ChatGroq


llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0
)

response = llm.invoke(
    "Explain docker in simple terms"
)

print(response.content)