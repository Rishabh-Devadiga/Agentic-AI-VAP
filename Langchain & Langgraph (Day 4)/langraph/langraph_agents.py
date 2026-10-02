import os
import re
import base64
import requests
import smtplib
from email.message import EmailMessage
from typing import TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from markdown_pdf import MarkdownPdf, Section

# Load environment variables from .env file FIRST
load_dotenv()

# ==========================================
# 1. LLM CONFIGURATION
# ==========================================
# Note: Replace this model string if you used a different one from your check_models.py script
llm = ChatGroq(
    model="openai/gpt-oss-20b", 
    temperature=0.3,
    api_key=os.environ["GROQ_API_KEY"]
)


# ==========================================
# 2. STATE DEFINITION
# ==========================================
class AgentState(TypedDict):
    question: str
    research: str
    technical: str
    report: str
    pdf_path: str
    email_status: str


# ==========================================
# 3. AGENT NODES
# ==========================================

# Node 0: Coordinator
def coordinator(state: AgentState):
    print("\n[Coordinator] Starting workflow...")
    return {
        "research": "",
        "technical": "",
        "report": "",
        "pdf_path": "",
        "email_status": ""
    }


# Node 1: Research Agent
def research_agent(state: AgentState):
    print("[Research Agent] Gathering core concepts...")

    prompt = f"""
    Analyze the following question as a Research Agent.
    Identify:
    - Important concepts
    - Requirements
    - Benefits
    - Challenges

    Question: {state['question']}
    """

    response = llm.invoke([
        SystemMessage(content="You are an IT Research Agent."),
        HumanMessage(content=prompt)
    ])
    return {"research": response.content}


# Node 2: Technical Agent
def technical_agent(state: AgentState):
    print("[Technical Agent] Designing architecture...")

    prompt = f"""
    You are a Kubernetes Technical Architect.
    Based on the research, propose a technical solution.

    Include:
    - Architecture
    - Kubernetes components
    - Deployment approach
    - Security
    - Monitoring

    CRITICAL INSTRUCTION:
    You MUST include a visual architecture diagram using Mermaid.js syntax.
    Wrap the diagram exactly inside a ```mermaid``` block.

    User question: {state['question']}
    Research findings: {state['research']}
    """

    response = llm.invoke([
        SystemMessage(content="You are a Kubernetes expert."),
        HumanMessage(content=prompt)
    ])
    return {"technical": response.content}


# Node 3: Report Agent
def report_agent(state: AgentState):
    print("[Report Agent] Structuring final report...")

    prompt = f"""
    Prepare a structured technical report. Include:
    1. Executive summary
    2. Research findings
    3. Technical architecture
    4. Implementation steps
    5. Conclusion

    CRITICAL INSTRUCTION: 
    If the technical solution contains a ```mermaid``` diagram code block, 
    you MUST copy that exact block into your final report so it can be rendered.

    User question: {state['question']}
    Research: {state['research']}
    Technical solution: {state['technical']}
    
    Do not invent unsupported facts.
    """

    response = llm.invoke([
        SystemMessage(content="You are a Technical Report Agent."),
        HumanMessage(content=prompt)
    ])
    return {"report": response.content}


# Node 4: Diagram Renderer
def diagram_renderer(state: AgentState):
    print("[Diagram Renderer] Checking for diagrams...")
    report = state['report']
    
    # Extract all mermaid code blocks using Regex
    mermaid_blocks = re.findall(r'```mermaid\s+(.*?)\s+```', report, re.DOTALL)
    
    for i, mermaid_code in enumerate(mermaid_blocks):
        try:
            print(f"[Diagram Renderer] Rendering diagram {i+1}...")
            
            # Base64 encode the mermaid code for the mermaid.ink URL
            encoded = base64.b64encode(mermaid_code.encode('utf-8')).decode('utf-8')
            image_url = f"https://mermaid.ink/img/{encoded}"
            
            # Download the image locally
            img_filename = f"architecture_diagram_{i}.png"
            response = requests.get(image_url)
            
            if response.status_code == 200:
                with open(img_filename, 'wb') as f:
                    f.write(response.content)
                
                # Replace the text block with a Markdown image tag
                original_block = f"```mermaid\n{mermaid_code}\n```"
                img_markdown = f"\n![Architecture Diagram]({img_filename})\n"
                report = report.replace(original_block, img_markdown)
                print(f"[Diagram Renderer] Successfully embedded diagram {i+1}.")
            else:
                print(f"[Diagram Renderer] Failed to render diagram. Status code: {response.status_code}")
                
        except Exception as e:
            print(f"[Diagram Renderer] Error processing diagram: {e}")
            
    return {"report": report}


# Node 5: PDF Generator
def pdf_generator(state: AgentState):
    print("[PDF Generator] Converting report to PDF...")
    file_name = "Technical_Report.pdf"
    
    try:
        pdf = MarkdownPdf(toc_level=2)
        pdf.add_section(Section(state['report']))
        pdf.save(file_name)
        print(f"[PDF Generator] Successfully saved PDF as: {file_name}")
    except Exception as e:
        print(f"[PDF Generator] Error generating PDF: {e}")

    return {"pdf_path": file_name}


# Node 6: Email Agent
def email_agent(state: AgentState):
    print("[Email Agent] Preparing to send report to manager...")
    
    sender_email = os.environ.get("SENDER_EMAIL")
    sender_password = os.environ.get("SENDER_PASSWORD")
    manager_email = os.environ.get("MANAGER_EMAIL")
    
    # Failsafe if credentials aren't set
    if not all([sender_email, sender_password, manager_email]):
        print("[Email Agent] Missing email credentials in .env file. Skipping email.")
        return {"email_status": "Skipped - Missing credentials"}

    try:
        msg = EmailMessage()
        msg['Subject'] = "Automated AI Technical Report Generated"
        msg['From'] = sender_email
        msg['To'] = manager_email
        
        email_body = (
            f"Hello,\n\n"
            f"Please find attached the new technical report regarding:\n"
            f"'{state['question']}'\n\n"
            f"Best regards,\n"
            f"Your Automated AI Team"
        )
        msg.set_content(email_body)
        
        # Attach the PDF
        pdf_path = state['pdf_path']
        if os.path.exists(pdf_path):
            with open(pdf_path, 'rb') as f:
                pdf_data = f.read()
            
            msg.add_attachment(
                pdf_data, 
                maintype='application', 
                subtype='pdf', 
                filename=os.path.basename(pdf_path)
            )
        else:
            print("[Email Agent] Warning: PDF file not found to attach.")
            
        # Send the email via Gmail's SMTP server
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)
            
        print(f"[Email Agent] Successfully sent report to {manager_email}!")
        return {"email_status": "Sent Successfully"}
        
    except Exception as e:
        print(f"[Email Agent] Failed to send email: {e}")
        return {"email_status": f"Failed: {e}"}


# ==========================================
# 4. BUILD THE GRAPH
# ==========================================
graph = StateGraph(AgentState)

# Register nodes
graph.add_node("coordinator", coordinator)
graph.add_node("research", research_agent)
graph.add_node("technical", technical_agent)
graph.add_node("report", report_agent)
graph.add_node("diagram_render", diagram_renderer)
graph.add_node("pdf_generation", pdf_generator)
graph.add_node("email_distribution", email_agent)

# Define workflow edges (Linear pipeline)
graph.add_edge(START, "coordinator")
graph.add_edge("coordinator", "research")
graph.add_edge("research", "technical")
graph.add_edge("technical", "report")
graph.add_edge("report", "diagram_render")
graph.add_edge("diagram_render", "pdf_generation")
graph.add_edge("pdf_generation", "email_distribution")
graph.add_edge("email_distribution", END)

# Compile graph
app = graph.compile()


# ==========================================
# 5. EXECUTE THE WORKFLOW
# ==========================================
if __name__ == "__main__":
    print("==========================================")
    print("       AI TECHNICAL REPORT PIPELINE       ")
    print("==========================================")
    
    question = input("\nEnter your technical question/prompt: ")

    result = app.invoke({
        "question": question,
        "research": "",
        "technical": "",
        "report": "",
        "pdf_path": "",
        "email_status": ""
    })

    print("\n==========================================")
    print("                WORKFLOW COMPLETE         ")
    print("==========================================")
    print(f"📄 PDF Location: {os.path.abspath(result['pdf_path'])}")
    print(f"✉️  Email Status: {result['email_status']}")