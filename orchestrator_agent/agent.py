from google.adk.agents import LlmAgent, Agent, SequentialAgent
from google.adk.runners import Runner
from google.adk.models.google_llm import Gemini
from google.adk.artifacts import InMemoryArtifactService
from google.adk.sessions import InMemorySessionService
from orchestrator_agent.document_parser_agent import parse_medical_document
from orchestrator_agent.fair_price_research_agent import google_search_agent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.plugins.save_files_as_artifacts_plugin import SaveFilesAsArtifactsPlugin
from google.adk.tools import ToolContext, load_artifacts 
from google.adk.tools import agent_tool
from dotenv import load_dotenv
from orchestrator_agent.insurance_advocate_agent import insurance_advocate_agent
import google.genai.types as types
import os
import asyncio
import sys
import logging

# Suppress non-critical warnings
logging.basicConfig(level=logging.DEBUG)

# Fix for Windows asyncio event loop issue
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()

async def process_user_file(tool_context: ToolContext) -> dict:
    '''
    Checks for and saves uploaded images to the uploads folder. 
    Call this tool FIRST on every user message to detect if files were uploaded.

    Arguments:
        tool_context: The ADK context used to load the artifact.

    Returns:
        Dictionary containing:
        - "has_files": bool - Whether any files were found
        - "message": str - Status message
        - "folder_path": str - Where files were saved (if any)
        - "file_count": int - Number of files saved
    '''
    d = {'has_files': False, 'message': '', 'folder_path': '', 'file_count': 0}

    try:
        # Create uploads directory if it doesn't exist
        os.makedirs("orchestrator_agent/uploads", exist_ok=True)
        
        # Get list of artifacts - MUST USE AWAIT
        artifacts = await tool_context.list_artifacts()
        
        if not artifacts or len(artifacts) == 0:
            d['message'] = "No files were uploaded with this message."
            return d
        print(f"Found {len(artifacts)} uploaded file(s). Saving...")
        saved_files = []
        for artifact in artifacts:
            try:
                # Load artifact - MUST USE AWAIT
                artifact_content = await tool_context.load_artifact(filename=artifact)
                file_name = artifact_content.inline_data.display_name 
                data_bytes = artifact_content.inline_data.data
                
                # Save to disk
                file_path = os.path.join("orchestrator_agent/uploads", file_name)
                with open(file_path, 'wb') as f:
                    f.write(data_bytes)
                saved_files.append(file_name)
            except Exception as e:
                d['message'] = f"Error loading artifact {artifact}: {str(e)}"
                return d

        d['has_files'] = True
        d['file_count'] = len(saved_files)
        d['message'] = f"Successfully saved {len(saved_files)} file(s): {', '.join(saved_files)}"
        d['folder_path'] = "orchestrator_agent/uploads"
        tool_context
        return d

    except Exception as e:
        d['message'] = f"Error processing files: {str(e)}"
        return d



model = "gemini-2.5-flash-lite"
APP_NAME = "medical_advocate_orchestrator_agent"

root_agent = LlmAgent(
    name="medical_advocate_orchestrator_agent",
    model=Gemini(model=model), 
    description="Orchestrator Agent to help answer questions & co-ordinate with patients",
    instruction="""
You are a helpful medical information advocate agent. Your role is to assist patients in understanding their medical bills, insurance claims, and related documents.

CRITICAL RULE - ALWAYS START HERE:
For EVERY user message, first call process_user_file.


After process_user_file has finished its execution, check the response:

IF "has_files" is true:
  1. Call parse_medical_document to extract structured data from uploaded documents
  2. Call google_search_agent to research fair market prices for the procedures
  3. Analyze and compare billed amounts vs fair market prices

  Your Analysis report should be in the following format:
**Medical Bill Analysis:**
- **Total Billed Amount:** [Sum of all billed amounts]
- **Fair Market Value Total:** [Sum of researched fair prices]
- **Overcharge Amount:** [Difference between billed and fair prices]
- **Summary of Findings:**
    - List each procedure with:
        - Billed Amount
        - Fair Market Price
        - Overcharge (if any)
        - Source links for price data
- **Overall Assessment:**
    - Is the patient being overcharged?

- **Next Steps:**
    - Provide clear guidance on how to dispute charges if overcharged

  4. If denial letter or user mentions insurance issues: call insurance_advocate_agent
  5. Provide clear, empathetic explanations about findings and next steps

IF "has_files" is false:
  - Respond to the user's question directly
  - If asking about medical bills without uploading documents, politely ask them to upload

   Insurance analysis should be in the following format:
**Insurance Claim Analysis:**
- **Summary of Findings:**
    - **Insurance Company:** [Name]
    - **Policy Type:** [Type]
    - **Claimed Services:** [List of services]
    - **Denial Reasons:** [List of reasons]
    - **Policy Coverage Analysis:** [Summary of policy terms related to denied services]
    - **Conclusion:** [Whether denial was justified or not]
    - **Next Steps:** [Instructions for user on how to proceed, including appeal process if applicable]
  
ALWAYS call process_user_file first - this is mandatory for every message!
""",
    tools=[process_user_file, parse_medical_document, agent_tool.AgentTool(agent=google_search_agent), agent_tool.AgentTool(agent=insurance_advocate_agent)],
)

# cross_reference_agent = SequentialAgent(
#     name="cross_reference_agent",
#     description="An agent that cross-references medical bill data with fair price research and insurance analysis.",
#     sub_agents=[google_search_agent, insurance_advocate_agent],
#     instruction=""" 
# You are an agent that cross-references medical bill data with fair price research and insurance analysis to provide a comprehensive report to the patient. 

# You will recieve the following inputs:

# 1. Medical Bill Data: Structured data extracted from the patient's medical bill, including procedure codes, descriptions, and billed amounts.
# 2. Insurance Denial Details : Information from the patient's insurance denial letter or EOB document, including denial reasons and coverage details.

# Your task is to:
# 1. Call the google_search_agent to research fair market prices for the mentioned procedures.
# 2. Call the insurance_advocate_agent to analyze the insurance denial details.
# 3. Compile a comprehensive report that includes:
# - A comparison of billed amounts vs fair market prices for each procedure.
# - An analysis of the insurance denial in the context of the researched prices.
# - Summarise whether the patient is being overcharged and if the insurance denial was justified.
# - Clear next steps for the patient, including how to dispute overcharges and appeal insurance denials.

#     """
    
# )

# Set up Artifact Service for handling   uploaded files
artifact_service = InMemoryArtifactService()
session_service = InMemorySessionService()

runner = Runner(
    agent=root_agent,
    session_service=session_service,
    artifact_service=artifact_service,
    app_name="medical_advocate_orchestrator_agent",
    plugins=[SaveFilesAsArtifactsPlugin()],
)

async def main():
    # When user uploads a file through ADK web, it comes as a Part with inline_data
    # Example of how an uploaded image would be represented:
    
    response = await runner.run_debug(
        "I've uploaded a medical bill image. Can you help me understand if I'm being overcharged?"
    )
    print(response)

if __name__ == "__main__":
    asyncio.run(main())