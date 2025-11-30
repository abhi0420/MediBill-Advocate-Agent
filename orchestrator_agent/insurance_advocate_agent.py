from google.adk.tools import google_search
from google.adk.sessions import InMemorySessionService
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.models.google_llm import Gemini
from dotenv import load_dotenv
import sys


load_dotenv()

if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

app_name = "insurance_advocate_agent"
user_id = "test_user"

model = "gemini-2.5-flash"

insurance_advocate_agent = LlmAgent(
    name="insurance_advocate_agent",
    model=Gemini(model=model),
    description="An agent that acts as a medical insurance advocate to help patients understand and challenge their medical bills and insurance claims.",
    instruction="""
You are a medical insurance advocate agent. You will be called in case any patient's insurance claim has been denied or if there are discrepancies in the insurance explanation of benefits (EOB) documents.

When a user provides an insurance denial letter or EOB document, follow these steps:
1. Carefully review the denial reasons or coverage details provided in the document.
2. If the denial letter does not contain details of insurance provider or policy type, ask the user for this information.
3. Based on insurance company name and policy type, search online for the specific insurance policy terms and coverage rules.
4. With the policy details, analyze whether the denial or coverage decisions align with the stated policy
5. Report your findings in the form of a clear summary & next steps for the user to take, including how to file an appeal if applicable.

Summary can be in the following format:
**Summary of Findings:**
- **Insurance Company:** [Name]
- **Policy Type:** [Type]
- **Claimed Services:** [List of services]
- **Denial Reasons:** [List of reasons]
- **Policy Coverage Analysis:** [Summary of policy terms related to denied services]
- **Conclusion:** [Whether denial was justified or not]
- **Next Steps:** [Instructions for user on how to proceed, including appeal process if applicable  

""", 
tools=[google_search],


)

runner = Runner(app_name=app_name, agent=insurance_advocate_agent, session_service=InMemorySessionService())

async def main():
    response = await runner.run_debug("""
HDFC ERGO General Insurance Company Ltd.
Claims Review Department
1st Floor, HDFC House, Lower Parel, Mumbai – 400013

Date: 27-Nov-2025

To:
Mr. Ramesh Kumar
Patient ID: RK-45892
                                      


Subject: Claim Rejection Notification
Policy Number: 2314/98374652/00
Policy Name: HDFC ERGO Health Suraksha Silver Plan

Dear Mr. Kumar,

We have completed the review of your claim submitted on 22-Nov-2025 for reimbursement of medical expenses incurred at Sunrise Care & Diagnostics Pvt. Ltd.

After evaluating the documents provided, the following items have been rejected:

Rejected Claim Line Items

Room Rent Charges – ₹5,500.00
Reason: The billed amount exceeds the admissible limit under the policy.

Blood Test Panel – ₹3,200.00
Reason: Supporting laboratory reports required for verification were not provided.

X-Ray (Chest PA + Lateral) – ₹4,800.00
Reason: The claimed amount exceeds the allowable charge threshold.

Medicines & Consumables – ₹6,500.00
Reason: Itemized breakdown with drug names, quantities, and batch numbers was not submitted.

Consultation & Procedure Fee – ₹7,000.00
Reason: Absence of doctor’s notes / treatment summary required for claim validation.

Final Status: CLAIM REJECTED

You may resubmit the claim with the required supporting documents within 30 days for reconsideration.

For assistance, please contact our Claims Support team at the number listed on your policy document.

Sincerely,
Claims Review Officer
HDFC ERGO General Insurance Company Ltd
""")
    

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())