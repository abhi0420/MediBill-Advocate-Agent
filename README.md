# MediBill Advocate Agent

## Built for Google X Kaggle Capstone Project

---

## Problem Statement

Medical billing in the US is complex, error-prone, and often results in denied insurance claims or inflated charges. Patients and advocates need tools to parse medical documents, research fair prices, and analyze insurance denials efficiently.

## Solution Overview

MediBill Advocate Agent is an AI-powered workflow that automates medical document parsing, fair price research, and insurance denial analysis using Google ADK and Gemini LLM. The system orchestrates multiple specialized agents to deliver actionable insights and recommendations for patients and advocates.

## Architecture

- **Orchestrator Agent**: Manages workflow, routes tasks, and aggregates results.
- **Document Parser Agent**: Parses multiple medical images/documents, extracts structured data, and deletes files after parsing.
- **Fair Price Research Agent**: Uses Google Search to research fair prices for procedures and medications.
- **Insurance Advocate Agent**: Analyzes insurance denials, provides recommendations, and leverages Google Search for supporting evidence.

Agents communicate via explicit context passing and schema-based outputs, ensuring robust and interpretable results.

## Technologies Used

- **Google ADK (Agent Development Kit)**: Agent orchestration and tool integration.
- **Gemini LLM**: Core language model for all agents.
- **Python**: Main programming language.
- **Pydantic**: Output schema validation.
- **dotenv**: Environment variable management.
- **VS Code**: Development environment.

## Build & Run Instructions

1. **Clone the repository**
   ```powershell
   git clone https://github.com/abhi0420/MediBill-Advocate-Agent.git
   cd MediBill-Advocate-Agent
   ```
2. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```
3. **Configure environment variables**
   - Copy `.env.example` to `.env` and fill in your API keys and settings.
4. **Run the ADK web server**
   ```powershell
   adk web
   ```
5. **Upload medical documents/images**
   - Place files in the `uploads/` folder.
6. **Interact with the Orchestrator Agent**
   - Use the web interface to trigger parsing, research, and analysis workflows.

## Demo Workflow

1. **File Upload**: User uploads medical documents/images to `uploads/`.
2. **Document Parsing**: Document Parser Agent extracts structured data and deletes files post-parsing.
3. **Agent Routing**: Orchestrator Agent routes parsed data to Fair Price Research and Insurance Advocate Agents.
4. **Analysis & Research**: Agents perform Google Search, analyze denials, and research fair prices.
5. **Summary Presentation**: Orchestrator Agent aggregates results and presents actionable insights.

## Agent Rationale

- **Modular Design**: Each agent specializes in a domain, improving accuracy and maintainability.
- **Explicit Context Passing**: Ensures agents receive only relevant information, reducing errors.
- **Schema-Based Outputs**: Enforces structured, interpretable results for downstream processing.
- **Automated File Management**: Prevents clutter and privacy risks by deleting files after parsing.

## Future Improvements

- Integrate OCR for handwritten documents.
- Add support for additional languages and international billing standards.
- Enhance insurance denial appeal recommendations with legal references.
- Expand fair price research to include real-time market data.
- Improve UI/UX for broader accessibility.
- Add audit logging and privacy controls.

## Contributing

Pull requests and issues are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License.

---

For questions or support, contact [Abhinand](mailto:your-email@example.com).
