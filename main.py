from pydantic_ai import Agent, ModelAPIError, ModelHTTPError, ToolOutput
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()


class ExtractedData(BaseModel):
    company_name: str | None = Field(default=None, description="Company or issuer name")
    issue_date: str | None = Field(default=None, description="Issue date, preferably YYYY-MM-DD")
    receiver: str | None = Field(default=None, description="Recipient/receiver name")
    total_income: float | None = Field(default=None, description="Total income amount")
    tax_paid: float | None = Field(default=None, description="Tax paid amount")
    social_security_number: str | None = Field(default=None, description="SSN or national ID")


def build_agent(model_name: str) -> Agent[None, ExtractedData]:
    provider = OpenRouterProvider(api_key=os.getenv("OPENROUTER_API_KEY"))
    model = OpenRouterModel(
        model_name,
        provider=provider,
        settings={"max_tokens": 400, "temperature": 0.0},
    )
    return Agent(
        model,
        output_type=ToolOutput(ExtractedData, name="return_extracted_data", strict=True),
        system_prompt=(
            "Extract the requested fields from the OCR text. "
            "If a field is missing, return null. "
            "For amounts, return numeric values without currency symbols."
        ),
        retries=1,
    )

# I have tried with german payslip. This model after ocr text extraction working fine
def main() -> None:
    with open("pdf_data/Scan – 2026-03-10 15_32_32.ocr.txt", "r", encoding="utf-8") as f:
        ocr_text = f.read()

    model_candidates = [
        "anthropic/claude-sonnet-4.6",
        "openai/gpt-5.2"        
    ]

    agent = build_agent(model_candidates[0])

    try:
        result = agent.run_sync(ocr_text)
    except (ModelHTTPError, ModelAPIError) as err:
        print(f"Error with primary model: {err}. Fallback model used.")
        fallback_agent = build_agent(model_candidates[1])
        result = fallback_agent.run_sync(ocr_text)

    print(result.output.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
