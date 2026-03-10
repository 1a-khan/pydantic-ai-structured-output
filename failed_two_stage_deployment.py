from pydantic_ai import Agent, ModelAPIError, ModelHTTPError
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pathlib import Path
import json
import os

load_dotenv()  # Load environment variables from .env file

with open("pdf_data/27031062_251217_183248.ocr.txt", "r", encoding="utf-8") as file:
    pdf_content = file.read()

provider = OpenRouterProvider(api_key=os.getenv("OPENROUTER_API_KEY"))

model_candidates = [
    "anthropic/claude-sonnet-4.6",
    "google/gemini-3-pro-preview",
]

settings = {"max_tokens": 400, "temperature": 0.2}


class ExtractedData(BaseModel):
    company_name: str | None = Field(default=None, description="Company or issuer name")
    issue_date: str | None = Field(default=None, description="Issue date, preferably YYYY-MM-DD")
    receiver: str | None = Field(default=None, description="Recipient/receiver name")
    total_income: float | None = Field(default=None, description="Total income amount")
    tax_paid: float | None = Field(default=None, description="Tax paid amount")
    social_security_number: str | None = Field(default=None, description="SSN or national ID")


stage1_system_prompt = (
    "Extract the following fields from the OCR text and return a JSON object only. "
    "Use this exact JSON structure and keys:\n"
    "{\n"
    '  "company_name": string | null,\n'
    '  "issue_date": string | null,\n'
    '  "receiver": string | null,\n'
    '  "total_income": number | null,\n'
    '  "tax_paid": number | null,\n'
    '  "social_security_number": string | null\n'
    "}\n"
    "If you are unsure, use null. Do not add extra keys or explanations."
)

stage2_system_prompt = (
    "Normalize the JSON you are given into the target schema. "
    "Ensure valid JSON only, no code fences. "
    "If a value is missing or uncertain, set it to null. "
    "For amounts, return numbers without currency symbols."
)


def run_stage1(model_name: str) -> str:
    agent = Agent(
        OpenRouterModel(
            model_name,
            provider=provider,
            settings=settings,
        ),
        output_type=str,
        system_prompt=stage1_system_prompt,
        retries=3,
    )
    result = agent.run_sync(pdf_content)
    return result.output


def run_stage2(model_name: str, raw_json: str) -> str:
    agent = Agent(
        OpenRouterModel(
            model_name,
            provider=provider,
            settings=settings,
        ),
        output_type=str,
        system_prompt=stage2_system_prompt,
        retries=3,
    )
    result = agent.run_sync(raw_json)
    return result.output


try:
    stage1_output = run_stage1(model_candidates[0])
except (ModelHTTPError, ModelAPIError) as err:
    print(f"Error with primary model: {err}. Fallback model used.")
    stage1_output = run_stage1(model_candidates[1])

print("STAGE1_OUTPUT_START")
print(stage1_output)
print("STAGE1_OUTPUT_END")

try:
    stage2_output = run_stage2(model_candidates[0], stage1_output)
except (ModelHTTPError, ModelAPIError) as err:
    print(f"Error with primary model: {err}. Fallback model used.")
    stage2_output = run_stage2(model_candidates[1], stage1_output)

print("STAGE2_OUTPUT_START")
print(stage2_output)
print("STAGE2_OUTPUT_END")

review_log_path = Path("logs/manual_review.jsonl")
review_log_path.parent.mkdir(parents=True, exist_ok=True)

try:
    parsed = json.loads(stage2_output)
    data = ExtractedData.model_validate(parsed)
    print(data.model_dump())
except Exception as e:
    record = {
        "reason": "validation_failed",
        "error": str(e),
        "stage1_output": stage1_output,
        "stage2_output": stage2_output,
    }
    with open(review_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Validation failed. Logged for manual review at {review_log_path}.")
