# OCR Structured Extraction with PydanticAI + OpenRouter

This project experiments with extracting structured data from OCR text using `pydantic_ai` and OpenRouter models. The goal is to pull fields like company name, issue date, receiver, total income, tax paid, and social security number from messy OCR text.

## What I tried

- **Strict structured output** using `pydantic_ai` output types.
- **Model comparisons** across OpenRouter providers.
- **Two-stage extraction** (loose extraction → normalization) when validation is unreliable.
- **Tool-based output** (`ToolOutput`) to force schema compliance.

The consistent result is that some models follow structured output far better than others. Claude and GPT-5.* have been the most reliable in this setup.

## Files

- `structured_run.py`: Strict structured output with `pydantic_ai` and OpenRouter.
- `main.py`: Two-stage extraction flow (looser extraction + normalization).
- `env.example`: Example environment variables.
- `pdf_data/`: OCR text inputs (ignored in git).
- `logs/`: Manual review logs (ignored in git).

## Quick Start

1. Create `.env` with your OpenRouter API key:

```bash
OPENROUTER_API_KEY=your_key_here
```

2. Run the structured extraction:

```bash
./venv/bin/python structured_run.py
```

## Notes

- For OpenRouter JSON mode, prompts often need the word `json`.
- Some models do not reliably support structured outputs and will fail validation.
- When validation fails, a manual review log can capture the raw outputs for inspection.
