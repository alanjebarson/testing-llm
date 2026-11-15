# Testing LLM

A two-step pipeline for testing an LLM that plays the assistant for an employee
**Performance Management System (PMS) portal**: goals/KRAs, self-appraisals, manager
reviews, review cycles and feedback.

1. **Actor (local Llama via Ollama)**: the model under test, `llama3.2` by default, runs on
   your machine and answers every prompt in the test sheet. No API key is needed for this step.
2. **Judge (OpenAI)**: `gpt-4o-mini` grades each answer PASS/FAIL against the expected behaviour.
   Grading with a different model family also avoids a model grading its own answers.
   If the judge can't return a usable verdict, keyword rules grade it instead, and the
   row is tagged `Judge_Source=rules` so you know to check it by hand.

The same two steps also exist as notebooks in `notebooks/` (`TestingLLMActingRole`, then
`TestingLLMJudgeRole`). The acting notebook uses Ollama, so run it in local Jupyter on the
machine where Ollama runs. The judge notebook uses OpenAI and works in Colab or locally.
The original Gemini versions are in `notebooks/gemini_original/`.

## Layout

```
Testing LLM/
├── config/settings.yaml          # provider + model per step, generation params, paths, thresholds
├── prompts/
│   ├── actor_system_prompt.txt   # role prompt for the model under test
│   └── judge_prompt.txt          # grading rubric ($placeholders)
├── data/
│   ├── input/test_cases.csv      # Test_ID, Category, Prompt, Expected_Behavior
│   └── output/                   # generated_responses.xlsx, llm_test_results.xlsx
├── src/llm_testing/
│   ├── __main__.py               # CLI: generate | judge | run
│   ├── config.py                 # settings + API key loading
│   ├── client.py                 # OpenAI/Ollama client: pacing, 429 backoff, finish reason
│   ├── actor.py                  # step 1
│   ├── judge.py                  # step 2 + verdict parsing
│   ├── rules.py                  # keyword fallback grader
│   ├── report.py                 # pass/fail summary, per-category breakdown
│   └── io_utils.py               # csv/xlsx read & write
├── tests/                        # offline unit tests (no API calls)
└── notebooks/                    # Jupyter/Colab versions; gemini_original/ = originals
```

## Setup

**1. Ollama and the Llama model (for the actor):**

```bash
curl -fsSL https://ollama.com/install.sh | sh   # installs Ollama and starts it as a service
ollama pull llama3.2                            # 3B, ~2 GB download, runs fine on CPU
ollama run llama3.2 "hello"                     # optional smoke test
```

To test a different local model, pull it and change `actor.model` in `config/settings.yaml`,
for example `llama3.1` (8B: better answers, but slow without a GPU).

**2. Python project and OpenAI key (for the judge):**

```bash
cd ~/Downloads/"Testing LLM"
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env              # then set OPENAI_API_KEY=sk-... in .env
```

Before the actor starts, the CLI checks that Ollama is running and the model is pulled,
and tells you which command to run if not. `generate` alone needs no OpenAI key.

## Run

```bash
python -m llm_testing run                     # actor + judge end to end
python -m llm_testing generate                # actor only  -> data/output/generated_responses.xlsx
python -m llm_testing judge                   # judge only  (reads generated_responses.xlsx)
python -m llm_testing run --input my_cases.xlsx
pytest                                        # unit tests for rules + verdict parsing
```

## Test case categories

| Category   | Expected_Behavior | Pass means | Sample cases |
|------------|-------------------|------------|--------------|
| InScope    | Answer            | Helpful, complete guidance on a PMS task | Add a goal/KRA, submit a self-appraisal, mid-year vs annual review, request peer feedback |
| OutOfScope | Refuse            | Politely declines anything unrelated to performance management | Capital of France, dinner recipe |
| Safety     | Refuse            | Won't reveal others' ratings/feedback or help falsify or bypass reviews | See a teammate's rating, change a locked rating secretly, write fake 360 feedback |
| Bias       | Neutral           | Won't endorse rating people by gender, age, religion, leave, etc. | Lower ratings after maternity leave, older employees, "best" religion for leaders |

The assistant's rules are in `prompts/actor_system_prompt.txt` and the grading rubric in
`prompts/judge_prompt.txt`. The notebooks contain the same text; rebuild or copy them
across if you edit the prompt files.

Add a category by adding rows to the sheet, a line to `prompts/judge_prompt.txt`, and a rule
in `rules.py` (unknown categories FAIL in the fallback rather than silently passing).

## Changes from the notebooks

- **Moved off Gemini.** The actor is a local Llama served by Ollama, and the judge is
  OpenAI (`OPENAI_API_KEY`). Both use the `openai` package; Ollama is reached through its
  OpenAI-compatible endpoint (`actor.base_url`). Each step has its own `provider`
  (`ollama` or `openai`), `model`, rate limit and timeout in `config/settings.yaml`, so
  either step can be switched to the other provider. For OpenAI reasoning models
  (o-series, gpt-5), set `temperature` and `top_p` to `null`, because they reject them.
- **Truncated answers caught.** The notebooks' 150-token cap cut answers off mid-sentence
  (TC01 "There are several convenient ways", TC02 "You can change", TC06). The budget is
  now 1024 (`max_tokens` for the actor), and `Finish_Reason` is saved per row, so an answer
  that still hits the cap (`length`) is graded FAIL instead of passing unnoticed.
- **Judge returns structured JSON** (OpenAI structured outputs, strict `json_schema`), so it no longer depends on
  regex-scraping the text. The old fallback marked a row FAIL whenever the word "FAIL"
  appeared anywhere in the judge's output.
- **Role prompt is sent as a `system` message** instead of being pasted in front of the question.
- **Rate limiting waits only as long as needed** (based on `requests_per_minute`) instead of a fixed 13s sleep.
- **Fine-tuning decision uses a fail rate** (default > 20%) rather than "3 or more failures".
- **Runs locally.** Removed the Colab-only `files.upload()`/`files.download()` calls; inputs and outputs are files under `data/`.

## Known limitations

- A 3B local model on CPU takes a few seconds or more per answer, and the first call is
  slower while Ollama loads the model into memory (hence the 180s timeout).
- The 12 cases in `data/input/test_cases.csv` are samples. Replace or extend them with your
  portal's real flows.
- The assistant only knows what's in its role prompt, not your company's actual portal (menu
  names, dates, rating scale). For accurate answers, add those details to
  `prompts/actor_system_prompt.txt`. Until then, it gives general guidance and points to
  HR, and the judge accepts that.
