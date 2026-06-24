# World Cup Prediction Harness

This project turns a match-prediction idea into a repeatable harness:

1. Load and fact-check match records.
2. Run four independent role analyses.
3. Run a summarizer after the role analyses complete.
4. Generate a dated prediction report.
5. Review real results, update a ledger, and feed lessons into the next run.

It is for research and entertainment only. It does not provide betting advice.

## Layout

- `worldcup_prediction/`: Python package and CLI.
- `worldcup_prediction/sample_data/`: offline sample matches and results.
- `runs/YYYY-MM-DD/`: generated run output.
- `ledger/results_ledger.jsonl`: long-term reviewed predictions.
- `ledger/lessons.md`: mistake book used as context in later predictions.
- `tests/`: minimal pytest coverage.

## Run A Sample Prediction

From this directory. Use `python3` if your system does not provide a `python` command:

```bash
python3 -m worldcup_prediction predict --date 2026-06-23
```

The command writes:

- `runs/2026-06-23/daily_report.md`
- `runs/2026-06-23/matches.json`
- `runs/2026-06-23/predictions.json`
- `runs/2026-06-23/agents/`
- `runs/2026-06-23/sources.json`
- `runs/2026-06-23/run.log`

Print an existing report:

```bash
python3 -m worldcup_prediction report --date 2026-06-23
```

## Review Real Results

Results JSON can be a list or an object with a `results` array:

```json
{
  "results": [
    {"match_id": "sample-01-BRA-CRO", "home_score": 2, "away_score": 1}
  ]
}
```

Run review:

```bash
python3 -m worldcup_prediction review \
  --date 2026-06-23 \
  --results worldcup_prediction/sample_data/results_2026-06-23.json
```

View cumulative stats:

```bash
python3 -m worldcup_prediction stats
```

## Data Sources

The data-source layer is pluggable. The default `sample` source is fully offline and repeatable. A local JSON source can be supplied with:

```bash
python3 -m worldcup_prediction predict \
  --date 2026-06-23 \
  --source-json path/to/matches.json
```

Every prediction is based on explicit `MatchRecord` data written to `matches.json`; the agents should never invent the schedule.

## LLM Provider

Default `--llm-provider auto` uses a real OpenAI-compatible endpoint and prefers the local DDSS config at `~/.config/worldcup_prediction/ddss.env`.
It can read:

- `DDSS_API_KEY` for the default DDSS endpoint, or
- `AUTODL_API_KEY` for AutoDL.Art/Qwen when `--llm-provider qwen` or `--llm-provider autodl` is selected, or
- `OPENAI_API_KEY` or `WCP_OPENAI_API_KEY`
- `OPENAI_MODEL` or `WCP_OPENAI_MODEL` if you want to override the default model
- `OPENAI_BASE_URL` or `WCP_OPENAI_BASE_URL` if you want to override the default endpoint

The DDSS default is `gpt-5.5` on `https://code.ddsst.online/v1`. Qwen/AutoDL defaults are `Qwen3.5-397B-A17B` on `https://www.autodl.art/api/v1`; put that config in `~/.config/worldcup_prediction/qwen.env` and select it explicitly with `--llm-provider qwen`.
The harness does not fall back to deterministic mock in default mode. Use `--llm-provider mock` only when you explicitly want a fully offline deterministic run, for example in tests.

When the active backend is DDSS, the harness calls DDSS one match at a time and aggregates the results into the same daily report. This avoids one large daily-batch request timing out and lets the run continue if a single match fails. DDSS Responses requests use reasoning effort `high`.

Prompt templates are centralized in `worldcup_prediction/prompts.py`.

## Tests

```bash
python3 -m unittest discover -s tests
```

If `pytest` is installed, `pytest -q` also works.
