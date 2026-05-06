# ARC-AGI-3 Training Pipeline

This is a small non-LLM starting pipeline for the method we discussed:

1. collect interaction data
2. analyze state transitions
3. generate bounded hypotheses from seven model families
4. rank hypotheses by evidence
5. later plug in planners/policies for the best family

## API Key

Copy the placeholder and fill in your key:

```bash
cp .env.example .env
```

Then edit `.env`:

```bash
ARC_API_KEY=put-your-api-key-here
```

The toolkit loads `.env` automatically.

## Collect Experience

From the project root:

```bash
cd /home/blue-lobster/p2/UCI/CS274p/arc-agi-3
source ../arc_agi/bin/activate
python src/collect_experience.py --game ls20 --steps 80 --render terminal-fast
```

This writes a JSONL run under `runs/` and also lets the official toolkit save its own recordings under `recordings/`.

If the game is already downloaded in `environment_files/`, you can avoid API calls:

```bash
python src/collect_experience.py --game ls20 --steps 80 --offline
```

By default, the run file is replaced each time. Add `--append` if you want to accumulate several probes in the same file.

To collect every game your key can access:

```bash
python src/collect_experience.py --game all --steps 80
```

## Analyze A Run

```bash
python src/analyze_experience.py runs/ls20-9607627b_seed0.jsonl
```

The analyzer currently creates hypotheses for:

- navigation
- click/select
- toggle
- collection
- push/block
- sequencing
- pattern transform

The first three have concrete evidence checks. The last four are intentionally conservative placeholders that summarize components/progress until you add more specialized tests.

## Next Implementation Milestones

1. Add a shortest-path planner when the top hypothesis is `navigation`.
2. Add click-candidate search when the top hypothesis is `click_select`.
3. Add replay-based validation: execute a predicted plan and lower the hypothesis score when prediction fails.
4. Add cross-level memory: keep action mappings and object rules after each level transition.
