# ARC-AGI-3 Training Pipeline

This project is a small non-LLM starting point for ARC-AGI-3 research. The goal is to build an agent that can enter an unfamiliar interactive grid game, probe the available actions, observe how the board changes, infer a likely game model, and later plan actions using that model.

ARC-AGI-3 is different from ARC-AGI-1/2. There is no static `input grid -> output grid` task to solve directly. Instead, the agent interacts with a game:

```text
observe frame -> choose action -> observe next frame -> repeat
```

So this code focuses on collecting and analyzing transitions:

```text
frame_t + action_t -> frame_t+1
```

The current pipeline is intentionally modest. It does not claim to solve ARC-AGI-3 yet. It gives you the foundation for training/debugging:

1. collect interaction data from games
2. compute frame differences
3. extract simple object/component summaries
4. generate bounded hypotheses from known model families
5. rank hypotheses by evidence
6. leave clear extension points for planners and stronger agents

## Project Layout

```text
arc-agi-3/
  README.md
  TRAINING_PIPELINE.md
  .env.example
  src/
    quickstart.py
    collect_experience.py
    analyze_experience.py
    arc3_pipeline/
      frame_utils.py
      experience.py
      model_families.py
    environment_files/
    recordings/
  runs/
```

Important files:

- `src/quickstart.py` is the original quick smoke-test script for the official toolkit.
- `src/collect_experience.py` runs a probing policy against one or more games and writes transition data to `runs/`.
- `src/analyze_experience.py` reads a collected run and prints ranked model-family hypotheses.
- `src/arc3_pipeline/frame_utils.py` contains frame conversion, hashing, changed-cell detection, color counts, background detection, and connected-component extraction.
- `src/arc3_pipeline/experience.py` defines the `Transition` record saved by the collector.
- `src/arc3_pipeline/model_families.py` contains the bounded hypothesis generator.
- `src/environment_files/` stores downloaded/local ARC-AGI-3 game files.
- `src/recordings/` stores official toolkit recordings when `save_recording=True`.
- `runs/` stores this pipeline's compact JSONL transition logs.

## API Key

Copy the example environment file:

```bash
cp .env.example .env
```

Then edit `.env`:

```bash
ARC_API_KEY=put-your-api-key-here
ENVIRONMENTS_DIR=src/environment_files
RECORDINGS_DIR=src/recordings
```

The official `arc_agi` package automatically loads `.env` from the project root.

You can still work offline with already-downloaded games by passing `--offline`.

## Environment Setup

From the project root:

```bash
cd /home/blue-lobster/p2/UCI/CS274p/arc-agi-3
source ../arc_agi/bin/activate
```

The virtual environment currently lives one directory above this project:

```text
/home/blue-lobster/p2/UCI/CS274p/arc_agi
```

If you recreate the venv later, keep it in one stable location. Python virtual environments should not be moved after creation because their scripts often contain absolute paths.

## Collect Experience

To collect transitions from the downloaded `ls20` game without contacting the API:

```bash
python src/collect_experience.py --game ls20 --steps 80 --offline
```

To render in the terminal while collecting:

```bash
python src/collect_experience.py --game ls20 --steps 80 --offline --render terminal-fast
```

To collect every game available to your API key:

```bash
python src/collect_experience.py --game all --steps 80
```

The collector writes compact transition logs like:

```text
runs/ls20-9607627b_seed0.jsonl
```

Each JSONL row stores:

```text
game_id
step
action
action_data
before_hash
after_hash
before_frame
after_frame
changed_cells
state
levels_completed
available_actions
```

By default, a run file is replaced each time. Add `--append` if you want to accumulate more probes in the same file:

```bash
python src/collect_experience.py --game ls20 --steps 80 --offline --append
```

## Analyze Experience

After collecting a run:

```bash
python src/analyze_experience.py runs/ls20-9607627b_seed0.jsonl
```

This prints ranked hypotheses. For example, if the analyzer sees repeated unit movement after actions, it may produce a `navigation` hypothesis with an inferred action map:

```text
ACTION1 -> UP
ACTION2 -> DOWN
ACTION3 -> LEFT
ACTION4 -> RIGHT
```

The exact output depends on the transitions collected. Short or uninformative probing runs may produce only low-confidence hypotheses.

## Model Families

The current code uses seven starting families:

```text
navigation
click_select
toggle
collection
push_block
sequencing
pattern_transform
```

These are not meant to cover every possible ARC-AGI-3 game. They are useful priors. The agent should eventually do this:

```text
try known families
score them against evidence
fall back to open-ended exploration if confidence is low
discover new patterns from logs
```

The current implementation has concrete checks for:

- `navigation`: looks for action effects that resemble one-cell movement.
- `click_select`: looks for actions that accept coordinate data, such as `{"x": ..., "y": ...}`.
- `toggle`: looks for small local color/state changes.

The remaining families currently produce conservative low-confidence summaries. They are placeholders for future specialized tests.

## Core Concept

The pipeline is based on bounded hypothesis search.

Instead of trying every possible model, it generates a small set of plausible models from evidence:

```text
changed cells -> object candidates -> action effects -> model-family hypotheses -> ranked models
```

A possible model should satisfy:

```text
simple
consistent with observed transitions
able to predict some next-frame changes
useful for planning
stable across levels when possible
```

If every model scores poorly, the future agent should switch into exploration mode:

```text
try actions from new states
test coordinate actions on object centers
avoid repeated states
avoid actions that caused GAME_OVER
prefer actions that create novel, meaningful changes
```

## Next Steps

The next useful additions are:

1. Add a shortest-path planner for strong `navigation` hypotheses.
2. Add click-candidate search for `click_select` hypotheses.
3. Add replay validation so failed predictions reduce a hypothesis score.
4. Keep action mappings and object rules across levels of the same game.
5. Add a benchmark script that runs collection and analysis across all available games.

This scaffold is the lab bench: it gathers evidence, summarizes it, and gives you places to plug in stronger reasoning.
