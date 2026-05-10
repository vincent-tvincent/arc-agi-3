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
    analyze_experience.py
    experience_collection/
      quickstart.py
      collect_experience.py
      batch_collect_training.py
      batch_collect_training.example.yaml
    playground/
      manual_play.py
      manual_play_keymap.yaml
    arc3_pipeline/
      frame_utils.py
      experience.py
      model_families.py
    hdbscan_pipeline/
      vector_builder.py
      cluster_games.py
  environment_files/
  recordings/
  analysis/
  analysis_csv/
  clusters/
  runs/
  training_runs/
  training_examples/
```

Important files:

- `src/experience_collection/quickstart.py` is the original quick smoke-test script for the official toolkit.
- `src/experience_collection/collect_experience.py` runs a probing policy against one or more games and writes transition data to `runs/`.
- `src/experience_collection/batch_collect_training.py` collects every available game with configurable steps/seeds from a YAML file.
- `src/playground/manual_play.py` opens an interactive grid playground using key bindings and mouse clicks.
- `src/analyze_experience.py` reads a collected run and prints ranked model-family hypotheses.
- `src/arc3_pipeline/frame_utils.py` contains frame conversion, hashing, changed-cell detection, color counts, background detection, and connected-component extraction.
- `src/arc3_pipeline/experience.py` defines the `Transition` record saved by the collector.
- `src/arc3_pipeline/model_families.py` contains the bounded hypothesis generator.
- `src/hdbscan_pipeline/vector_builder.py` converts training examples into fixed-length run vectors.
- `src/hdbscan_pipeline/cluster_games.py` clusters those vectors for unsupervised game-type discovery.
- `environment_files/` stores downloaded/local ARC-AGI-3 game files.
- `recordings/` stores official toolkit recordings when `save_recording=True`.
- `runs/` stores this pipeline's compact JSONL transition logs.
- `training_runs/` is the recommended raw-transition output directory for larger training collections.
- `training_examples/` stores step-level JSONL examples for model training.
- `analysis/` stores machine-readable per-run analyzer JSON.
- `analysis_csv/` stores compact analyzer CSV files for human debugging.
- `clusters/` stores unsupervised clustering outputs.

## API Key

Copy the example environment file:

```bash
cp .env.example .env
```

Then edit `.env`:

```bash
ARC_API_KEY=put-your-api-key-here
ENVIRONMENTS_DIR=environment_files
RECORDINGS_DIR=recordings
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

The collector now uses root-level environment and recording directories by default:

```text
environment_files/
recordings/
```

So from the project root, offline collection will load games from `environment_files/`, and toolkit recordings will be written to `recordings/`.

If you need custom locations, use:

```bash
python src/experience_collection/collect_experience.py \
  --game ls20 \
  --steps 80 \
  --offline \
  --environments-dir environment_files \
  --recordings-dir recordings
```

To collect transitions from the downloaded `ls20` game without contacting the API:

```bash
python src/experience_collection/collect_experience.py --game ls20 --steps 80 --offline
```

To render in the terminal while collecting:

```bash
python src/experience_collection/collect_experience.py --game ls20 --steps 80 --offline --render terminal-fast
```

To collect every game available to your API key:

```bash
python src/experience_collection/collect_experience.py --game all --steps 80
```

The collector writes compact transition logs like:

```text
runs/ls20-9607627b_seed0.jsonl
```

To generate files under `training_runs/` instead, pass `--out-dir training_runs`:

```bash
python src/experience_collection/collect_experience.py --game ls20 --steps 80 --offline --out-dir training_runs
```

To regenerate one run file per downloaded local game under `training_runs/`:

```bash
python src/experience_collection/collect_experience.py --game all --steps 80 --offline --out-dir training_runs
```

For real training data collection, prefer writing raw transition logs and training examples together:

```bash
python src/experience_collection/collect_experience.py \
  --game all \
  --steps 200 \
  --offline \
  --out-dir training_runs \
  --training-out-dir training_examples
```

This produces:

```text
training_runs/<game_id>_seed0.jsonl
training_examples/<game_id>_seed0.examples.jsonl
```

To run multiple seeds per game, either call the collector with `--seed` repeatedly or use the YAML-driven batch collector.

Example batch collection:

```bash
python src/experience_collection/batch_collect_training.py --config src/experience_collection/batch_collect_training.example.yaml
```

The batch collector reads settings such as `steps`, `seeds`, `offline`, `environments_dir`, `recordings_dir`, `out_dir`, and `training_out_dir` from YAML. The example config is commented, and its default output directories are:

```text
training_runs/
training_examples/
```

By default, the collector also writes step-level training examples like:

```text
training_examples/ls20-9607627b_seed0.examples.jsonl
```

Each training row represents one observed transition:

```json
{
  "example_id": "ls20-9607627b_seed0_step12",
  "analysis_id": "ls20-9607627b_seed0",
  "game_id": "ls20-9607627b",
  "seed": 0,
  "step": 12,
  "input": {
    "frame": [[0, 1, 2]],
    "available_actions": ["ACTION1", "ACTION2"]
  },
  "action": {
    "name": "ACTION2",
    "data": {}
  },
  "target": {
    "next_frame": [[0, 2, 2]],
    "changed_cells": [{"x": 1, "y": 0, "before": 1, "after": 2}]
  },
  "outcome": {
    "state_after": "NOT_FINISHED",
    "levels_completed_before": 0,
    "levels_completed_after": 0,
    "level_delta": 0,
    "is_win": false,
    "is_game_over": false,
    "changed_cell_count": 1,
    "frame_changed": true
  },
  "metadata": {
    "before_hash": "abc",
    "after_hash": "def"
  }
}
```

Use `--no-training-export` if you only want the raw transition log:

```bash
python src/experience_collection/collect_experience.py --game ls20 --steps 80 --offline --no-training-export
```

Use `--training-out-dir` to choose a different training-example directory:

```bash
python src/experience_collection/collect_experience.py --game ls20 --steps 80 --offline --training-out-dir data/train
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
python src/experience_collection/collect_experience.py --game ls20 --steps 80 --offline --append
```

## Analyze Experience

After collecting a run:

```bash
python src/analyze_experience.py runs/ls20-9607627b_seed0.jsonl
```

The analyzer still prints ranked hypotheses to the terminal. It also writes machine-readable JSON and human-debug CSV by default:

```text
analysis/ls20-9607627b_seed0.analysis.json
analysis_csv/ls20-9607627b_seed0.analysis.csv
```

The analysis JSON contains game-level labels that can be joined to training examples through `analysis_id`:

```json
{
  "analysis_id": "ls20-9607627b_seed0",
  "game_id": "ls20-9607627b",
  "transition_count": 80,
  "top_family": "click_select",
  "key_actions": ["ACTION6"],
  "second_family": "toggle",
  "second_actions": ["ACTION5"],
  "hypotheses": []
}
```

You can override either export path:

```bash
python src/analyze_experience.py runs/ls20-9607627b_seed0.jsonl \
  --json-out custom.analysis.json \
  --csv-out custom.analysis.csv
```

This prints ranked hypotheses. For example, if the analyzer sees repeated unit movement after actions, it may produce a `navigation` hypothesis with an inferred action map:

```text
ACTION1 -> UP
ACTION2 -> DOWN
ACTION3 -> LEFT
ACTION4 -> RIGHT
```

The exact output depends on the transitions collected. Short or uninformative probing runs may produce only low-confidence hypotheses.

## Manual Playground

The manual playground reads non-complex keyboard bindings from a keymap file and sends mouse clicks as complex coordinate actions:

```bash
python src/playground/manual_play.py ls20
```

Use a custom keymap with:

```bash
python src/playground/manual_play.py cn04 --keymap src/playground/manual_play_keymap.yaml
```

The keymap format is:

```yaml
keys:
  Up: ACTION1
  Down: ACTION2
  Left: ACTION3
  Right: ACTION4
mouse_action: ACTION6
```

## Cluster Game Runs

After collecting training examples, run HDBSCAN clustering from the project root:

```bash
python src/hdbscan_pipeline/cluster_games.py \
  --examples-dir training_examples \
  --out-dir clusters \
  --method hdbscan
```

This pipeline does not consume the old hardcoded analyzer results. It builds fixed-length vectors directly from `training_examples/*.examples.jsonl`, then clusters game runs from those transition statistics.

The default HDBSCAN outputs are:

```text
clusters/game_clusters.csv
clusters/game_clusters.json
clusters/feature_columns.json
```

Use the YAML config when you want repeatable clustering settings:

```bash
python src/hdbscan_pipeline/cluster_games.py --config src/hdbscan_pipeline/config.example.yaml
```

CLI arguments still override YAML values. For example:

```bash
python src/hdbscan_pipeline/cluster_games.py \
  --config src/hdbscan_pipeline/config.example.yaml \
  --min-cluster-size 4 \
  --min-samples 2
```

Main tunable HDBSCAN parameters:

- `min_cluster_size`: minimum number of runs needed to form a cluster.
- `min_samples`: how conservative the model is about marking points as outliers.

Cluster label `-1` means HDBSCAN treated that run as noise or an outlier, not as a discovered game type.

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

## GNN + PPO Agent Pipeline

This repo now also includes an additive object-centric agent under `src/arc_agent/`. It follows the plan in `arc_agi3_gnn_ppo_implementation_plan.md`: grid segmentation, object tracking, graph construction, belief updates, candidate subgoals, A* planning, optional GNN affordance prediction, and PPO high-level strategy selection.

The command implementations live in `src/arc_agent/cli/`. The root `scripts/` files are compatibility wrappers for the command patterns in the implementation plan.

Install from the project root:

```bash
pip install -e .
pip install -e .[train]
```

Run the focused tests:

```bash
pytest tests -q
```

Collect synthetic/mock rollout data:

```bash
python scripts/collect_rollouts.py --config configs/train_gnn.yaml --env mock --episodes 5000
```

Train the GNN:

```bash
python scripts/train_gnn.py --config configs/train_gnn.yaml
```

Train PPO with a pretrained GNN:

```bash
python scripts/train_ppo.py --config configs/train_ppo.yaml --pretrained-gnn runs/<run>/checkpoints/gnn_best_val_f1.pt
```

Resume PPO:

```bash
python scripts/train_ppo.py --config configs/train_ppo.yaml --resume runs/<run>/checkpoints/ppo_latest.pt
```

Evaluate:

```bash
python scripts/evaluate_agent.py --config configs/debug_cpu.yaml --env mock --episodes 20
python scripts/evaluate_agent.py --config configs/hardware_3080_10gb.yaml --env arcagi3 --checkpoint runs/<run>/checkpoints/ppo_best_success.pt
```

Replay and inspect:

```bash
python scripts/replay_episode.py --replay runs/<run>/eval/replays/episode_0000.json.gz
python scripts/inspect_graphs.py --replay runs/<run>/eval/replays/episode_0000.json.gz --step 10
```

Logs, metrics, replays, and checkpoints are stored under `runs/<run_name>/`. The detailed docs are in `docs/architecture.md`, `docs/training.md`, `docs/data_format.md`, and `docs/evaluation.md`.
