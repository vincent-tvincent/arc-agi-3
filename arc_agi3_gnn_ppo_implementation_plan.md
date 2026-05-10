# ARC-AGI-3 GNN + PPO Implementation Plan for Codex

> **Purpose of this document:** This is an implementation prompt/plan intended to be given to Codex or another coding agent. Prioritize modular code, explicit configuration, clear logs, reproducible experiments, checkpointing, and easy human tuning.
>
> **Target hardware for first version:** Intel i9-10850K, RTX 3080 10GB VRAM, 32GB DDR4 RAM, 100GB SSD volume, 1TB HDD volume.
>
> **Core design principle:** Do not build a game-specific solver. Build a general object-centric interactive reasoning agent. The neural models should learn priors and heuristics. The per-game adaptation should happen through belief updates, causal graph updates, and replanning, not through online gradient updates during evaluation.

---

## 0. Research References Codex Should Inspect

Codex should read or search these before implementation details are finalized:

1. ARC-AGI-3 Docs / Quickstart
   - https://docs.arcprize.org/
   - Use the documentation index if available: https://docs.arcprize.org/llms.txt

2. ARC-AGI-3 Games Documentation
   - https://docs.arcprize.org/games
   - Key idea: games are turn-based systems where agents interact with 2D grid environments through a standardized action interface.

3. ARC-AGI-3 Main Page / Public Game Set
   - https://arcprize.org/arc-agi/3
   - Key idea: agents must explore novel environments, infer goals, build adaptable world models, and act efficiently.

4. ARC-AGI-3 Technical Report
   - https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf
   - Also searchable as arXiv: https://arxiv.org/html/2603.24621v2

5. Graph-Based Exploration for ARC-AGI-3 Interactive Reasoning Tasks
   - https://arxiv.org/abs/2512.24156
   - Important because it supports the idea of graph-based state/action exploration and explicit hypothesis tracking.

6. PPO Original Paper
   - https://arxiv.org/abs/1707.06347
   - Implement PPO with clipped policy objective, value loss, entropy bonus, GAE advantage estimation, and checkpointing.

7. Graph Neural Network / Message Passing Background
   - Comprehensive Survey on GNNs: https://arxiv.org/pdf/1901.00596
   - Message passing foundation: https://openreview.net/forum?id=Bc8GiEZkTe5

---

## 1. High-Level System Summary

Implement this pipeline:

```text
raw observation/frame
    -> visual parser / segmentation
    -> object tracker
    -> graph builder
    -> GNN affordance encoder
    -> belief/world-model updater
    -> candidate subgoal generator
    -> planner/search module
    -> PPO policy/value module scores high-level choices
    -> action executor converts chosen subgoal into one low-level environment action
    -> env.step(action)
    -> repeat until done or budget exhausted
```

Important: PPO should **not** directly learn from raw pixels to low-level actions in v1. PPO should operate on compact graph/belief embeddings and select high-level strategies or candidate subgoals. The planner remains responsible for concrete pathfinding and safety checks.

---

## 2. Repository Layout

Codex should create this structure:

```text
arc_agi3_gnn_ppo_agent/
    README.md
    pyproject.toml
    requirements.txt
    configs/
        default.yaml
        hardware_3080_10gb.yaml
        debug_cpu.yaml
        train_gnn.yaml
        train_ppo.yaml
    docs/
        architecture.md
        training.md
        evaluation.md
        data_format.md
        troubleshooting.md
    scripts/
        train_gnn.py
        train_ppo.py
        evaluate_agent.py
        collect_rollouts.py
        inspect_graphs.py
        replay_episode.py
    src/
        arc_agent/
            __init__.py
            envs/
                __init__.py
                arc_env_adapter.py
                mock_grid_env.py
            perception/
                __init__.py
                segmenter.py
                grid_extractor.py
                object_tracker.py
                object_types.py
            graph/
                __init__.py
                graph_builder.py
                graph_features.py
                graph_data.py
                graph_visualizer.py
            models/
                __init__.py
                gnn_affordance.py
                ppo_actor_critic.py
                encoders.py
            belief/
                __init__.py
                belief_state.py
                belief_update.py
                causal_graph.py
            planning/
                __init__.py
                candidates.py
                planner.py
                astar.py
                bfs.py
                risk.py
                information_gain.py
            rl/
                __init__.py
                rollout_buffer.py
                ppo_trainer.py
                reward_shaping.py
            execution/
                __init__.py
                action_executor.py
                action_space.py
            training/
                __init__.py
                gnn_dataset.py
                replay_dataset.py
                checkpointing.py
                logging_utils.py
                metrics.py
            utils/
                __init__.py
                config.py
                seed.py
                device.py
                io.py
                timing.py
    tests/
        test_segmenter.py
        test_tracker.py
        test_graph_builder.py
        test_belief_update.py
        test_planner.py
        test_ppo_shapes.py
```

Use type hints and dataclasses for all major data structures.

---

## 3. Configuration Requirements

All hyperparameters must be YAML-driven. No important hyperparameter should be hardcoded inside model/training code.

Use `omegaconf` or plain `yaml.safe_load`. Prefer `omegaconf` if available.

Example `configs/hardware_3080_10gb.yaml`:

```yaml
project:
  name: arc_agi3_gnn_ppo_agent
  seed: 42
  output_dir: runs
  run_name: gnn_ppo_v1_3080

hardware:
  device: cuda
  mixed_precision: true
  num_workers: 4
  pin_memory: true
  compile_model: false
  max_vram_gb: 9.2
  ssd_cache_dir: /mnt/ssd/arc_cache
  hdd_archive_dir: /mnt/hdd/arc_runs

storage:
  save_replays: true
  replay_compression: true
  max_replay_gb: 60
  checkpoint_keep_last: 3
  checkpoint_keep_best: 5

perception:
  grid_mode: true
  connected_components: true
  min_component_area: 1
  merge_same_color_adjacent: true
  background_color_mode: auto

tracking:
  method: hungarian
  max_match_distance: 4
  iou_weight: 0.4
  color_weight: 0.3
  position_weight: 0.3

features:
  normalize_positions: true
  include_color_onehot: true
  max_colors: 16
  include_motion_delta: true
  include_action_onehot: true
  include_temporal_flags: true

edge_builder:
  spatial_radius: 6
  include_touching_edges: true
  include_same_color_edges: true
  include_same_row_col_edges: true
  include_temporal_edges: true
  include_candidate_causal_edges: true

model:
  gnn:
    type: gatv2
    node_in_dim: auto
    edge_in_dim: auto
    hidden_dim: 128
    layers: 3
    heads: 4
    dropout: 0.1
    global_pool: attention
    output_affordances:
      - blocking
      - movable
      - collectible
      - hazardous
      - goal_related
      - trigger
      - controllable
      - ui_or_status
      - unknown_interactable
  ppo:
    graph_embedding_dim: 128
    belief_feature_dim: auto
    hidden_dim: 256
    recurrent: false
    action_head: categorical

planner:
  low_level_search: astar
  max_search_nodes: 20000
  replan_every_step: true
  execute_plan_steps_before_replan: 1
  risk_threshold: 0.65
  unknown_test_bonus: 2.0
  goal_bonus: 10.0
  step_cost: 0.02

ppo_training:
  total_env_steps: 1000000
  rollout_steps: 512
  num_envs: 8
  minibatch_size: 512
  update_epochs: 4
  gamma: 0.99
  gae_lambda: 0.95
  clip_coef: 0.2
  ent_coef: 0.02
  vf_coef: 0.5
  max_grad_norm: 0.5
  learning_rate: 0.00025
  target_kl: 0.03

logging:
  log_interval_steps: 1000
  eval_interval_steps: 25000
  checkpoint_interval_steps: 50000
  use_tensorboard: true
  use_csv_log: true
  console_level: info
```

For the user's hardware, start with:

- PPO `num_envs`: 8
- rollout steps: 512
- minibatch size: 512
- GNN hidden dim: 128
- GNN layers: 3
- GAT heads: 4
- mixed precision: true
- keep replay buffers bounded to avoid filling the 100GB SSD
- archive old logs/checkpoints to the 1TB HDD

Avoid large CNN backbones in v1. The first version should be object/graph-based.

---

## 4. Data Structures

Define these core dataclasses.

### 4.1 ObjectComponent

```python
@dataclass
class ObjectComponent:
    id: int
    color_id: int | None
    pixels: np.ndarray
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    area: int
    frame_index: int
    track_id: int | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
```

### 4.2 GraphState

```python
@dataclass
class GraphState:
    nodes: list[ObjectComponent]
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    graph_features: torch.Tensor | None
    action_context: torch.Tensor | None
    frame_index: int
```

### 4.3 BeliefState

```python
@dataclass
class BeliefState:
    object_beliefs: dict[int, dict[str, float]]
    causal_edges: dict[tuple[int, int], dict[str, float]]
    reachable_cells: set[tuple[int, int]]
    blocked_cells: set[tuple[int, int]]
    hazards: set[int]
    goals: set[int]
    unknown_objects: set[int]
    visited_states: set[str]
    last_action: int | None
    step_count: int
```

### 4.4 CandidateSubgoal

```python
@dataclass
class CandidateSubgoal:
    name: str
    target_object_id: int | None
    target_position: tuple[int, int] | None
    expected_information_gain: float
    expected_goal_progress: float
    expected_risk: float
    planner_path: list[int]
    features: np.ndarray
```

PPO should choose among `CandidateSubgoal`s, not raw environment movement actions in v1.

---

## 5. Component Implementation Details

## 5.1 Environment Adapter

Create `src/arc_agent/envs/arc_env_adapter.py`.

Responsibilities:

- Wrap official ARC-AGI-3 SDK or a compatible mock environment.
- Expose a stable interface:

```python
class EnvAdapter(Protocol):
    def reset(self, game_id: str | None = None, level_id: str | None = None) -> np.ndarray: ...
    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]: ...
    @property
    def action_space_n(self) -> int: ...
```

Also implement `mock_grid_env.py` for unit tests and local training before official integration.

The mock environment should include:

- wall/blocking mechanics
- collectible mechanics
- pushable mechanics
- trigger/door mechanics
- hazard mechanics
- goal mechanics

This lets Codex test the pipeline without depending on the official benchmark immediately.

---

## 5.2 Segmentation

Create `src/arc_agent/perception/segmenter.py`.

V1 methods:

1. Convert frame to grid if ARC observation is grid-like.
2. Detect background color automatically.
3. Run connected components by color.
4. Optionally merge adjacent cells of same color.
5. Return `ObjectComponent` list.

Do not use YOLO/CNN object detection in v1.

Output must be deterministic for the same frame.

Add unit tests:

- one object
- multiple disconnected same-color objects
- background ignored
- status/UI region retained but marked separately later

---

## 5.3 Object Tracking

Create `src/arc_agent/perception/object_tracker.py`.

Implement Hungarian matching using a cost matrix:

```text
cost = position_weight * normalized_distance
     + color_weight * color_mismatch
     + iou_weight * (1 - IoU)
```

Fallback to nearest-neighbor if scipy is not installed.

Outputs:

- stable `track_id`
- disappeared objects
- newly appeared objects
- moved objects

Tracking is required for state differencing and causal belief updates.

---

## 5.4 Graph Builder

Create `src/arc_agent/graph/graph_builder.py`.

Inputs:

- current tracked objects
- previous tracked objects
- previous action
- optional belief state

Outputs:

- PyTorch Geometric-style graph object or project-defined `GraphState`

Node features should include:

```text
normalized x/y centroid
width/height
area
color one-hot
motion dx/dy
changed flag
disappeared/new flag
distance to controllable-agent candidates
touching-agent flag
affordance belief probabilities from previous step
action one-hot from previous action
```

Edge types/features should include:

```text
dx, dy
distance
touching flag
same color flag
same row flag
same column flag
temporal same-track flag
candidate causal relation probability
reachable relation flag
```

V1 can represent relation type as edge attributes. Later can add heterogeneous graphs.

---

## 5.5 GNN Affordance Model

Create `src/arc_agent/models/gnn_affordance.py`.

Recommended v1 model:

- GATv2Conv or GraphSAGE if PyTorch Geometric is available.
- If PyTorch Geometric is difficult, implement a small custom message-passing layer with `edge_index` and `scatter_add`.
- Output per-node affordance logits.
- Output a graph-level embedding for PPO.

Per-node affordance outputs:

```text
blocking
movable
collectible
hazardous
goal_related
trigger
controllable
ui_or_status
unknown_interactable
```

Model forward output:

```python
@dataclass
class GNNOutput:
    node_embeddings: torch.Tensor
    graph_embedding: torch.Tensor
    affordance_logits: torch.Tensor
    affordance_probs: torch.Tensor
```

Loss for supervised/pretraining if labels exist:

```text
multi-label BCEWithLogitsLoss for affordances
optional contrastive loss for similar mechanics
optional transition prediction auxiliary loss
```

For v1, synthetic/mock environments can generate labels automatically.

---

## 5.6 Belief / World Model Updater

Create:

- `src/arc_agent/belief/belief_state.py`
- `src/arc_agent/belief/belief_update.py`
- `src/arc_agent/belief/causal_graph.py`

This is a critical component. It performs online adaptation during evaluation without changing neural weights.

Implement deterministic/probabilistic update rules:

```text
if attempted movement but agent position unchanged:
    increase blocking probability for object in attempted direction

if object disappeared after contact:
    increase collectible probability

if object moved after contact:
    increase movable probability

if remote object changed after interacting with object A:
    add/increase causal edge A -> remote object

if reward increased or done=True after contact/reaching object:
    increase goal_related probability

if reset/death-like transition occurred:
    increase hazardous probability
```

Use clamped probabilities in `[0.01, 0.99]`.

Recommended update formula:

```python
new_p = clamp(old_p + alpha * evidence_strength * (1 - old_p), 0.01, 0.99)
```

For negative evidence:

```python
new_p = clamp(old_p * (1 - beta * evidence_strength), 0.01, 0.99)
```

Keep update coefficients in YAML.

---

## 5.7 Candidate Subgoal Generator

Create `src/arc_agent/planning/candidates.py`.

Generate generic candidates, not game-specific candidates.

Candidate types:

```text
REACH_KNOWN_GOAL
TEST_UNKNOWN_OBJECT
TEST_BLOCKING_OBJECT
COLLECT_COLLECTIBLE_LIKE_OBJECT
TEST_PUSHABLE_OBJECT
ACTIVATE_TRIGGER_LIKE_OBJECT
RETRY_PREVIOUSLY_BLOCKED_PATH
EXPLORE_FRONTIER
AVOID_HAZARD_ROUTE
```

Each candidate should include:

- target object or target cell
- planned low-level path if available
- expected information gain
- expected goal progress
- expected risk
- candidate feature vector for PPO scoring

Do not hardcode game names such as `ls20` in planner logic.

---

## 5.8 Planner/Search Module

Create:

- `src/arc_agent/planning/planner.py`
- `src/arc_agent/planning/bfs.py`
- `src/arc_agent/planning/astar.py`
- `src/arc_agent/planning/risk.py`
- `src/arc_agent/planning/information_gain.py`

Planner responsibilities:

1. Convert belief state into a navigable occupancy/risk map.
2. For each candidate subgoal, compute a low-level action path.
3. Score candidates using symbolic heuristics.
4. Pass candidate features to PPO for learned scoring.
5. Combine symbolic and PPO scores.
6. Return one low-level action to execute.

Use A* for movement paths when a grid is available.

Combined score:

```text
score = planner_goal_weight * expected_goal_progress
      + info_gain_weight * expected_information_gain
      + ppo_weight * ppo_candidate_score
      - risk_weight * expected_risk
      - step_cost_weight * path_length
```

All weights must be YAML-configurable.

Important: execute only one low-level action before replanning in v1.

---

## 5.9 PPO Module

Create:

- `src/arc_agent/models/ppo_actor_critic.py`
- `src/arc_agent/rl/rollout_buffer.py`
- `src/arc_agent/rl/ppo_trainer.py`
- `src/arc_agent/rl/reward_shaping.py`

Use PPO to select among high-level candidate subgoals.

### PPO Input

The policy input should combine:

```text
graph embedding from GNN
belief summary vector
candidate feature vectors
recent action/history summary
```

Two implementation options:

### Option A: Fixed High-Level Action Space

PPO outputs one of fixed strategies:

```text
0: REACH_KNOWN_GOAL
1: TEST_NEAREST_UNKNOWN
2: TEST_HIGHEST_UNCERTAINTY
3: COLLECT_PROBABLE_COLLECTIBLE
4: TEST_PROBABLE_PUSHABLE
5: ACTIVATE_PROBABLE_TRIGGER
6: RETRY_BLOCKED_PATH
7: EXPLORE_FRONTIER
```

This is easiest for v1.

### Option B: Candidate Scoring Policy

PPO scores a variable number of candidate subgoals using masking.

For v1, implement Option A first. Add Option B later.

### PPO Reward

Use shaped reward carefully:

```text
+ external environment reward
+ success bonus
+ information gain bonus
+ new causal relation bonus
+ new object affordance confidence bonus
+ reaching useful subgoal bonus
- step penalty
- repeated-state penalty
- hazard/death penalty
```

Keep all reward weights in YAML.

### PPO Checkpoint Metrics

Save best checkpoints by:

```text
best_eval_success_rate
best_eval_mean_return
best_eval_human_efficiency_proxy / action_efficiency
best_eval_levels_solved
```

Always save:

- `latest.pt`
- `best_success.pt`
- `best_return.pt`
- periodic checkpoints every N steps

---

## 6. Training Strategy

## Stage 0: Mock Environment Validation

Before training anything, make the pipeline work in mock environments.

Goals:

- segmentation produces objects
- graph builder produces valid tensors
- belief updater correctly infers blocking/collectible/movable/trigger/hazard from controlled examples
- planner solves simple mock tasks without GNN/PPO

Run:

```bash
python scripts/evaluate_agent.py --config configs/debug_cpu.yaml --env mock --episodes 20
```

Expected:

- no crashes
- text logs every episode
- saved episode replay JSON

---

## Stage 1: Train GNN Affordance Model on Synthetic/Mock Data

Generate synthetic episodes from mock grid environments with known labels.

Run:

```bash
python scripts/collect_rollouts.py --config configs/train_gnn.yaml --env mock --episodes 5000
python scripts/train_gnn.py --config configs/train_gnn.yaml
```

GNN training logs should print and save:

```text
train/loss
val/loss
val/affordance_f1_blocking
val/affordance_f1_movable
val/affordance_f1_collectible
val/affordance_f1_hazardous
val/affordance_f1_goal_related
learning_rate
gpu_memory_allocated_mb
epoch_time_sec
```

Save checkpoints:

```text
runs/<run_name>/checkpoints/gnn_latest.pt
runs/<run_name>/checkpoints/gnn_best_val_f1.pt
runs/<run_name>/checkpoints/gnn_epoch_XXXX.pt
```

---

## Stage 2: Rule-Based Planner Baseline with Frozen GNN

Use the trained GNN for affordance priors, but no PPO yet.

Run:

```bash
python scripts/evaluate_agent.py --config configs/default.yaml --mode heuristic --episodes 100
```

Metrics:

```text
success_rate
mean_return
mean_steps_to_success
mean_unique_states_visited
mean_repeated_state_rate
mean_information_gain
```

This establishes a baseline before PPO.

---

## Stage 3: Train PPO High-Level Subgoal Selector

Freeze the GNN initially.

PPO controls high-level strategy selection. Planner still executes low-level movement.

Run:

```bash
python scripts/train_ppo.py --config configs/train_ppo.yaml --pretrained-gnn runs/<run>/checkpoints/gnn_best_val_f1.pt
```

PPO logs must include:

```text
step
episodes
mean_return
success_rate
mean_episode_length
policy_loss
value_loss
entropy
approx_kl
clip_fraction
explained_variance
learning_rate
grad_norm
reward_external
reward_info_gain
reward_success
reward_step_penalty
gpu_memory_allocated_mb
steps_per_second
```

Write logs to:

```text
runs/<run_name>/logs/train.log
runs/<run_name>/logs/metrics.csv
runs/<run_name>/tensorboard/
```

Console should print an important summary every `logging.log_interval_steps`.

---

## Stage 4: Evaluate on Public ARC-AGI-3 Games

After mock training, integrate official ARC-AGI-3 environment adapter.

Run:

```bash
python scripts/evaluate_agent.py --config configs/hardware_3080_10gb.yaml --env arcagi3 --games public --episodes-per-game 5
```

Save per-game reports:

```text
runs/<run_name>/eval/public_summary.json
runs/<run_name>/eval/per_game_metrics.csv
runs/<run_name>/eval/replays/<game>/<episode>.json.gz
```

Do not special-case by game ID. If a game needs investigation, add a generic mechanic template only if it can apply to more than one game.

---

## 7. Online Evaluation Loop Pseudocode

Codex should implement the final agent around this loop:

```python
def act(self, obs: np.ndarray) -> int:
    # 1. Segment current observation
    objects = self.segmenter.segment(obs)

    # 2. Track objects across time
    tracked = self.tracker.update(objects)

    # 3. Build current graph
    graph = self.graph_builder.build(
        tracked_objects=tracked,
        previous_action=self.last_action,
        previous_graph=self.previous_graph,
        belief=self.belief,
    )

    # 4. Run frozen GNN inference
    with torch.no_grad():
        gnn_out = self.gnn(graph.to(self.device))

    # 5. Compare previous and current graph, update belief
    delta = self.delta_analyzer.compare(self.previous_graph, graph, self.last_action)
    self.belief_updater.update(
        belief=self.belief,
        graph=graph,
        gnn_output=gnn_out,
        delta=delta,
        last_reward=self.last_reward,
        last_done=self.last_done,
    )

    # 6. Generate candidate subgoals
    candidates = self.candidate_generator.generate(self.belief, graph, gnn_out)

    # 7. Planner computes feasible paths and symbolic scores
    planned_candidates = self.planner.plan_candidates(self.belief, graph, candidates)

    # 8. PPO scores high-level strategy/candidate
    with torch.no_grad():
        ppo_scores = self.ppo_policy.score(self.belief, gnn_out.graph_embedding, planned_candidates)

    # 9. Select best candidate
    chosen = self.selector.combine_scores(planned_candidates, ppo_scores)

    # 10. Execute one low-level action from chosen plan
    action = self.executor.next_action(chosen)

    # 11. Save state for next step
    self.previous_graph = graph
    self.last_action = action
    return action
```

During official evaluation, do not run optimizer steps unless explicitly enabled for experiments. Default evaluation mode uses frozen GNN and PPO weights.

---

## 8. Logging Requirements

Implement a shared logger in `src/arc_agent/training/logging_utils.py`.

Requirements:

1. Print important metrics to console.
2. Write the same metrics to a log file.
3. Write scalar metrics to CSV.
4. Optionally write TensorBoard events.
5. Include timestamps and run names.
6. Include GPU memory info when CUDA is available.

Console example:

```text
[2026-05-09 20:14:02][train_ppo][step=128000]
return=4.82 success=0.31 ep_len=67.4 policy_loss=-0.018 value_loss=0.42 entropy=1.71 kl=0.012 clip=0.09 fps=812 gpu_mem=6.8GB
```

Also log important non-scalar events:

```text
checkpoint saved
best checkpoint updated
NaN detected
OOM recovered
new best success rate
```

---

## 9. Checkpointing Requirements

Implement `src/arc_agent/training/checkpointing.py`.

Each checkpoint should include:

```python
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "config": full_config_dict,
    "global_step": int,
    "epoch": int | None,
    "metrics": dict,
    "rng_state": dict,
    "git_commit": str | None,
}
```

For PPO checkpoints include:

```text
actor_critic state
optimizer state
normalization statistics
global step
best metrics
```

Checkpoint policy:

```text
always save latest
save periodic every checkpoint_interval_steps
save best_success when eval success improves
save best_return when eval return improves
keep last N periodic checkpoints
never delete best checkpoints unless user explicitly asks
```

If storage is limited, archive old periodic checkpoints to HDD and keep best/latest on SSD.

---

## 10. Hardware-Aware Defaults

For i9-10850K + RTX 3080 10GB + 32GB RAM:

### GPU/Training

```yaml
mixed_precision: true
gnn.hidden_dim: 128
gnn.layers: 3
gnn.heads: 4
ppo.hidden_dim: 256
ppo.rollout_steps: 512
ppo.num_envs: 8
ppo.minibatch_size: 512
ppo.update_epochs: 4
```

### CPU/RAM

```yaml
num_workers: 4
prefetch_factor: 2
max_replay_cache_ram_gb: 8
```

### Storage

Use SSD for active run:

```text
runs/current
cache/current_rollouts
active checkpoints
```

Use HDD for archive:

```text
old replays
old periodic checkpoints
large logs
```

Limit saved replays by config. Compress JSON replays using gzip.

---

## 11. Documentation Codex Must Generate

Codex must generate these docs:

### `docs/architecture.md`

Explain:

- each component
- data flow
- why GNN predicts affordances, not only object categories
- why PPO selects high-level strategy, not raw pixel actions
- how online belief updates differ from neural training

### `docs/training.md`

Explain:

- how to train GNN
- how to train PPO
- how to resume from checkpoint
- how to evaluate
- what metrics mean
- expected hardware usage

### `docs/data_format.md`

Explain:

- replay JSON format
- graph tensor format
- checkpoint format
- CSV metric format

### `docs/evaluation.md`

Explain:

- how to run public ARC-AGI-3 games
- how to run mock environments
- how to inspect failed episodes
- how to verify no game-specific hardcoding exists

### `README.md`

Must include:

```text
installation
quickstart
train GNN command
train PPO command
evaluate command
resume from checkpoint command
where logs/checkpoints are stored
```

---

## 12. Minimal Commands to Support

Implement these exact command patterns:

```bash
# install
pip install -e .

# run unit tests
pytest tests -q

# collect synthetic/mock rollout data
python scripts/collect_rollouts.py --config configs/train_gnn.yaml --env mock --episodes 5000

# train GNN
python scripts/train_gnn.py --config configs/train_gnn.yaml

# train PPO with pretrained GNN
python scripts/train_ppo.py --config configs/train_ppo.yaml --pretrained-gnn runs/<run>/checkpoints/gnn_best_val_f1.pt

# resume PPO
python scripts/train_ppo.py --config configs/train_ppo.yaml --resume runs/<run>/checkpoints/ppo_latest.pt

# evaluate
python scripts/evaluate_agent.py --config configs/hardware_3080_10gb.yaml --checkpoint runs/<run>/checkpoints/ppo_best_success.pt

# replay a saved episode
python scripts/replay_episode.py --replay runs/<run>/replays/example.json.gz

# inspect graph construction visually/textually
python scripts/inspect_graphs.py --replay runs/<run>/replays/example.json.gz --step 10
```

---

## 13. Testing Requirements

Write unit tests before full training.

Required tests:

1. `test_segmenter.py`
   - connected components return expected object count
   - background ignored
   - components have valid bbox/centroid

2. `test_tracker.py`
   - stable ID across small movement
   - detects disappeared object
   - detects new object

3. `test_graph_builder.py`
   - edge_index shape correct
   - edge_attr shape correct
   - touching edge exists when objects adjacent
   - same-color edge exists when enabled

4. `test_belief_update.py`
   - failed movement increases blocking belief
   - disappearance increases collectible belief
   - remote change creates causal relation

5. `test_planner.py`
   - A* reaches target around obstacle
   - avoids high-risk hazard cells
   - returns None if target unreachable

6. `test_ppo_shapes.py`
   - actor output shape correct
   - value output shape correct
   - action mask works if implemented

---

## 14. Implementation Order for Codex

Do not implement everything randomly. Follow this order:

1. Project skeleton, config loader, logger, checkpoint manager.
2. Mock grid environment.
3. Segmenter and object tracker.
4. Graph builder and graph visualization/debug dump.
5. Belief state and belief updater.
6. BFS/A* planner and candidate generator.
7. Heuristic-only agent that works on mock environment.
8. GNN model and supervised training on synthetic mock labels.
9. PPO actor-critic and rollout buffer.
10. PPO training using high-level action space.
11. Evaluation scripts and replay inspection.
12. Official ARC-AGI-3 adapter integration.
13. Documentation pass.
14. Unit test cleanup and hyperparameter config cleanup.

At each stage, keep the project runnable.

---

## 15. First-Version Success Criteria

The first version is successful if:

1. It can run mock environments end-to-end.
2. It logs to console and file simultaneously.
3. It saves latest and best checkpoints.
4. It can resume training from checkpoint.
5. GNN training improves affordance F1 on synthetic validation data.
6. PPO training improves success rate or action efficiency over heuristic baseline in mock environments.
7. Evaluation produces replay files and metrics CSV.
8. All important hyperparameters are editable from YAML.
9. No game-specific hardcoded solver logic exists.
10. Documentation explains training and evaluation commands clearly.

---

## 16. Important Design Warnings

1. Do not make PPO output raw movement from pixels in v1.
2. Do not assume exactly one controllable agent.
3. Do not assume a status bar always exists.
4. Do not assume colors have fixed meanings across games.
5. Do not hardcode public game IDs.
6. Do not train neural weights during official evaluation by default.
7. Do not save unlimited replays/checkpoints to the 100GB SSD.
8. Do not hide hyperparameters inside Python files.
9. Do not skip unit tests for segmentation/tracking/graph construction.
10. Do not trust the GNN blindly; belief updates and environment feedback are the source of task-specific truth.

---

## 17. Final Intended Runtime Behavior

At evaluation time, the agent should work like this:

```text
1. Receive private/hidden game observation.
2. Segment objects.
3. Track object identities.
4. Build object graph.
5. Run frozen GNN to estimate affordances and graph embedding.
6. Compare new graph with previous graph to detect effects of last action.
7. Update belief state and causal graph.
8. Generate candidate subgoals.
9. Planner computes safe feasible paths.
10. PPO selects/ranks high-level strategy.
11. Executor emits one valid low-level action.
12. Environment returns next observation.
13. Repeat until success/failure/budget limit.
```

The submitted/evaluated behavior is only the sequence of environment actions. All GNN/PPO/planner logic is internal machinery to generate better actions.
