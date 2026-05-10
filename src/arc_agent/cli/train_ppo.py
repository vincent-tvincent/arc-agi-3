#!/usr/bin/env python3
"""Train PPO high-level strategy selection on the mock environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from arc_agent.belief.belief_state import BeliefState
from arc_agent.belief.belief_update import BeliefUpdater, compare_graphs
from arc_agent.envs.mock_grid_env import COLOR_LABELS, MockGridEnv
from arc_agent.execution.action_executor import ActionExecutor
from arc_agent.graph.graph_builder import GraphBuilder
from arc_agent.graph.graph_data import GraphState
from arc_agent.graph.graph_features import AFFORDANCE_NAMES
from arc_agent.models.gnn_affordance import GNNAffordanceModel
from arc_agent.models.ppo_actor_critic import PPOActorCritic
from arc_agent.perception.object_tracker import ObjectTracker
from arc_agent.perception.segmenter import GridSegmenter
from arc_agent.planning.candidates import CandidateType, CandidateGenerator
from arc_agent.planning.planner import Planner
from arc_agent.rl.ppo_trainer import PPOTrainer
from arc_agent.rl.reward_shaping import RewardShaper
from arc_agent.rl.rollout_buffer import RolloutBuffer
from arc_agent.training.checkpointing import CheckpointManager, build_checkpoint
from arc_agent.training.logging_utils import RunLogger
from arc_agent.utils.config import load_config, run_dir
from arc_agent.utils.device import select_device, torch_available
from arc_agent.utils.seed import set_global_seed


STRATEGY_NAMES = [item.name for item in CandidateType][:8]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_ppo.yaml")
    parser.add_argument("--pretrained-gnn", default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    if not torch_available():
        raise SystemExit("PyTorch is required for PPO training. Install with `pip install -e .[train]`.")
    import torch

    config = load_config(args.config)
    set_global_seed(int(config.get("project", {}).get("seed", 42)))
    device = select_device(config.get("hardware", {}).get("device", "auto"))
    root = run_dir(config)
    logger = RunLogger(config.get("project", {}).get("run_name", "train_ppo"), root / "logs", True, config.get("logging", {}).get("use_tensorboard", False))
    manager = CheckpointManager(root / "checkpoints", int(config.get("storage", {}).get("checkpoint_keep_last", 3)))

    env = MockGridEnv(config.get("mock_env", {}))
    obs = env.reset()
    segmenter = GridSegmenter(config.get("perception", {}))
    tracker = ObjectTracker(config.get("tracking", {}))
    builder = GraphBuilder(config.get("edge_builder", {}), config.get("features", {}))
    initial_graph = build_graph(obs, 0, None, segmenter, tracker, builder, BeliefState(), config)
    model_cfg = config.get("model", {})
    ppo_cfg = model_cfg.get("ppo", {})
    gnn_cfg = model_cfg.get("gnn", {})
    gnn = GNNAffordanceModel(
        initial_graph.node_feature_dim,
        initial_graph.edge_feature_dim,
        hidden_dim=int(gnn_cfg.get("hidden_dim", 128)),
        layers=int(gnn_cfg.get("layers", 3)),
        num_affordances=len(AFFORDANCE_NAMES),
        dropout=float(gnn_cfg.get("dropout", 0.1)),
    ).to(device)
    if args.pretrained_gnn:
        checkpoint = manager.load(args.pretrained_gnn, map_location=device)
        gnn.load_state_dict(checkpoint["model_state_dict"])
    gnn.eval()

    actor = PPOActorCritic(
        graph_embedding_dim=int(ppo_cfg.get("graph_embedding_dim", gnn_cfg.get("hidden_dim", 128))),
        belief_feature_dim=int(ppo_cfg.get("belief_feature_dim", 32)),
        hidden_dim=int(ppo_cfg.get("hidden_dim", 256)),
        num_actions=int(ppo_cfg.get("high_level_actions", 8)),
    ).to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=float(config.get("ppo_training", {}).get("learning_rate", 0.00025)))
    trainer = PPOTrainer(actor, optimizer, config.get("ppo_training", {}))
    if args.resume:
        checkpoint = manager.load(args.resume, map_location=device)
        actor.load_state_dict(checkpoint["actor_critic_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    rollout_steps = int(config.get("ppo_training", {}).get("rollout_steps", 512))
    total_steps = int(config.get("ppo_training", {}).get("total_env_steps", 100000))
    gamma = float(config.get("ppo_training", {}).get("gamma", 0.99))
    gae_lambda = float(config.get("ppo_training", {}).get("gae_lambda", 0.95))
    reward_shaper = RewardShaper(config.get("reward", {}))
    buffer = RolloutBuffer()
    global_step = 0
    episodes = 0
    episode_return = 0.0
    successes: list[bool] = []
    returns: list[float] = []
    obs = env.reset()
    tracker.reset()
    previous_graph: GraphState | None = None
    belief = BeliefState()
    updater = BeliefUpdater(config.get("belief", {}))
    planner = Planner(config.get("planner", {}))
    generator = CandidateGenerator(config.get("candidate_generator", {}))
    executor = ActionExecutor()
    last_action: int | None = None

    while global_step < total_steps:
        graph = build_graph(obs, global_step, last_action, segmenter, tracker, builder, belief, config)
        seed_mock_beliefs(graph, belief)
        with torch.no_grad():
            gnn_out = gnn(graph.to_torch(device))
            graph_embedding = gnn_out.graph_embedding
            belief_features = torch.as_tensor(belief.summary_vector(int(ppo_cfg.get("belief_feature_dim", 32))), dtype=torch.float32, device=device)
            high_action, log_prob, value = actor.act(graph_embedding, belief_features)
        strategy = STRATEGY_NAMES[int(high_action.item()) % len(STRATEGY_NAMES)]
        candidates = generator.generate(belief, graph, gnn_out)
        planned = planner.plan_candidates(belief, graph, candidates, {strategy: 1.0}, frame_shape=obs.shape)
        low_action = executor.next_action(planner.choose(planned))
        next_obs, reward, done, info = env.step(low_action)
        shaped = reward_shaper.shape(reward, done, info_gain=max((c.expected_information_gain for c in planned), default=0.0), hazard=info.get("event") == "hazard")
        buffer.add(graph_embedding, belief_features, int(high_action.item()), log_prob, shaped.total, done, value)
        delta = compare_graphs(previous_graph, graph, low_action, reward, done)
        updater.update(belief, graph, gnn_out, delta, reward, done)
        previous_graph = graph
        last_action = low_action
        obs = next_obs
        global_step += 1
        episode_return += reward

        if done:
            successes.append(bool(reward > 0))
            returns.append(episode_return)
            episodes += 1
            obs = env.reset()
            tracker.reset()
            previous_graph = None
            belief = BeliefState()
            last_action = None
            episode_return = 0.0

        if len(buffer.rewards) >= rollout_steps:
            with torch.no_grad():
                last_graph = build_graph(obs, global_step, last_action, segmenter, tracker, builder, belief, config)
                last_embedding = gnn(last_graph.to_torch(device)).graph_embedding
                last_belief = torch.as_tensor(belief.summary_vector(int(ppo_cfg.get("belief_feature_dim", 32))), dtype=torch.float32, device=device)
                last_value = actor(last_embedding, last_belief).value.reshape(())
            buffer.compute_returns_and_advantages(last_value, gamma, gae_lambda)
            train_metrics = trainer.update(buffer)
            metrics = {
                **train_metrics,
                "episodes": episodes,
                "mean_return": sum(returns[-20:]) / len(returns[-20:]) if returns else 0.0,
                "success_rate": sum(successes[-20:]) / len(successes[-20:]) if successes else 0.0,
            }
            logger.log_metrics("train_ppo", global_step, metrics)
            checkpoint = build_checkpoint(
                config,
                global_step=global_step,
                metrics=metrics,
                actor_critic_state_dict=actor.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
            )
            manager.save_latest(checkpoint, "ppo")
            buffer.clear()
    logger.close()


def build_graph(obs, frame_index, last_action, segmenter, tracker, builder, belief, config):
    objects = segmenter.segment(obs, frame_index=frame_index)
    for obj in objects:
        labels = COLOR_LABELS.get(int(obj.color_id or -1), set())
        if "controllable" in labels:
            obj.attributes["label"] = "agent"
    tracked = tracker.update(objects)
    return builder.build(tracked.objects, previous_action=last_action, belief=belief, frame_shape=obs.shape)


def seed_mock_beliefs(graph: GraphState, belief: BeliefState) -> None:
    for node in graph.nodes:
        object_id = node.track_id or node.id
        object_belief = belief.ensure_object(object_id)
        labels = COLOR_LABELS.get(int(node.color_id or -1), set())
        if "controllable" in labels:
            node.attributes["label"] = "agent"
            object_belief["controllable"] = 0.95
        for label in labels:
            if label in AFFORDANCE_NAMES:
                object_belief[label] = max(object_belief.get(label, 0.1), 0.9)
        if "blocking" in labels:
            object_belief["blocking"] = 0.9
        if "goal_related" in labels:
            belief.goals.add(object_id)
        if "hazardous" in labels:
            belief.hazards.add(object_id)


if __name__ == "__main__":
    main()
