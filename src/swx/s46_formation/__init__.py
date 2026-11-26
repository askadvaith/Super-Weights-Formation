"""
Super-weight formation, cross-family sweep, and controlled pretraining modules.

    g1_activation_family.py   Cross-family pretrained model zoo sweep (Table 1)
    train_activation_twins.py Parameter-matched 124M Transformer pretraining and interventions
    plot_trajectories.py      Telemetry aggregation, summaries, and trajectory plotting
    _common.py                Shared sweep and checkpoint-walking helpers
"""

from __future__ import annotations
