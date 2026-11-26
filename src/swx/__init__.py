"""
swx -- Super Weights analysis, detection, and controlled pretraining package.

LAYOUT
------
    core/               shared detection, checkpoint slicing, modeling, metrics, and directory
    s46_formation/      pretrained zoo sweep (g1), controlled training twins, and trajectory plotting
    registry.py         CLI experiment registry
    cli.py              argument parsing and dispatch
    chat.py             interactive REPL with super-weight ablation toggle

TWO HARD CONSTRAINTS
--------------------
NO QUANTISATION. `core.modeling.load_model` refuses int8/int4/nf4 outright and
requires an explicit flag for fp16. Quantisation directly perturbs super outliers.

FLOAT64 FOR EVERY METRIC. `core.metrics` promotes to float64 regardless of the
model's dtype to avoid bf16 accumulation error.
"""

__version__ = "0.1.0"
