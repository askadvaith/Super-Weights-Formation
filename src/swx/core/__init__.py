"""
Shared infrastructure for detection, modeling, checkpoint slicing, and metrics.

    args.py           ExpArgs -- shared configuration dataclass
    sw_directory.py   ModelSpec registry, tabulated super-weight coordinates, and Pythia seed families
    modeling.py       model loading, architecture-agnostic module resolution, interventions
    detection.py      activation-spike super-weight detection and dominance verification
    metrics.py        distributional metrics in float64
    data.py           evaluation prompts and data helpers
    logging_utils.py  Run / Prediction / Verdict run-directory logging
    checkpoints.py    HuggingFace revisions and single-tensor remote HTTP range reads
"""
