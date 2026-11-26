"""Tests for dataset selection and token cache configuration in training twins."""

from pathlib import Path
import pytest
from swx.cli import build_parser
from swx.s46_formation.train_activation_twins import TrainConfig


def test_cli_dataset_arguments_default():
    parser = build_parser()
    args = parser.parse_args(["train-twins", "--variant", "swiglu"])
    assert args.dataset == "HuggingFaceFW/fineweb-edu"
    assert args.dataset_config is None
    assert args.dataset_split == "train"
    assert args.dataset_text_column == "text"
    assert args.token_cache_dir == ""


def test_cli_dataset_arguments_custom():
    parser = build_parser()
    args = parser.parse_args([
        "train-twins",
        "--variant", "swiglu",
        "--dataset", "wikitext",
        "--dataset-config", "wikitext-103-raw-v1",
        "--dataset-split", "train",
        "--dataset-text-column", "text",
        "--token-cache-dir", "custom/cache/dir",
    ])
    assert args.dataset == "wikitext"
    assert args.dataset_config == "wikitext-103-raw-v1"
    assert args.dataset_split == "train"
    assert args.dataset_text_column == "text"
    assert args.token_cache_dir == "custom/cache/dir"


def test_train_config_dataset_fields():
    cfg = TrainConfig(
        variant="swiglu",
        dataset="wikitext",
        dataset_config="wikitext-103-raw-v1",
        dataset_split="validation",
        dataset_text_column="content",
    )
    assert cfg.dataset == "wikitext"
    assert cfg.dataset_config == "wikitext-103-raw-v1"
    assert cfg.dataset_split == "validation"
    assert cfg.dataset_text_column == "content"


def test_resolve_dataset():
    from swx.s46_formation.train_activation_twins import resolve_dataset

    # Aliases resolve to canonical paths and configs
    c4_name, c4_cfg = resolve_dataset("c4")
    assert c4_name == "allenai/c4"
    assert c4_cfg == "en"

    wiki_name, wiki_cfg = resolve_dataset("wikitext")
    assert wiki_name == "Salesforce/wikitext"
    assert wiki_cfg == "wikitext-103-raw-v1"

    sp_name, sp_cfg = resolve_dataset("slimpajama")
    assert sp_name == "DKYoon/SlimPajama-6B"
    assert sp_cfg is None

    # Explicit override preserved
    wiki_custom, wiki_custom_cfg = resolve_dataset("wikitext", "wikitext-2-raw-v1")
    assert wiki_custom == "Salesforce/wikitext"
    assert wiki_custom_cfg == "wikitext-2-raw-v1"

    # Unknown passes through
    other_name, other_cfg = resolve_dataset("custom/dataset", "sub")
    assert other_name == "custom/dataset"
    assert other_cfg == "sub"

