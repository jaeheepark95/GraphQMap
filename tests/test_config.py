"""Tests for config loader."""

import pytest

from configs.config_loader import Config, deep_merge, load_config


def test_config_dot_access():
    cfg = Config({"model": {"embedding_dim": 64, "dropout": 0.1}})
    assert cfg.model.embedding_dim == 64
    assert cfg.model.dropout == 0.1


def test_config_to_dict():
    original = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
    cfg = Config(original)
    assert cfg.to_dict() == original


def test_deep_merge_basic():
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    override = {"b": {"c": 99}, "e": 5}
    result = deep_merge(base, override)
    assert result == {"a": 1, "b": {"c": 99, "d": 3}, "e": 5}


def test_deep_merge_no_mutation():
    base = {"a": {"b": 1}}
    override = {"a": {"b": 2}}
    deep_merge(base, override)
    assert base["a"]["b"] == 1  # original unchanged


def test_load_stage1_config():
    cfg = load_config("configs/stage1.yaml")
    assert cfg.model.embedding_dim == 64
    assert cfg.training.stage == 1
    assert cfg.sinkhorn.tau_max == 1.0
    assert cfg.sinkhorn.tau_min == 0.05
    assert cfg.loss.type == "cross_entropy"


def test_load_stage2_config():
    cfg = load_config("configs/stage2.yaml")
    assert cfg.training.stage == 2
    assert cfg.sinkhorn.tau == 0.05
    assert cfg.sinkhorn.schedule == "fixed"
    assert cfg.loss.type == "surrogate"
    assert cfg.loss.weights.l_surr == 1.0
    assert cfg.loss.weights.alpha == 0.1
    assert cfg.loss.weights.lambda_sep == 0.1
