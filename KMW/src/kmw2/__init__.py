from __future__ import annotations

from importlib import import_module

__all__ = [
    "AssignmentHead",
    "MappingProxyLoss",
    "QubitMappingDataset",
    "Trainer",
    "UNetMapping",
    "build_hardware_cache",
    "build_model_inputs",
    "evaluate_manifest",
    "evaluate_one_circuit",
    "train_from_config",
]


def __getattr__(name: str):
    lookup = {
        "QubitMappingDataset": ("kmw2.data.dataset", "QubitMappingDataset"),
        "evaluate_manifest": ("kmw2.evaluation.evaluate", "evaluate_manifest"),
        "evaluate_one_circuit": ("kmw2.evaluation.evaluate", "evaluate_one_circuit"),
        "MappingProxyLoss": ("kmw2.losses.loss", "MappingProxyLoss"),
        "AssignmentHead": ("kmw2.models.model", "AssignmentHead"),
        "UNetMapping": ("kmw2.models.model", "UNetMapping"),
        "build_hardware_cache": ("kmw2.preprocessing.pipeline", "build_hardware_cache"),
        "build_model_inputs": ("kmw2.preprocessing.pipeline", "build_model_inputs"),
        "Trainer": ("kmw2.training.trainer", "Trainer"),
        "train_from_config": ("kmw2.training.trainer", "train_from_config"),
    }
    if name not in lookup:
        raise AttributeError(name)
    module_name, attr_name = lookup[name]
    module = import_module(module_name)
    return getattr(module, attr_name)
