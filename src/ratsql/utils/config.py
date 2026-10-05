"""YAML configuration with single inheritance and dotted overrides.

A config file may contain ``inherit: <relative path>``; the parent is loaded
first and the child is deep-merged on top.  Command-line overrides use dotted
keys, e.g. ``--set model.rat.num_layers=2 train.batch_size=8``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable

import yaml


def merge_dicts(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_dicts(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _parse_scalar(text: str) -> Any:
    """Parse an override value using YAML scalar semantics (ints, floats, bools, null, lists)."""
    return yaml.safe_load(text)


def set_dotted(d: dict, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def get_dotted(d: dict, dotted_key: str, default: Any = None) -> Any:
    cur: Any = d
    for k in dotted_key.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _load_yaml_with_inheritance(path: Path, _seen: set | None = None) -> dict:
    path = path.resolve()
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"Circular config inheritance at {path}")
    _seen.add(path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    parent = data.pop("inherit", None)
    if parent is None:
        return data
    parents = parent if isinstance(parent, list) else [parent]
    merged: dict = {}
    for p in parents:
        merged = merge_dicts(merged, _load_yaml_with_inheritance((path.parent / p), _seen))
    return merge_dicts(merged, data)


class Config(dict):
    """A dict with attribute access and dotted ``get``."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as e:
            raise AttributeError(item) from e
        return Config(value) if isinstance(value, dict) and not isinstance(value, Config) else value

    def get_path(self, dotted_key: str, default: Any = None) -> Any:
        return get_dotted(self, dotted_key, default)

    def to_dict(self) -> dict:
        return json.loads(json.dumps(self))


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> Config:
    """Load a YAML config (with inheritance) and apply ``key=value`` overrides."""
    data = _load_yaml_with_inheritance(Path(path))
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        key, raw = item.split("=", 1)
        set_dotted(data, key.strip(), _parse_scalar(raw))
    data.setdefault("_meta", {})["config_path"] = str(Path(path))
    return Config(data)


def apply_overrides(cfg: dict, overrides: dict) -> Config:
    """Apply a nested dict of overrides (used by the ablation runner)."""
    return Config(merge_dicts(cfg, overrides))
