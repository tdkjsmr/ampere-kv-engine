"""项目配置文件的通用读取与校验工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """表示配置缺失、类型错误或违反项目契约。"""


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """读取 YAML，并保证顶层对象是映射。

    Args:
        path: YAML 文件路径。

    Returns:
        顶层字典。

    Raises:
        ConfigError: 文件为空或顶层不是映射时抛出。
    """

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)

    if not isinstance(value, dict):
        raise ConfigError(f"配置顶层必须是映射：{config_path}")
    return value


def require_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    """从父映射中取出必需的子映射。"""

    value = parent.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"字段 {key!r} 必须存在且类型为映射")
    return value


def require_value(parent: dict[str, Any], key: str, expected_type: type):
    """读取必需字段并校验 Python 类型。

    bool 是 int 的子类，因此当期望 int 时要显式拒绝 bool，避免把开关值
    意外当作层数、Head 数或字节数。
    """

    if key not in parent:
        raise ConfigError(f"缺少必需字段：{key}")

    value = parent[key]
    if expected_type is int and isinstance(value, bool):
        raise ConfigError(f"字段 {key!r} 不能使用布尔值代替整数")
    if not isinstance(value, expected_type):
        raise ConfigError(
            f"字段 {key!r} 类型错误：期望 {expected_type.__name__}，"
            f"实际为 {type(value).__name__}"
        )
    return value
