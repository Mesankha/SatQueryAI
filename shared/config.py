"""
Shared configuration loader using Pydantic Settings.
Provides type-safe access to config.yaml for all modules.
"""
import os
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CatalogConfig(BaseSettings):
    db_path: str = "./data/catalog.db"
    image_root: str = "./data"
    manifest_path: str = "./data/catalog_manifest.json"

    model_config = SettingsConfigDict(env_prefix="CATALOG_")


class ParserConfig(BaseSettings):
    model_id: str = "microsoft/Phi-4-mini-instruct"
    load_in_4bit: bool = True
    timeout_seconds: int = 30
    fallback_enabled: bool = True

    model_config = SettingsConfigDict(env_prefix="PARSER_")


class RetrievalConfig(BaseSettings):
    embedding_model: str = "remoteclip-vit-l-14"
    faiss_index_path: str = "./data/faiss.index"
    batch_size: int = 16
    mock: bool = True  # Can be overridden by RETRIEVAL_MOCK env var

    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_")


class VLMConfig(BaseSettings):
    model_path: str = "mbzuai-oryx/GeoChat"
    load_in_4bit: bool = True
    inference_timeout: int = 30
    lora_adapter_path: Optional[str] = "./models/geochat_lora"
    sar_lora_adapter_path: Optional[str] = "./models/geochat_sar_lora"
    mock: bool = True          # true = mock VLM (optical), false = real 4-bit VLM
    sar_mock: bool = True      # true = mock SAR VLM, false = real SAR VLM
    service_port: int = 8001
    service_host: str = "0.0.0.0"

    model_config = SettingsConfigDict(env_prefix="VLM_")


class ChangeConfig(BaseSettings):
    ndvi_threshold: float = 0.5
    sar_log_ratio_threshold: float = 2.0
    enable_llm_polish: bool = False
    mock: bool = True  # true = mock M4, false = real NDVI/log-ratio
    sar_water_max_vv_db: float = -17.0
    sar_builtup_min_vv_db: float = -10.0
    sar_builtup_min_vh_db: float = -20.0

    model_config = SettingsConfigDict(env_prefix="CHANGE_")


class APIConfig(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    request_timeout: int = 60
    replay_cache_dir: str = "./replay_cache"
    max_replay_entries: int = 100

    model_config = SettingsConfigDict(env_prefix="API_")


class SessionConfig(BaseSettings):
    storage_dir: str = "./data/sessions"
    ttl_hours: int = 24

    model_config = SettingsConfigDict(env_prefix="SESSION_")


class LoggingConfig(BaseSettings):
    level: str = "INFO"
    json_format: bool = True

    model_config = SettingsConfigDict(env_prefix="LOG_")


class Settings(BaseSettings):
    catalog: CatalogConfig = Field(default_factory=CatalogConfig)
    parser: ParserConfig = Field(default_factory=ParserConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    change: ChangeConfig = Field(default_factory=ChangeConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    model_config = SettingsConfigDict(
        yaml_file="config.yaml",
        env_nested_delimiter="__",
        extra="ignore",
    )


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get global settings instance (singleton)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Force reload settings from config.yaml and environment."""
    global _settings
    _settings = Settings()
    return _settings


def get_config_dict() -> Dict[str, Any]:
    """Get config as dict for backward compatibility with existing code."""
    s = get_settings()
    config = {
        "catalog": s.catalog.model_dump(),
        "parser": s.parser.model_dump(),
        "retrieval": s.retrieval.model_dump(),
        "vlm": s.vlm.model_dump(),
        "change": s.change.model_dump(),
        "api": s.api.model_dump(),
        "session": s.session.model_dump(),
        "logging": s.logging.model_dump(),
    }
    # Support M0_CATALOG_DB env var as alias for catalog_db path
    import os
    m0_catalog_db = os.environ.get("M0_CATALOG_DB")
    if m0_catalog_db:
        config["catalog"]["db_path"] = m0_catalog_db
        # Also update paths section if it exists
        if "paths" in config:
            config["paths"]["catalog_db"] = m0_catalog_db
    
    # Also load raw YAML to get paths section and any other non-typed config
    raw = load_yaml_config("config.yaml")
    if "paths" in raw:
        config["paths"] = raw["paths"]
        # Override with env var if set
        if m0_catalog_db:
            config["paths"]["catalog_db"] = m0_catalog_db
    return config


def load_yaml_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load YAML config directly (for modules that need raw dict)."""
    import yaml
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def get_config() -> Dict[str, Any]:
    """Get config as dict (alias for get_config_dict for backward compatibility)."""
    return get_config_dict()