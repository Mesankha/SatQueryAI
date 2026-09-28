"""
M0 Catalog Configuration
"""
import os
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BigEarthNetConfig(BaseSettings):
    source_path: str = "./data/raw/bigearthnet"
    enabled: bool = True
    model_config = SettingsConfigDict(env_prefix="M0_BIGEARTHNET_")


class CDVQAConfig(BaseSettings):
    source_path: str = "./data/raw/cdvqa"
    enabled: bool = True
    model_config = SettingsConfigDict(env_prefix="M0_CDVQA_")


class GEEAssamConfig(BaseSettings):
    project_id: Optional[str] = None
    region: str = "Assam, India"
    date_range: list[str] = ["2023-01-01", "2024-12-31"]
    enabled: bool = True
    model_config = SettingsConfigDict(env_prefix="M0_GEE_ASSAM_")


class OutputConfig(BaseSettings):
    db_path: str = "./data/catalog.db"
    image_root: str = "./data"
    manifest_path: str = "./data/catalog_manifest.json"
    model_config = SettingsConfigDict(env_prefix="M0_OUTPUT_")


class LoRADatasetConfig(BaseSettings):
    output_dir: str = "./data/bigearthnet_lora"
    max_samples: int = 5000
    enabled: bool = True
    model_config = SettingsConfigDict(env_prefix="M0_LORA_")


class M0CatalogConfig(BaseSettings):
    bigearthnet: BigEarthNetConfig = Field(default_factory=BigEarthNetConfig)
    cdvqa: CDVQAConfig = Field(default_factory=CDVQAConfig)
    gee_assam: GEEAssamConfig = Field(default_factory=GEEAssamConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    lora_dataset: LoRADatasetConfig = Field(default_factory=LoRADatasetConfig)

    model_config = SettingsConfigDict(
        yaml_file="config.yaml",
        env_nested_delimiter="__",
        extra="ignore",
    )


def get_m0_config() -> M0CatalogConfig:
    return M0CatalogConfig()