"""Runtime configuration for DataSleuth.

All deployment and per-run knobs live in :class:`Settings`. Values come from
environment variables (prefix ``DATASLEUTH_``), an optional ``.env`` file, or a
YAML config file (used by the evaluation harness for ablations).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["gemini", "anthropic", "openai", "ollama", "fake"]
SandboxKind = Literal["subprocess", "docker"]


class Settings(BaseSettings):
    """Application settings; every field can be overridden by ``DATASLEUTH_<NAME>``."""

    model_config = SettingsConfigDict(
        env_prefix="DATASLEUTH_", env_file=".env", extra="ignore", populate_by_name=True
    )

    # --- LLM -------------------------------------------------------------
    llm_provider: Provider = "gemini"
    llm_model: str = "gemini-3.1-flash-lite"
    llm_temperature: float = 0.0
    llm_timeout_s: float = 90.0
    llm_rpm: int = Field(default=10, description="Max requests/minute (0 = unlimited).")
    llm_daily_limit: int = Field(
        default=0, description="Max uncached calls per calendar day (0 = unlimited)."
    )
    llm_max_backoff_s: float = 90.0
    llm_thinking_budget: int | None = Field(
        default=None, description="Gemini thinking-token budget (0 disables thinking)."
    )
    cache_dir: Path = Path(".cache/llm")
    log_dir: Path = Path(".cache/logs")
    use_cache: bool = True

    gemini_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "GEMINI_API_KEY", "GOOGLE_API_KEY", "DATASLEUTH_GEMINI_API_KEY"
        ),
    )
    anthropic_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "DATASLEUTH_ANTHROPIC_API_KEY"),
    )
    openai_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("OPENAI_API_KEY", "DATASLEUTH_OPENAI_API_KEY")
    )
    ollama_host: str = Field(
        default="http://localhost:11434",
        validation_alias=AliasChoices("OLLAMA_HOST", "DATASLEUTH_OLLAMA_HOST"),
    )

    # --- Agent -----------------------------------------------------------
    max_retries: int = 3
    max_llm_calls: int = 12
    allow_pandas: bool = True
    use_router: bool = True
    use_schema_linker: bool = True
    schema_linker_min_tables: int = Field(
        default=4, description="Skip LLM linking when the schema has fewer tables than this."
    )
    use_few_shot: bool = False
    few_shot_k: int = 3
    few_shot_path: Path | None = None
    use_critic_semantic: bool = True
    use_analyst: bool = True
    use_chart: bool = True
    sample_values_in_prompt: bool = True

    # --- Execution / sandbox ---------------------------------------------
    sandbox: SandboxKind = "subprocess"
    docker_image: str = "datasleuth-sandbox:latest"
    row_cap: int = 1000
    cell_cap_chars: int = 500
    sql_timeout_s: float = 10.0
    pandas_timeout_s: float = 20.0
    pandas_memory_mb: int = 1024
    pandas_cpu_s: int = 20

    # --- Data / sessions -------------------------------------------------
    data_dir: Path = Path(".data")
    upload_max_mb: int = 200
    profile_sample_rows: int = 50_000
    session_ttl_min: int = 60
    allow_db_connect: bool = Field(
        default=False,
        description="Let API users connect external databases (SSRF risk: trusted deployments only).",
    )
    db_connect_allowed_hosts: list[str] = Field(
        default_factory=list, description="If set, only these DB hosts may be connected."
    )
    cors_origins: list[str] = ["http://localhost:5173"]
    frontend_dist: Path = Path("frontend/dist")

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> Settings:
        """Load settings from a YAML file; explicit ``overrides`` win over the file."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        data.update(overrides)
        return cls(**data)

    def api_key_for(self, provider: Provider) -> str | None:
        """Return the API key configured for ``provider`` (None if not applicable)."""
        return {
            "gemini": self.gemini_api_key,
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
        }.get(provider)


def get_settings(**overrides: Any) -> Settings:
    """Build settings from the environment, applying keyword overrides."""
    return Settings(**overrides)
