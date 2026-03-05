from __future__ import annotations
from pathlib import Path
from typing import Self
import os
from dotenv import dotenv_values, load_dotenv

class Config:
    _instance: Self | None = None
    _initialized = False

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, env_file: str = ".env") -> None:
        if self._initialized:
            return

        env_path = Path(env_file).resolve()
        load_dotenv(dotenv_path=env_path)

        self._values: dict[str, str] = {}
        for key, value in dotenv_values(env_path).items():
            if value is None:
                continue
            self._values[key] = value
            setattr(self, key, os.getenv(key, value))

        self._initialized = True

    def get(self, key: str, default: str | None = None) -> str | None:
        return self._values.get(key, default)


config = Config()
