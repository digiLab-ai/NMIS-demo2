from dataclasses import dataclass

@dataclass(frozen=True)
class Brand:
    INDIGO: str = "#16425B"
    INDIGO_50: str = "#8AA0AD"
    KEPPEL: str = "#16D5C2"
    KEY_LIME: str = "#EBF38B"
    CHARCOAL: str = "#2A2A2A"
    LIGHT_GREY: str = "#EAEAEA"

BRAND = Brand()

@dataclass(frozen=True)
class AppConfig:
    default_samples: int = 100
    train_fraction: float = 0.8
    seed: int = 42

CONFIG = AppConfig()
