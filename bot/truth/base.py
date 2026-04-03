from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class TruthResult:
    probability: float  # 0-1, the "true" probability
    confidence: float   # 0-1, how confident we are
    source: str         # which truth engine produced this

    def edge(self, market_price: float) -> float:
        return self.probability - market_price


class TruthEngine(ABC):
    @abstractmethod
    def get_truth(self, market: dict) -> TruthResult | None:
        """Return true probability for this market, or None if no opinion."""
        ...

    @abstractmethod
    def can_handle(self, market: dict) -> bool:
        """Return True if this engine can evaluate this market type."""
        ...
