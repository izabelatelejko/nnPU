from typing import Optional

from attr import dataclass


@dataclass
class MetricValues:
    model: str
    dataset: str
    label_frequency: float
    exp_number: int  # seed

    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float

    loss: Optional[float] = None

    epoch: Optional[int] = None
    stopping_epoch: Optional[int] = None
    time: Optional[float] = None
