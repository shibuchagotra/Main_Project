from dataclasses import dataclass
from typing import Optional,Any

@dataclass
class GraphState:
    question: str
    model:str
    role="assistant"
    # outputs
    answer: Optional[Any] = None
    generated_code: Optional[str] = None
    executed_code: Optional[str] = None
    error: Optional[str] = None
