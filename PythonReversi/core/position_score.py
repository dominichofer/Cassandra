from dataclasses import dataclass
from .position import Position
import re

undefined_score = +70;

@dataclass(slots=True)
class PositionScore:
    pos: Position
    score: int = undefined_score

    @staticmethod
    def from_string(s: str):
        # Example input:
        # 'OO-XXXX-OOOOOXX-OOOOXOXOOXOXOXXXOXOOOXXXOXOXOXXXOOXXXXXXOOOOOOOO X'
        # 'OO-XXXX-OOOOOXX-OOOOXOXOOXOXOXXXOXOOOXXXOXOXOXXXOOXXXXXXOOOOOOOO O +02'
        score = int(s[69:]) if len(s) > 66 else undefined_score
        return PositionScore(Position.from_string(s), score)
    
    def __str__(self) -> str:
        return f'{self.pos} % {self.score:+03}'


def is_position_score(s: str):
    return re.fullmatch(r'[XO-]{64} [XO] % [+-]\d{2}', s) is not None
