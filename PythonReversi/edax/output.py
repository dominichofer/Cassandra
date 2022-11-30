from core import parse_field

class Line:
    def __init__(self, string: str):
        index, rest = string.split('|')
        depth = rest[:6].strip().split('@')

        self.index = int(index)
        self.depth = int(depth[0])
        self.selectivity = int(depth[1][:-1]) if len(depth) == 2 else None
        self.confidence = selectivity_to_confidence(self.selectivity)
        self.score = int(rest[7:12].strip())
        self.time = rest[13:27].strip()
        self.nodes = int(rest[28:41].strip())
        speed = rest[42:52].strip()
        self.speed = int(speed) if speed else None
        pv_as_str = rest[53:73].strip().split(' ')
        self.pv = [parse_field(x) for x in pv_as_str if x != '']


def selectivity_to_confidence(selectivity: int) -> float:
    if selectivity is None:
        return float('inf')
    if selectivity == 73:
        return 1.1
    if selectivity == 87:
        return 1.5
    if selectivity == 95:
        return 2.0
    if selectivity == 98:
        return 2.6
    if selectivity == 99:
        return 3.3


def parse(string: str) -> list[Line]:
    return [Line(l) for l in string.split('\n')[2:-4]]
