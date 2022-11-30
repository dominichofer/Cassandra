from core import parse_field

class Line:
    def __init__(self, string: str):
        index, rest = string.split('|')
        depth = rest[:6].strip().split('@')

        self.index = int(index)
        self.depth = int(depth[0])
        self.confidence = float(depth[1]) if len(depth) == 2 else float('inf')
        self.score = int(rest[7:12].strip())
        self.time = rest[13:27].strip()
        self.nodes = int(rest[28:44].strip().replace("'", ''))
        speed = rest[45:58].strip().replace("'", '')
        self.speed = int(speed) if speed else None
        pv_as_str = rest[59:79].strip().split(' ')
        self.pv = [parse_field(x) for x in pv_as_str if x != '']


def parse(string: str) -> list[Line]:
    return [Line(l) for l in string.split('\n')[2:-3]]
