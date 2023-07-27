from core import parse_field

class Line:
    def __init__(self, string: str):
        index, depth, evl, score, time, nodes, nps, pv = string.split('|')
        depth = depth.strip().split('@')

        self.index = int(index.strip().replace("'", ''))
        self.depth = int(depth[0])
        self.confidence = float(depth[1]) if len(depth) == 2 else float('inf')
        self.score = int(evl.strip())
        self.time = time.strip()
        self.nodes = int(nodes.strip().replace("'", ''))
        speed = nps.strip().replace("'", '')
        self.speed = int(speed) if speed else None
        self.pv = [parse_field(pv.strip())]


def parse(string: str) -> list[Line]:
    return [Line(l) for l in string.split('\n')[2:-3]]
