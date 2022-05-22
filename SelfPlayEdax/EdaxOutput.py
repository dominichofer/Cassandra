class Line:
    def __init__(self, string:str):
        index, rest = string.split('|')

        self.index = int(index)
        self.depth = int(rest[:6].strip().split('@')[0])
        self.score = int(rest[7:12].strip())
        self.time = rest[13:27].strip()
        self.nodes = int(rest[28:41].strip())
        speed = rest[42:52].strip()
        self.speed = int(speed) if speed else None
        self.pv = rest[53:73].strip().split(' ')
        if self.pv == ['']:
            self.pv = None


def Parse(string:str):
    return [Line(l) for l in string.split('\n')[2:-4]]