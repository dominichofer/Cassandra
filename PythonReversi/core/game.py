from .field import field_to_string, parse_field
from .position import Position, play, play_pass, possible_moves
import re


class Game:
    def __init__(self, start: Position, moves = None):
        if not possible_moves(start):
            passed = play_pass(start)
            if possible_moves(passed):
                start = passed
        self.__start = start
        self.__current = start
        self.__moves = []
        for move in (moves or []):
            self.play(move)

    @staticmethod
    def from_string(s: str):
        # Example input:
        # 'OO-XXXX-OOOOOXX-OOOOXOXOOXOXOXXXOXOOOXXXOXOXOXXXOOXXXXXXOOOOOOOO O C1 h1'

        return Game(
            Position.from_string(s),
            [parse_field(m) for m in s[66:].strip().split(' ')]
            )

    def __str__(self) -> str:
        return ' '.join([str(self.__start)] + [field_to_string(m) for m in self.__moves])
                
    def __eq__(self, o):
        return self.__start == o.__start and self.__moves == o.__moves

    def __neq__(self, o):
        return not self == o

    @property
    def start_position(self) -> Position:
        return self.__start
    
    @property
    def current_position(self) -> Position:
        return self.__current
    
    @property
    def moves(self) -> list:
        return self.__moves

    def play(self, move):
        self.__moves.append(move)
        self.__current = play(self.__current, move)

        # Pass implicit if game is not over
        if not possible_moves(self.__current):
            passed = play_pass(self.__current)
            if possible_moves(passed):
                self.__current = passed

    def positions(self):
        pos = self.__start
        yield pos
        for move in self.__moves:
            pos = play(pos, move)
            yield pos

    
def is_game(s: str):
    return re.fullmatch(r'[XO-]{64} [XO]( [A-H][1-8])*', s) is not None
            

#def Children(game: Game, empty_count_diff: int = 1):
#    if empty_count_diff == 0:
#        yield game
#        return
#    if game.IsOver():
#        return
#    for move in game.possible_moves():
#        yield from Children(play(game, move), empty_count_diff - 1)
        

#def WriteToFile(data, file_path: Path):
#    if isinstance(data, Iterable):
#        file_path.write_text('\n'.join(str(Game(d)) for d in data))
#    else:
#        file_path.write_text(str(Game(data)))
