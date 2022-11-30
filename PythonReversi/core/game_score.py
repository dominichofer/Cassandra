from .game import Game
from .field import parse_field
from .position import Position
from .position_score import undefined_score


class GameScore:
    def __init__(self, start: Position, moves = None, scores = None):
        self.__game = Game(start, moves)
        self.scores = scores or [undefined_score] * (len(self.moves) + 1)

    @staticmethod
    def from_game(game: Game):
        return GameScore(game.start_position, game.moves)

    @staticmethod
    def from_string(s: str):
        # Example input:
        # 'OO-XXXX-OOOOOXX-OOOOXOXOOXOXOXXXOXOOOXXXOXOXOXXXOOXXXXXXOOOOOOOO O C1 h1 +70 -02 +12'

        parts = s.strip().split(' ')
        pos = Position.from_string(f'{parts[0]} {parts[1]}')
        moves = []
        scores = []
        for part in parts[2:]:
            if part[0] in ['+', '-'] and part[1:].isdecimal():
                scores.append(int(part))
            else:
                moves.append(parse_field(part))
        return GameScore(pos, moves, scores)

    def __str__(self) -> str:
        return ' '.join([str(self.__game)] + [f'{s:+03}' for s in self.scores])
                
    def __eq__(self, o):
        return self.__game == o.__game and self.scores == o.scores

    def __neq__(self, o):
        return not self == o

    @property
    def start_position(self) -> Position:
        return self.__game.start_position
    
    @property
    def current_position(self) -> Position:
        return self.__game.current_position
    
    @property
    def moves(self) -> list:
        return self.__game.moves

    def clear_scores(self):
        self.scores = [undefined_score] * (len(self.moves) + 1)

    def play(self, move):
        self.__game.play(move)
        self.scores.append(undefined_score)

    def positions(self):
        return self.__game.positions()

    def pos_scores(self):
        return zip(self.positions(), self.scores)
