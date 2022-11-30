from .position import Position, possible_moves
import random

# TODO: Rename RandomPlayer!

class RandomEngine:
    def name(self, *_):
        return 'Random'

    @staticmethod
    def __random_move(pos: Position):
        pm = possible_moves(pos)
        if pm:
            return pm[random.randint(0, pm.size() - 1)]
        return 64

    def choose_move(self, pos) -> list[int]:
        return [self.__random_move(p) for p in pos]
