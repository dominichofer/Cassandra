import datetime
from copy import deepcopy
from .game import Game

def self_play(player, games) -> list[Game]:
    if isinstance(games, Game):
        games = [games]
    games = deepcopy(games)
    for d in range(60):
        moves = player.choose_move([g.current_position for g in games])
        for game, move in zip(games, moves):
            if move != 64:
                game.play(move)
    return games
