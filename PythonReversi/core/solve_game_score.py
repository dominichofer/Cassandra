from .game_score_file import parse_game_score_file, write_game_score_file
from .position_score import undefined_score


def solve_game_scores(engine, game_scores, min_empty_count = 0, max_empty_count = 60):
    triple = [
        (pos, game_index, pos_index)
        for game_index, gs in enumerate(game_scores)
        for pos_index, pos in enumerate(gs.positions())
        if min_empty_count <= pos.empty_count() <= max_empty_count and gs.scores[pos_index] == undefined_score
        ]

    lines = engine.solve([t[0] for t in triple])

    for t, line in zip(triple, lines):
        game_scores[t[1]].scores[t[2]] = line.score


def solve_game_score_file(engine, file, min_empty_count, max_empty_count):
    game_scores = parse_game_score_file(file)
    solve_game_scores(engine, game_scores, min_empty_count, max_empty_count)
    write_game_score_file(game_scores, file)
