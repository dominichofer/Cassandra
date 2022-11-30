from .game_score_file import parse_game_score_file, write_game_score_file


def solve_game_scores(engine, game_scores, min_empty_count = 0, max_empty_count = 60):
    positions = [
        pos
        for gs in game_scores
        for pos in gs.positions()
        if min_empty_count <= pos.empty_count() <= max_empty_count
        ]

    table = {
        pos : (game_index, pos_index)
        for game_index, gs in enumerate(game_scores)
        for pos_index, pos in enumerate(gs.positions())
        }

    lines = engine.solve(positions)

    for pos, line in zip(positions, lines):
        game_index, pos_index = table[pos]
        game_scores[game_index].scores[pos_index] = line.score


def solve_game_score_file(engine, file, min_empty_count, max_empty_count):
    game_scores = parse_game_score_file(file)
    solve_game_scores(engine, game_scores, min_empty_count, max_empty_count)
    write_game_score_file(game_scores, file)
