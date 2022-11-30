from copy import deepcopy
import sqlite3
import numpy
from core import Position, Game, field_to_string, parse_field


def parse_position(P, O) -> Position:
    return Position(
        numpy.frombuffer(P, dtype=numpy.uint64)[0],
        numpy.frombuffer(O, dtype=numpy.uint64)[0]
    )


def parse_moves(string: str) -> list[int]:
    return [parse_field(string[i:i+2]) for i in range(0, len(string), 2)]

def moves_to_string(moves: list) -> str:
    return ''.join(field_to_string(m) for m in moves)


class DataBase:
    def __init__(self, file_path: str = ':memory:'):
        self.con = sqlite3.connect(file_path)
        self.cur = self.con.cursor()

    def __del__(self):
        self.con.commit()
        self.con.close()

    def commit(self):
        self.con.commit()

    def execute(self, *argv):
        self.cur.execute(*argv)

    def fetch_all(self):
        return self.cur.fetchall()

    def create_table(self):
        self.cur.execute("PRAGMA foreign_keys = ON;")

        # Evaluator
        self.cur.execute("""CREATE TABLE Evaluator(
            evaluator_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            UNIQUE(name)
            );""")

        # Pair
        self.cur.execute("""CREATE TABLE Pair(
            pair_id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_eval_id INTEGER NOT NULL,
            first_level TEXT NOT NULL,
            second_eval_id INTEGER NOT NULL,
            second_level TEXT NOT NULL,
            FOREIGN KEY(first_eval_id) REFERENCES Evaluator(evaluator_id),
            FOREIGN KEY(second_eval_id) REFERENCES Evaluator(evaluator_id),
            UNIQUE(first_eval_id, first_level, second_eval_id, second_level)
            );""")

        # EvaluatorInput
        self.cur.execute("""CREATE TABLE EvaluatorInput(
            evaluator_id INTEGER NOT NULL,
            pair_id INTEGER NOT NULL,
            train TINYINT(1) NOT NULL,
            FOREIGN KEY(evaluator_id) REFERENCES Evaluator(evaluator_id),
            FOREIGN KEY(pair_id) REFERENCES Pair(pair_id)
            );""")

        # Position
        self.cur.execute("""CREATE TABLE Position(
            pos_id INTEGER PRIMARY KEY AUTOINCREMENT,
            player BIGINT NOT NULL,
            opponent BIGINT NOT NULL,
            empty_count TINYINT NOT NULL,
            UNIQUE(player, opponent)
            );""")

        # Evaluation
        self.cur.execute("""CREATE TABLE Evaluation(
            evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            evaluator_id INTEGER NOT NULL,
            pos_id INTEGER NOT NULL,
            depth INTEGER NOT NULL,
            confidence FLOAT NOT NULL,
            score FLOAT NOT NULL,
            FOREIGN KEY(evaluator_id) REFERENCES Evaluator(evaluator_id),
            FOREIGN KEY(pos_id) REFERENCES Position(pos_id),
            UNIQUE(evaluator_id, pos_id, depth, confidence)
            );""")

        # Group
        self.cur.execute("""CREATE TABLE Group(
            group_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            UNIQUE(name)
            );""")

        # PositionToGroup
        self.cur.execute("""CREATE TABLE PositionToGroup(
            pos_id INTEGER NOT NULL,
            group_id INTEGER NOT NULL,
            FOREIGN KEY(pos_id) REFERENCES Position(pos_id),
            FOREIGN KEY(group_id) REFERENCES Group(group_id),
            UNIQUE(pos_id, group_id)
            );""")

        # PositionToGame
        self.cur.execute("""CREATE TABLE PositionToGame(
            pos_id INTEGER NOT NULL,
            pair_id INTEGER NOT NULL,
            FOREIGN KEY(pos_id) REFERENCES Position(pos_id),
            FOREIGN KEY(pair_id) REFERENCES Pair(pair_id),
            UNIQUE(pos_id, pair_id)
            );""")

        # PositionToPair
        self.cur.execute("""CREATE TABLE PositionToPair(
            pos_id INTEGER NOT NULL,
            pair_id INTEGER NOT NULL,
            FOREIGN KEY(pos_id) REFERENCES Position(pos_id),
            FOREIGN KEY(pair_id) REFERENCES Pair(pair_id)
            );""")

    # Evaluator
    def add_evaluator(self, name: str):
        self.cur.execute('INSERT OR IGNORE INTO Evaluator (name) VALUES (?)', (name,))

    def get_evaluator_id(self, name: str) -> int:
        self.cur.execute('SELECT evaluator_id FROM Evaluator WHERE name = ?', (name,))
        return self.cur.fetchall()[0][0]

    # Group
    def add_group(self, name: str):
        self.cur.execute('INSERT OR IGNORE INTO Group (name) VALUES (?)', (name,))

    def get_group_id(self, name: str) -> int:
        self.cur.execute('SELECT group_id FROM Group WHERE name = ?', (name,))
        return self.cur.fetchall()[0][0]

    # Position
    def add_positions(self, pos):
        if isinstance(pos, Position):
            pos = [pos]
        self.cur.executemany(
            'INSERT OR IGNORE INTO Position (player, opponent, empty_count) VALUES (?, ?, ?)',
            ((p.P, p.O, p.EmptyCount()) for p in pos))
        
    def get_position_id(self, pos: Position) -> int:
        self.cur.execute('SELECT pos_id FROM Position WHERE player = ? AND opponent = ?', (pos.P, pos.O))
        return self.cur.fetchall()[0][0]

    def has_position(self, pos):
        self.cur.execute('SELECT count(*) FROM Position WHERE player = ? and opponent = ?', (pos.P, pos.O))
        return self.cur.fetchall()[0][0] == 1
    
    # Game
    def add_games(self, games, first_evaluator: str, first_level: str, second_evaluator: str, second_level: str):
        if isinstance(games, Game):
            games = [games]
        first_eval_id = self.get_evaluator_id(first_evaluator)
        second_eval_id = self.get_evaluator_id(second_evaluator)
        for g in games:
            self.cur.executemany('''
                INSERT OR IGNORE INTO Game (first_eval_id, first_level, second_eval_id, second_level)
                VALUES (?, ?, ?, ?)''',
                ((first_eval_id, first_level, second_eval_id, second_level) for g in games))

    def get_game_id(self, game: Game, first_evaluator: str, first_level: str, second_evaluator: str, second_level: str) -> int:
        first_eval_id = self.get_evaluator_id(first_evaluator)
        second_eval_id = self.get_evaluator_id(second_evaluator)
        pos = game.StartPosition()
        moves = moves_to_string(game.Moves())
        self.cur.execute(
            'SELECT id FROM Game WHERE first_eval_id = ? AND first_level = ? AND second_eval_id = ? AND second_level = ? AND player = ? AND opponent = ? AND moves = ?',
            (first_eval_id, first_level, second_eval_id, second_level, pos.P, pos.O, moves))
        return self.cur.fetchall()[0][0]

    # Position to Game
    def link_positions_to_game(self, pos, game: Game, first_evaluator: str, first_level: str, second_evaluator: str, second_level: str):
        if isinstance(pos, Position):
            pos = [pos]
        game_id = self.get_game_id(game, first_evaluator, first_level, second_evaluator, second_level)
        self.cur.executemany(
            'INSERT OR IGNORE INTO PositionToGame (pos_id, game_id) VALUES (?, ?)',
            [(self.get_position_id(p), game_id) for p in pos])

    # Position to Group
    def link_positions_to_group(self, pos, group_name: str):
        if isinstance(pos, Position):
            pos = [pos]
        group_id = self.get_group_id(group_name)
        self.cur.executemany(
            'INSERT OR IGNORE INTO PositionToGroup (pos_id, group_id) VALUES (?, ?)',
            [(self.get_position_id(p), group_id) for p in pos])

    # Evaluation
    def add_evaluation_to_position(self, pos: Position, eval_name: str, depth: int, confidence: float, score: float):
        eval_id = self.get_evaluator_id(eval_name)
        pos_id = self.get_position_id(pos)
        self.cur.execute(
            'INSERT OR IGNORE INTO Evaluation (evaluator_id, pos_id, depth, confidence, score) VALUES (?, ?, ?, ?, ?)',
            (eval_id, pos_id, depth, confidence, score))
    

    ## Higher order functions (aka composites)

    def add_group_with_positions(self, group_name: str, pos):
        if isinstance(pos, Position):
            pos = [pos]
        self.add_group(group_name)
        self.add_positions(pos)
        self.link_positions_to_group(pos, group_name)

    def add_games_and_positions(self, games, first_evaluator: str, first_level: str, second_evaluator: str, second_level: str):
        """Adds games. Adds their positions. Links them accordingly."""
        if isinstance(games, Game):
            games = [games]
        self.add_games(games, first_evaluator, first_level, second_evaluator, second_level)
        self.add_positions(pos for game in games for pos in game.Positions())
        for game in games:
            self.link_positions_to_game([deepcopy(pos) for pos in game.Positions()], game, first_evaluator, first_level, second_evaluator, second_level)

    # Group -> Positions
    def get_positions_of_group(self, name: str) -> list[Position]:
        self.cur.execute('''
            SELECT Position.player, Position.opponent
            FROM PositionToGroup AS gtp
            INNER JOIN Group AS g ON gtp.group_id = g.id
            INNER JOIN Position ON Position.id = gtp.pos_id
            WHERE g.name = ?
            ''', (name,))
        return [parse_position(P, O) for P, O in self.cur.fetchall()]

    # Contrahents -> Games
    def get_games_of_contrahents(self, first_evaluator: str, first_level: str, second_evaluator: str, second_level: str) -> list[Game]:
        first_eval_id = self.get_evaluator_id(first_evaluator)
        second_eval_id = self.get_evaluator_id(second_evaluator)
        self.cur.execute('''
            SELECT player, opponent, moves FROM Game
            WHERE first_eval_id = ? AND first_level = ? AND second_eval_id = ? AND second_level = ?''',
            (first_eval_id, first_level, second_eval_id, second_level))
        return [Game(parse_position(P, O), parse_moves(moves)) for P, O, moves in self.cur.fetchall()]

    # Contrahents -> Positions
    def get_positions_of_contrahents(self, first_evaluator: str, first_level: str, second_evaluator: str, second_level: str) -> list[Position]:
        first_eval_id = self.get_evaluator_id(first_evaluator)
        second_eval_id = self.get_evaluator_id(second_evaluator)
        self.cur.execute('''
            SELECT DISTINCT Position.player, Position.opponent
            FROM PositionToGame gtp
            LEFT JOIN Game ON Game.id = gtp.game_id
            LEFT JOIN Position ON Position.id = gtp.pos_id
            LEFT JOIN Evaluator player ON Game.first_eval_id = player.id
            LEFT JOIN Evaluator opponent ON Game.second_eval_id = opponent.id
            WHERE Game.first_eval_id = ? AND Game.first_level = ?
            AND Game.second_eval_id = ? AND Game.second_level = ?''',
            (first_eval_id, first_level, second_eval_id, second_level))
        return [parse_position(P, O) for P, O in self.cur.fetchall()]


    #def get_evaluations_of_position(self, pos: Position):
    #    self.cur.execute('''
    #        SELECT Evaluator.name, Evaluation.depth, Evaluation.confidence, Evaluation.score
    #        FROM Evaluation
    #        INNER JOIN Evaluator ON Evaluator.id = Evaluation.evaluator_id
    #        INNER JOIN Position ON Position.id = Evaluation.pos_id
    #        WHERE Position.player = ? AND Position.opponent = ?
    #        ''', (pos.P, pos.O))
    #    return self.cur.fetchall()
