import sqlite3
import numpy as np
from Position import *
from Game import Game
import timeit


def ParsePosition(P, O) -> Position:
    return Position(
        np.frombuffer(P, dtype=np.uint64)[0],
        np.frombuffer(O, dtype=np.uint64)[0])


def ParseMoves(string: str):
    return [StringToField(string[i:i+2]) for i in range(0, len(string), 2)]


class DataBase:
    def __init__(self, file_path: str = ':memory:'):
        self.con = sqlite3.connect(file_path)
        self.cur = self.con.cursor()

    def __del__(self):
        self.con.commit()
        self.con.close()

    def Commit(self):
        self.con.commit()

    def Execute(self, *argv):
        self.cur.execute(*argv)

    def Fetchall(self, statement: str):
        self.cur.execute(statement)
        return self.cur.fetchall()

    def CreateTables(self):
        self.cur.execute("PRAGMA foreign_keys = ON;")
        self.cur.execute("""CREATE TABLE Evaluator(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            UNIQUE(name)
            );""")
        self.cur.execute("""CREATE TABLE Game(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_eval_id INTEGER NOT NULL,
            player_level TEXT NOT NULL,
            opponent_eval_id INTEGER NOT NULL,
            opponent_level TEXT NOT NULL,
            player BIGINT NOT NULL,
            opponent BIGINT NOT NULL,
            moves TEXT NOT NULL,
            FOREIGN KEY(player_eval_id) REFERENCES Evaluator(id),
            FOREIGN KEY(opponent_eval_id) REFERENCES Evaluator(id)
            );""")
        self.cur.execute("""CREATE TABLE EvaluatorInput(
            evaluator_id INTEGER NOT NULL,
            game_id INTEGER NOT NULL,
            train TINYINT(1) NOT NULL,
            FOREIGN KEY(evaluator_id) REFERENCES Evaluator(id),
            FOREIGN KEY(game_id) REFERENCES Game(id)
            );""")
        self.cur.execute("""CREATE TABLE Position(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player BIGINT NOT NULL,
            opponent BIGINT NOT NULL,
            empty_count TINYINT NOT NULL,
            UNIQUE(player, opponent)
            );""")
        self.cur.execute("""CREATE TABLE Evaluation(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            evaluator_id INTEGER NOT NULL,
            pos_id INTEGER NOT NULL,
            depth INTEGER NOT NULL,
            confidence FLOAT NOT NULL,
            score INTEGER NOT NULL,
            FOREIGN KEY(evaluator_id) REFERENCES Evaluator(id),
            FOREIGN KEY(pos_id) REFERENCES Position(id),
            UNIQUE(evaluator_id, pos_id, depth, confidence)
            );""")
        self.cur.execute("""CREATE TABLE PositionGroup(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            UNIQUE(name)
            );""")
        self.cur.execute("""CREATE TABLE GroupToPosition(
            pos_id INTEGER NOT NULL,
            group_id INTEGER NOT NULL,
            FOREIGN KEY(pos_id) REFERENCES Position(id),
            FOREIGN KEY(group_id) REFERENCES PositionGroup(id)
            );""")

    def AddEvaluator(self, name: str):
        self.cur.execute('INSERT OR IGNORE INTO Evaluator (name) VALUES (?)', (name,))

    def GetEvaluatorId(self, name: str):
        self.cur.execute('SELECT id FROM Evaluator WHERE name = ?', (name,))
        return self.cur.fetchall()[0][0]


    def CreateGroup(self, name: str):
        self.cur.execute('INSERT OR IGNORE INTO PositionGroup (name) VALUES (?)', (name,))

    def GetGroupId(self, name: str):
        self.cur.execute('SELECT id FROM PositionGroup WHERE name = ?', (name,))
        return self.cur.fetchall()[0][0]


    def HasPosition(self, pos):
        self.cur.execute('SELECT count(*) FROM Position WHERE player = ? and opponent = ?', (pos.P, pos.O))
        return self.cur.fetchall()[0][0] == 1

    def AddPosition(self, pos):
        if isinstance(pos, Position):
            pos = [pos]
        self.cur.executemany('INSERT OR IGNORE INTO Position (player, opponent, empty_count) VALUES (?, ?, ?)', [(p.P, p.O, p.EmptyCount()) for p in pos])
        
    def GetPositionId(self, pos: Position):
        self.cur.execute('SELECT id FROM Position WHERE player = ? and opponent = ?', (pos.P, pos.O))
        return self.cur.fetchall()[0][0]


    def CreateGroupWithPositions(self, group_name: str, pos):
        self.CreateGroup(group_name)
        group_id = self.GetGroupId(group_name)
        for p in pos:
            self.AddPosition(p)
            pos_id = self.GetPositionId(p)
            self.cur.execute('INSERT INTO GroupToPosition (pos_id, group_id) VALUES (?, ?)', (pos_id, group_id))

    def GetGroup(self, name):
        self.cur.execute('''
            SELECT Position.player, Position.opponent
            FROM GroupToPosition AS gtp
            INNER JOIN PositionGroup AS g ON gtp.group_id = g.id
            INNER JOIN Position ON Position.id = gtp.pos_id
            WHERE g.name = ?
            ''', (name,))
        return [ParsePosition(x[0], x[1]) for x in self.cur.fetchall()]

    def AddEvaluationToPosition(self, pos: Position, eval_name: str, depth: int, confidence: float, score: int):
        eval_id = self.GetEvaluatorId(eval_name)
        pos_id = self.GetPositionId(pos)
        self.cur.execute('INSERT OR IGNORE INTO Evaluation (evaluator_id, pos_id, depth, confidence, score) VALUES (?, ?, ?, ?, ?)',
                        (eval_id, pos_id, depth, confidence, score))

    def GetEvaluationsOfPosition(self, pos: Position):
        self.cur.execute('''
            SELECT Evaluator.name, Evaluation.depth, Evaluation.confidence, Evaluation.score
            FROM Evaluation
            INNER JOIN Evaluator ON Evaluator.id = Evaluation.evaluator_id
            INNER JOIN Position ON Position.id = Evaluation.pos_id
            WHERE Position.player = ? AND Position.opponent = ?
            ''', (pos.P, pos.O))
        return self.cur.fetchall()


    def AddGame(self, game: Game, player_evaluator: str, player_level: str, opponent_evaluator: str, opponent_level: str):
        player_eval_id = self.GetEvaluatorId(player_evaluator)
        opponent_eval_id = self.GetEvaluatorId(opponent_evaluator)
        pos = game.StartPosition()
        moves = ''.join(FieldToString(m) for m in game.Moves())
        self.cur.execute('''
            INSERT INTO Game (player_eval_id, player_level, opponent_eval_id, opponent_level, player, opponent, moves)
            VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (player_eval_id, player_level, opponent_eval_id, opponent_level, pos.P, pos.O, moves))

    def GetGames(self, player_evaluator: str, player_level: str, opponent_evaluator: str, opponent_level: str):
        player_eval_id = self.GetEvaluatorId(player_evaluator)
        opponent_eval_id = self.GetEvaluatorId(opponent_evaluator)
        self.cur.execute('''
            SELECT player, opponent, moves FROM Game
            WHERE player_eval_id = ? AND player_level = ? AND opponent_eval_id = ? AND opponent_level = ?''',
            (player_eval_id, player_level, opponent_eval_id, opponent_level))
        return [Game(ParsePosition(x[0], x[1]), ParseMoves(x[2])) for x in self.cur.fetchall()]



#db = DataBase(r'G:\Reversi\DB.db')
##db = DataBase()
#db.CreateTables()
#db.AddEvaluator('RND')
#db.AddEvaluator('Edax4.4')
#db.AddEvaluator('Cassandra')

#for d in range(11):
#    start = timeit.default_timer()
#    db.CreateGroupWithPositions(f'all_unique_e{60-d}', AllUniqueChildren(Position.Start(), d))
#    stop = timeit.default_timer()
#    print(stop - start)