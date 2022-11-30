import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .board_panel import BoardPanel
from .score_panel import ScorePanel, WhiskedValue