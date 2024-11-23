import random
import numpy as np
import time

# Constants
BOARD_SIZE = 8
PLAYER_1 = 1
PLAYER_2 = 2
EMPTY = 0

# Heuristic 1 functions
def defensive_heuristic(board, player):
    opponent = PLAYER_2 if player == PLAYER_1 else PLAYER_1
    own_pieces = sum(row.count(player) for row in board)
    return 2 * own_pieces + random.random()


def offensive_heuristic(board, player):
    opponent = PLAYER_2 if player == PLAYER_1 else PLAYER_1
    opponent_pieces = sum(row.count(opponent) for row in board)
    return 2 * opponent_pieces + random.random()

# Heuristic 2 functions
def offensive_heuristic_2(board, player):
    opponent = PLAYER_2 if player == PLAYER_1 else PLAYER_1
    
    # Calculate row sum indices for the heuristic
    sum_row_indices = sum(i for i, row in enumerate(board) for j, piece in enumerate(row) if piece == player)
    
    # Capture opportunities calculation
    capture_opportunities = sum(
        1 for i, row in enumerate(board[:-1]) for j, piece in enumerate(row)
        if piece == player and board[i+1][j-1:j+2].count(opponent) > 0
    )
    
    # Remaining pieces ratio
    remaining_pieces_ratio = (sum(row.count(player) for row in board) / 
                              max(1, sum(row.count(opponent) for row in board)))
    
    # Return the combined evaluation
    return 1 * sum_row_indices + 5 * capture_opportunities + 2 * remaining_pieces_ratio

def defensive_heuristic_2(board, player):
    opponent = PLAYER_2 if player == PLAYER_1 else PLAYER_1
    
    # Blockade strength calculation
    blockade_strength = sum(
        1 for i, row in enumerate(board[:-1]) for j, piece in enumerate(row)
        if piece == player and board[i+1][j:j+3].count(opponent) > 0
    )
    
    # Exposed workers count
    exposed_workers = sum(
        1 for i, row in enumerate(board[:-1]) for j, piece in enumerate(row)
        if piece == player and board[i+1][j-1:j+2].count(opponent) > 0
    )
    
    # Protected rows occupation (first three rows)
    protected_rows_occupation = sum(
        sum(1 for cell in row if cell == player) for row in board[:3]
    )
    
    # Return the final defensive evaluation
    return 3 * blockade_strength - 2 * exposed_workers + 1 * protected_rows_occupation

# Minimax agent
class MinimaxAgent:
    def __init__(self, depth, heuristics):
        self.depth = depth
        self.heuristics = heuristics
        self.nodes_expanded = 0

    def minimax(self, game_state, depth, maximizing_player):
        self.nodes_expanded += 1
        if depth == 0 or game_state.is_goal_state() != 0:
            return self.evaluate_state(game_state), None

        available_actions = game_state.available_actions()
        best_move = None

        if maximizing_player:
            max_eval = -float('inf')
            for action in available_actions:
                new_game_state = game_state.transfer(action)
                eval_value, _ = self.minimax(new_game_state, depth-1, False)
                if eval_value > max_eval:
                    max_eval = eval_value
                    best_move = action
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for action in available_actions:
                new_game_state = game_state.transfer(action)
                eval_value, _ = self.minimax(new_game_state, depth-1, True)
                if eval_value < min_eval:
                    min_eval = eval_value
                    best_move = action
            return min_eval, best_move
    
    def evaluate_state(self, game_state):
        return sum(heuristic(game_state.get_board_matrix(), game_state.player) for heuristic in self.heuristics)

    def get_best_move(self, game_state):
        _, best_move = self.minimax(game_state, self.depth, True)
        return best_move

# Alpha-beta agent
class AlphaBetaAgent:
    def __init__(self, depth, heuristics):
        self.depth = depth
        self.heuristics = heuristics
        self.nodes_expanded = 0

    def alphabeta(self, game_state, depth, alpha, beta, maximizing_player):
        self.nodes_expanded += 1
        if depth == 0 or game_state.is_goal_state() != 0:
            return self.evaluate_state(game_state), None

        available_actions = game_state.available_actions()
        best_move = None

        if maximizing_player:
            max_eval = -float('inf')
            for action in available_actions:
                new_game_state = game_state.transfer(action)
                eval_value, _ = self.alphabeta(new_game_state, depth-1, alpha, beta, False)
                if eval_value > max_eval:
                    max_eval = eval_value
                    best_move = action
                alpha = max(alpha, eval_value)
                if beta <= alpha:
                    break  # Beta cutoff
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for action in available_actions:
                new_game_state = game_state.transfer(action)
                eval_value, _ = self.alphabeta(new_game_state, depth-1, alpha, beta, True)
                if eval_value < min_eval:
                    min_eval = eval_value
                    best_move = action
                beta = min(beta, eval_value)
                if beta <= alpha:
                    break  # Alpha cutoff
            return min_eval, best_move
        
    def evaluate_state(self, game_state):
        return sum(heuristic(game_state.get_board_matrix(), game_state.player) for heuristic in self.heuristics)

    def get_best_move(self, game_state):
        _, best_move = self.alphabeta(game_state, self.depth, -float('inf'), float('inf'), True)
        return best_move

class Action:
    def __init__(self, position, direction, player_turn):
        self.position = position
        self.direction = direction
        self.player_turn = player_turn

    def get_description(self):
        directions = {
            1: "left",
            2: "middle",
            3: "right"
        }
        return f"Player {self.player_turn} moves from {self.position} to {directions[self.direction]}"
    
class GameState:
    def __init__(self, board=None, black_positions=None, white_positions=None, turn=1, board_width=8, board_height=8):
        self.board_width = board_width
        self.board_height = board_height
        self.player = turn
        self.black_positions = set(black_positions or [])
        self.white_positions = set(white_positions or [])

        if board:
            self._initialize_positions_from_board(board)

    def _initialize_positions_from_board(self, board):
        for row in range(self.board_height):
            for col in range(self.board_width):
                if board[row][col] == 1:
                    self.black_positions.add((row, col))
                elif board[row][col] == 2:
                    self.white_positions.add((row, col))

    def transfer(self, action):
        new_black_positions = set(self.black_positions)
        new_white_positions = set(self.white_positions)

        if action.player_turn == 1:
            self._move_piece(new_black_positions, new_white_positions, action, 1)
        else:
            self._move_piece(new_white_positions, new_black_positions, action, 2)

        return GameState(
            black_positions=new_black_positions,
            white_positions=new_white_positions,
            turn=switch_turn(action.player_turn),
            board_width=self.board_width,
            board_height=self.board_height
        )

    def _move_piece(self, own_positions, opponent_positions, action, player_turn):
        if action.position in own_positions:
            own_positions.remove(action.position)
            new_position = compute_new_position(action.position, action.direction, player_turn)
            own_positions.add(new_position)
            if new_position in opponent_positions:
                opponent_positions.remove(new_position)

    def available_actions(self):
        actions = []
        positions = self.black_positions if self.player == 1 else self.white_positions

        for position in sorted(positions, key=lambda p: (p[0], -p[1]) if self.player == 1 else (p[0], p[1])):
            self._generate_actions_for_position(position, actions)

        return actions

    def _generate_actions_for_position(self, position, actions):
        directions = [1, 2, 3]  # Left, middle, right
        for direction in directions:
            new_position = compute_new_position(position, direction, self.player)
            if self.player == 1 and new_position not in self.black_positions and 0 <= new_position[1] < self.board_width:
                actions.append(Action(position, direction, 1))
            elif self.player == 2 and new_position not in self.white_positions and 0 <= new_position[1] < self.board_width:
                actions.append(Action(position, direction, 2))

    def get_board_matrix(self):
        matrix = [[0 for _ in range(self.board_width)] for _ in range(self.board_height)]
        for item in self.black_positions:
            matrix[item[0]][item[1]] = 1
        for item in self.white_positions:
            matrix[item[0]][item[1]] = 2
        return matrix

    def utility(self):
        return len(self.black_positions) - len(self.white_positions)

    def is_goal_state(self):
        if 0 in [pos[0] for pos in self.white_positions] or len(self.black_positions) == 0:
            return 2  # White wins
        if self.board_height - 1 in [pos[0] for pos in self.black_positions] or len(self.white_positions) == 0:
            return 1  # Black wins
        return 0  # No winner yet
    
class GameManager:
    def __init__(self, board, agent1, agent2):
        self.board = board
        self.agent1 = agent1
        self.agent2 = agent2
        self.board_width = len(board[0])
        self.board_height = len(board)
        self.current_player = PLAYER_1
        self.nodes_expanded = {PLAYER_1: 0, PLAYER_2: 0}
        self.time_taken = {PLAYER_1: 0, PLAYER_2: 0}
        self.opponent_workers_captured = {PLAYER_1: 0, PLAYER_2: 0}

    def play_game(self):
        game_state = GameState(board=self.board)
        while game_state.is_goal_state() == 0:
            start_time = time.time()
            if game_state.player == 1:
                action = self.agent1.get_best_move(game_state)
                self.nodes_expanded[PLAYER_1] += self.agent1.nodes_expanded
                self.time_taken[PLAYER_1] += time.time() - start_time
            else:
                action = self.agent2.get_best_move(game_state)
                self.nodes_expanded[PLAYER_2] += self.agent2.nodes_expanded
                self.time_taken[PLAYER_2] += time.time() - start_time

            game_state = game_state.transfer(action)
            #print(f"Player {game_state.player} moves: {action.get_description()}")
            #self.display_board(game_state.get_board_matrix())

            # Update opponent workers captured
            if game_state.player == 1:
                self.opponent_workers_captured[PLAYER_1] += sum(1 for row in game_state.get_board_matrix() if row.count(PLAYER_2) < 2)
            else:
                self.opponent_workers_captured[PLAYER_2] += sum(1 for row in game_state.get_board_matrix() if row.count(PLAYER_1) < 2)

        self.display_winner(game_state.is_goal_state())
        print(f"Current Board State:\n{np.array(game_state.get_board_matrix())}")
        print(f"Available Actions: {game_state.available_actions}")
        print(f"Nodes Expanded: {self.nodes_expanded}")
        print(f"Time Taken: {self.time_taken}")
        print(f"Opponent Workers Captured: {self.opponent_workers_captured}")

    """
    def display_board(self, board_matrix):
        for row in board_matrix:
            print(' '.join(str(cell) for cell in row))
        print()  # Blank line between board displays
    """
    
    def display_winner(self, winner):
        if winner == 1:
           print("Player 1 (Black) wins!")
        elif winner == 2:
            print("Player 2 (White) wins!")
        else:
           print("The game ended in a draw.")
    
def switch_turn(current_turn):
        return 2 if current_turn == 1 else 1

def compute_new_position(position, direction, player):
        move_offsets = {
        1: (-1, -1), # left
        2: (-1, 0), # middle
        3: (-1, 1), # right
        } if player == 2 else {
        1: (1, -1), # left
        2: (1, 0), # middle
        3: (1, 1), # right
        }
        return position[0] + move_offsets[direction][0], position[1] + move_offsets[direction][1]
    

def main():
    # Initialize game board
    board = [[PLAYER_1 if i < 2 else (PLAYER_2 if i > 5 else EMPTY) for j in range(BOARD_SIZE)] for i in range(BOARD_SIZE)]

    # Initialize heuristics
    heuristics = {
        "offensive_1": offensive_heuristic,
        "offensive_2": offensive_heuristic_2,
        "defensive_1": defensive_heuristic,
        "defensive_2": defensive_heuristic_2
    }

    # Initialize agents
    agent1 = MinimaxAgent(depth=3, heuristics=[heuristics["offensive_1"]])
    agent2 = AlphaBetaAgent(depth=3, heuristics=[heuristics["offensive_1"]])
    agent3 = AlphaBetaAgent(depth=3, heuristics=[heuristics["offensive_2"]])
    agent4 = AlphaBetaAgent(depth=3, heuristics=[heuristics["defensive_1"]])
    agent5 = AlphaBetaAgent(depth=3, heuristics=[heuristics["defensive_2"]])

    # Define matchups
    matchups = [
        (agent1, agent2),
        (agent3, agent4),
        (agent5, agent2),
        (agent3, agent2),
        (agent5, agent4),
        (agent3, agent5)
    ]

    # Play matchups
    for i, (agent1, agent2) in enumerate(matchups):
        print(f"Matchup {i+1}:")
        game_manager = GameManager(board, agent1, agent2)
        game_manager.play_game()
        print()

if __name__ == "__main__":
    main()

