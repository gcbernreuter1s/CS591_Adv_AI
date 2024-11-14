# CS591_Adv_AI
The MiniMaxAgent code is composed of three classes: GameState, Action, and MiniMax. The GameState class tracks the positions of black and white pieces, whose turn it is, and functions for calculating effective moves. The Action class performs a move by knowing a piece’s position from the GameState, and direction of movement. The MinimaxAgent class is responsible for selecting the best move based on the minimax algorithm. It recursively evaluates all possible future game states to a depth of 3. The agent alternates between maximizing and minimizing the difference in the number of black and white pieces and investigating potential moves until the game ends in a tie, win, or loss. The select_best_move method chooses the best action using this minimax algorithm and then returns the new game state after the best move. 

1. Implement minimax search for a search tree depth of 3. - **Morgan**
2. Implement alpha-beta search for a search tree of depth more than that of minimax. - **Kamal**
3. Implement Defensive Heuristic 1 and Offensive Heuristic 1. **Vira**
4. Design and implement an Offensive Heuristic 2 with the idea of beating Defensive Heuristic 1. AND Design and implement a Defensive Heuristic 2 with the idea of beating Offensive Heuristic 1. - **Garrett**
5. Integrate search functions and heuristics to complete the 6 match-ups. This includes reporting requirements for each match-up. - **Garrett**
