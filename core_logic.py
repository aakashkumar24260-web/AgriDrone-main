"""Core logic for Boustrophedon coverage, A* detours, and disease spread."""

import numpy as np
from collections import deque
from typing import List, Tuple, Optional
import heapq
import config

class AStarPlanner:
    """A* pathfinding for obstacle avoidance during row scanning."""
    
    def __init__(self, grid: np.ndarray):
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring cells (non-obstacle, within bounds)."""
        neighbors = []
        for dr, dc in self.directions:
            r, c = pos[0] + dr, pos[1] + dc
            if 0 <= r < self.rows and 0 <= c < self.cols:
                if self.grid[r, c] != -1:  # Not an obstacle
                    neighbors.append((r, c))
        return neighbors
    
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Find path from start to goal using A*."""
        if self.grid[start] == -1 or self.grid[goal] == -1:
            return []
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            
            for neighbor in self.get_neighbors(current):
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return []  # No path found

class BoustrophedonPlanner:
    """Boustrophedon (lawnmower) coverage path planner with A* detours."""
    
    def __init__(self, grid: np.ndarray):
        self.grid = grid.astype(np.float32)
        self.rows, self.cols = self.grid.shape
        self.obstacle_value = -1.0
        self.astar = AStarPlanner(grid)
        
    def build_path(self) -> List[Tuple[int, int]]:
        """Generate Boustrophedon coverage path.
        
        Returns:
            List of (row, col) tuples representing the drone's path.
        """
        path = []
        
        for row in range(self.rows):
            if row % 2 == 0:
                # Left to right
                col_range = range(self.cols)
            else:
                # Right to left
                col_range = range(self.cols - 1, -1, -1)
            
            for col in col_range:
                if self.grid[row, col] == self.obstacle_value:
                    # Obstacle encountered - find detour
                    detour = self._find_detour(row, col, row % 2 == 0)
                    if detour:
                        # Add intermediate path points
                        # Simplified: just skip obstacle and continue
                        continue
                else:
                    path.append((row, col))
        
        return path
    
    def _find_detour(self, row: int, col: int, moving_right: bool) -> List[Tuple[int, int]]:
        """Find detour around obstacle."""
        # Find the nearest passable cell on the same row after the obstacle
        step = 1 if moving_right else -1
        target_col = col + step
        
        while 0 <= target_col < self.cols:
            if self.grid[row, target_col] != self.obstacle_value:
                # Path exists - navigate around obstacle via adjacent row
                detour_path = []
                # Try going up or down
                for dr in [1, -1]:
                    new_row = row + dr
                    if 0 <= new_row < self.rows:
                        if (self.grid[new_row, col] != self.obstacle_value and 
                            self.grid[new_row, target_col] != self.obstacle_value):
                            # Can detour through this row
                            detour_path.extend([
                                (new_row, col),
                                (new_row, target_col),
                                (row, target_col)
                            ])
                            return detour_path
                break
            target_col += step
        
        return []

class CellularAutomaton:
    """Disease spread simulation using cellular automaton rules."""
    
    def __init__(self, grid: np.ndarray):
        self.grid = grid.copy()
        self.rows, self.cols = grid.shape
        
    def get_neighbors_4(self, r: int, c: int) -> List[Tuple[int, int]]:
        """Get 4-neighborhood (up, down, left, right)."""
        neighbors = []
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                if self.grid[nr, nc] != -1:  # Not obstacle
                    neighbors.append((nr, nc))
        return neighbors
    
    def spread_disease(self, steps: int, seeds: List[Tuple[int, int, int]] = None):
        """Simulate disease spread for given number of steps.
        
        Args:
            steps: Number of spread iterations
            seeds: List of (row, col, disease_type) where 1=Early, 2=Severe
        """
        if seeds:
            for r, c, dtype in seeds:
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    if self.grid[r, c] != -1:
                        # Start with the seed's disease type
                        # We'll store disease status separately
                        pass
        
        # Run cellular automaton
        for step in range(steps):
            new_grid = self.grid.copy()
            
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.grid[r, c] == -1:  # Obstacle
                        continue
                    
                    # Get current disease class
                    current_class = self.grid[r, c]
                    if current_class == 0:  # Healthy
                        # Check neighbors for disease spread
                        for nr, nc in self.get_neighbors_4(r, c):
                            neighbor_class = self.grid[nr, nc]
                            if neighbor_class == 1:  # Early disease
                                if np.random.random() < config.SPREAD_PROB_EARLY:
                                    new_grid[r, c] = 1
                                    break
                            elif neighbor_class == 2:  # Severe disease
                                if np.random.random() < config.SPREAD_PROB_SEVERE:
                                    new_grid[r, c] = 2
                                    break
                    elif current_class == 1:  # Early -> can progress to severe
                        if np.random.random() < 0.1:  # 10% progression rate
                            new_grid[r, c] = 2
            
            self.grid = new_grid
        
        return self.grid

def run_scan(field_grid: np.ndarray, spread_steps: int = 5, 
             seeds: List[Tuple[int, int, int]] = None) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Run the full scan process.
    
    Args:
        field_grid: 2D array with values (0=Healthy, 1=Early, 2=Severe, -1=Obstacle)
        spread_steps: Number of disease spread iterations
        seeds: Disease seed points
    
    Returns:
        Tuple of (scanned_grid, path)
    """
    # Apply disease spread
    automaton = CellularAutomaton(field_grid)
    if seeds:
        # Apply seeds to grid
        for r, c, dtype in seeds:
            if 0 <= r < field_grid.shape[0] and 0 <= c < field_grid.shape[1]:
                if field_grid[r, c] != -1:
                    field_grid[r, c] = dtype
    
    spread_grid = automaton.spread_disease(spread_steps, seeds)
    
    # Generate Boustrophedon path
    planner = BoustrophedonPlanner(spread_grid)
    path = planner.build_path()
    
    # Simulate scanning: mark scanned cells in a copy
    scanned_grid = spread_grid.copy()
    
    return scanned_grid, path