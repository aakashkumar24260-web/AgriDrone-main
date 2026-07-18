"""Grid utilities for loading, validating, and generating field layouts."""

import json
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import config

def load_field_from_json(file_path: str) -> Dict:
    """Load field configuration from JSON file.
    
    Args:
        file_path: Path to JSON file
    
    Returns:
        Dictionary with field configuration
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def validate_grid(grid: np.ndarray) -> bool:
    """Validate grid dimensions and values."""
    if grid.shape != (config.GRID_ROWS, config.GRID_COLS):
        return False
    valid_values = {0, 1, 2, -1}
    unique = np.unique(grid)
    return all(val in valid_values for val in unique)

def create_grid_from_config(config_dict: Dict) -> np.ndarray:
    """Create grid array from configuration dictionary.
    
    Args:
        config_dict: Field configuration with obstacles and disease seeds
    
    Returns:
        2D numpy array (25x25) with disease class values
    """
    grid = np.zeros((config.GRID_ROWS, config.GRID_COLS))
    
    # Mark obstacles
    obstacles = config_dict.get('obstacles', [])
    for r, c in obstacles:
        if 0 <= r < config.GRID_ROWS and 0 <= c < config.GRID_COLS:
            grid[r, c] = -1
    
    # Place disease seeds
    seeds = config_dict.get('disease_seeds', [])
    for seed in seeds:
        cell = seed.get('cell', [])
        if len(cell) == 2:
            r, c = cell
            disease_type = 1 if seed.get('type') == 'early' else 2
            spread_radius = seed.get('spread_radius', 1)
            
            # Mark cells within radius with disease
            for dr in range(-spread_radius, spread_radius + 1):
                for dc in range(-spread_radius, spread_radius + 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < config.GRID_ROWS and 0 <= nc < config.GRID_COLS:
                        if grid[nr, nc] != -1:  # Not obstacle
                            # Distance-based disease severity
                            dist = abs(dr) + abs(dc)
                            if dist <= 1:
                                grid[nr, nc] = disease_type
                            elif dist <= 2 and disease_type == 2:
                                grid[nr, nc] = 1  # Early disease around severe
    
    return grid

def generate_random_grid() -> np.ndarray:
    """Generate a random grid for testing."""
    grid = np.random.choice([0, 1, 2], size=(config.GRID_ROWS, config.GRID_COLS), 
                           p=[0.6, 0.25, 0.15])
    # Add some obstacles
    num_obstacles = np.random.randint(5, 15)
    for _ in range(num_obstacles):
        r = np.random.randint(0, config.GRID_ROWS)
        c = np.random.randint(0, config.GRID_COLS)
        grid[r, c] = -1
    return grid

def save_field_json(grid: np.ndarray, name: str, file_path: str):
    """Save grid as JSON field configuration."""
    config_dict = {
        "name": name,
        "crop_type": "Wheat",
        "grid_configuration": {
            "rows": config.GRID_ROWS,
            "cols": config.GRID_COLS,
            "drone_start": [0, 0]
        },
        "obstacles": [],
        "disease_seeds": []
    }
    
    # Extract obstacles
    obstacles = np.argwhere(grid == -1)
    config_dict["obstacles"] = obstacles.tolist()
    
    # Extract disease seeds (non-zero, non-obstacle)
    disease_cells = np.argwhere(grid > 0)
    for r, c in disease_cells:
        dtype = "early" if grid[r, c] == 1 else "severe"
        config_dict["disease_seeds"].append({
            "cell": [int(r), int(c)],
            "type": dtype,
            "spread_radius": 1
        })
    
    with open(file_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

def get_available_fields() -> List[Tuple[str, str]]:
    """Get list of available field configurations."""
    fields = []
    for json_file in config.DATA_DIR.glob("*.json"):
        fields.append((json_file.name, config.DEFAULT_FIELDS.get(json_file.name, "Custom field")))
    return fields