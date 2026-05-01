#!/usr/bin/env python3
"""
Multi-agent exploration on a mask image.

Given a mask PNG, this script:
1) Builds a grid map with cell size 20x20 pixels
2) Derives a traversal cost map from mask occupancy
3) Runs a cooperative multi-agent exploration simulation
4) Visualizes final exploration and optionally saves an animation
"""

from __future__ import annotations

import argparse
import heapq
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


GridPos = Tuple[int, int]  # (row, col)


@dataclass
class Agent:
    agent_id: int
    pos: GridPos
    path: List[GridPos]
    trajectory: List[GridPos]


def load_mask(mask_path: Path) -> np.ndarray:
    """Load mask image as grayscale numpy array in [0, 255]."""
    img = Image.open(mask_path).convert("L")
    return np.array(img, dtype=np.uint8)


def build_grid_cost_map(
    mask: np.ndarray,
    cell_size: int = 20,
    free_threshold: int = 128,
    min_free_ratio: float = 0.10,
    max_extra_cost: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert pixel mask into grid occupancy/cost maps.

    Returns:
        traversable: bool array [rows, cols], True for free cells
        costs: float array [rows, cols], traversal cost for each free cell
    """
    h, w = mask.shape
    rows = math.ceil(h / cell_size)
    cols = math.ceil(w / cell_size)

    traversable = np.zeros((rows, cols), dtype=bool)
    costs = np.full((rows, cols), np.inf, dtype=float)

    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * cell_size, min((r + 1) * cell_size, h)
            x0, x1 = c * cell_size, min((c + 1) * cell_size, w)
            patch = mask[y0:y1, x0:x1]

            free_ratio = float(np.mean(patch >= free_threshold))
            if free_ratio >= min_free_ratio:
                traversable[r, c] = True
                # Lower free ratio means harder traversal -> larger cost.
                costs[r, c] = 1.0 + (1.0 - free_ratio) * max_extra_cost

    #img = Image.fromarray(traversable.astype(np.uint8) * 255, mode='L')
    #img.save("../generated/planning/traversibility_mask.png")
    return traversable, costs


def neighbors(pos: GridPos, shape: Tuple[int, int]) -> Iterable[GridPos]:
    r, c = pos
    rows, cols = shape
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc


def heuristic(a: GridPos, b: GridPos) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def a_star(
    start: GridPos,
    goal: GridPos,
    traversable: np.ndarray,
    costs: np.ndarray,
) -> Optional[List[GridPos]]:
    """A* path search on grid. Returns path including start and goal."""
    if start == goal:
        return [start]
    if not traversable[goal]:
        return None

    open_heap: List[Tuple[float, float, GridPos]] = []
    heapq.heappush(open_heap, (heuristic(start, goal), 0.0, start))

    came_from: Dict[GridPos, GridPos] = {}
    g_score: Dict[GridPos, float] = {start: 0.0}

    while open_heap:
        _, g, current = heapq.heappop(open_heap)
        if current == goal:
            break
        if g > g_score.get(current, float("inf")):
            continue

        for nb in neighbors(current, traversable.shape):
            if not traversable[nb]:
                continue
            tentative = g + float(costs[nb])
            if tentative < g_score.get(nb, float("inf")):
                g_score[nb] = tentative
                came_from[nb] = current
                f_score = tentative + heuristic(nb, goal)
                heapq.heappush(open_heap, (f_score, tentative, nb))

    if goal not in came_from:
        return None

    path = [goal]
    curr = goal
    while curr != start:
        curr = came_from[curr]
        path.append(curr)
    path.reverse()
    return path


def farthest_start_positions(
    traversable: np.ndarray, num_agents: int, rng: random.Random
) -> List[GridPos]:
    free_cells = list(zip(*np.where(traversable)))
    if len(free_cells) < num_agents:
        raise ValueError("Not enough free cells for requested number of agents.")

    starts = [free_cells[rng.randrange(len(free_cells))]]
    while len(starts) < num_agents:
        best_cell = None
        best_dist = -1.0
        for cell in free_cells:
            if cell in starts:
                continue
            d = min(heuristic(cell, s) for s in starts)
            if d > best_dist:
                best_dist = d
                best_cell = cell
        starts.append(best_cell)
    return starts


def choose_next_target(
    agent_pos: GridPos,
    unvisited: Set[GridPos],
    reserved_targets: Set[GridPos],
    traversable: np.ndarray,
    costs: np.ndarray,
) -> Tuple[Optional[GridPos], Optional[List[GridPos]]]:
    """
    Pick a reachable unvisited target.

    We shortlist nearest candidates by Manhattan distance, then pick first
    reachable via A*.
    """
    if not unvisited:
        return None, None

    candidates = sorted(unvisited - reserved_targets, key=lambda p: heuristic(agent_pos, p))
    for target in candidates[:80]:
        path = a_star(agent_pos, target, traversable, costs)
        if path is not None:
            return target, path
    return None, None


def run_exploration(
    traversable: np.ndarray,
    costs: np.ndarray,
    num_agents: int,
    seed: int,
    max_steps: int = 3000,
    start_position: GridPos = None,
) -> Tuple[List[Agent], Set[GridPos], List[List[GridPos]], int]:
    rng = random.Random(seed)
    if start_position is not None:
        if not traversable[start_position]:
            raise ValueError(f"Start position {start_position} is not traversable.")
        starts = [start_position]*num_agents
    else:
        starts = farthest_start_positions(traversable, num_agents, rng)
    
    agents = [Agent(agent_id=i, pos=starts[i], path=[], trajectory=[starts[i]])
            for i in range(num_agents)
        ]

    free_cells = set(zip(*np.where(traversable)))
    visited: Set[GridPos] = set(starts)
    frames: List[List[GridPos]] = [[a.pos for a in agents]]

    for step in range(max_steps):
        if visited == free_cells:
            return agents, visited, frames, step

        unvisited = free_cells - visited
        reserved_targets: Set[GridPos] = set()
        for agent in agents:
            if agent.path:
                reserved_targets.add(agent.path[-1])

        for agent in agents:
            if not agent.path:
                target, path = choose_next_target(
                    agent.pos, unvisited, reserved_targets, traversable, costs
                )
                if target is not None and path is not None:
                    # Skip current position; move begins next step.
                    agent.path = path[1:]
                    reserved_targets.add(target)

        for agent in agents:
            if agent.path:
                agent.pos = agent.path.pop(0)
                visited.add(agent.pos)
            agent.trajectory.append(agent.pos)

        frames.append([a.pos for a in agents])

    return agents, visited, frames, max_steps


def plot_results(
    mask: np.ndarray,
    traversable: np.ndarray,
    costs: np.ndarray,
    agents: Sequence[Agent],
    visited: Set[GridPos],
    cell_size: int,
) -> None:
    rows, cols = traversable.shape
    coverage = len(visited) / max(1, int(np.sum(traversable)))

    fig, axs = plt.subplots(1, 3, figsize=(17, 5))

    axs[0].imshow(mask, cmap="gray")
    axs[0].set_title("Input Mask (white = unmasked/free)")
    axs[0].axis("off")

    cost_show = np.where(traversable, costs, np.nan)
    im = axs[1].imshow(cost_show, cmap="viridis")
    axs[1].set_title(f"Grid Cost Map ({cell_size}x{cell_size}px)")
    axs[1].set_xlabel("Grid Col")
    axs[1].set_ylabel("Grid Row")
    fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04, label="Traversal cost")

    explored = np.zeros((rows, cols), dtype=float)
    for r, c in visited:
        explored[r, c] = 1.0
    explored[~traversable] = np.nan
    axs[2].imshow(explored, cmap="Greens", vmin=0.0, vmax=1.0)
    axs[2].set_title(f"Exploration Coverage: {coverage * 100:.1f}%")
    axs[2].set_xlabel("Grid Col")
    axs[2].set_ylabel("Grid Row")

    for ag in agents:
        ys = [p[0] for p in ag.trajectory]
        xs = [p[1] for p in ag.trajectory]
        axs[2].plot(xs, ys, linewidth=1.5, label=f"Agent {ag.agent_id}")
        axs[2].scatter([xs[0]], [ys[0]], marker="o", s=40)
        axs[2].scatter([xs[-1]], [ys[-1]], marker="x", s=60)

    axs[2].legend(loc="upper right", fontsize=8)
    #axs[2].invert_yaxis()
    plt.tight_layout()
    # Draw the figure to the canvas and extract as RGB image array
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((height, width, 3))
    
    plt.close(fig)  # Close to free memory, since we're not displaying
    return image


def save_animation(
    traversable: np.ndarray,
    costs: np.ndarray,
    frames: Sequence[Sequence[GridPos]],
    out_gif: Optional[Path] = None,
    interval_ms: int = 150,
) -> List[np.ndarray]:
    """
    Create animation frames and optionally save as GIF.
    
    Returns:
        List of animation frames as RGB numpy arrays (height, width, 3).
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    base = np.where(traversable, costs, np.nan)
    ax.imshow(base, cmap="viridis")
    ax.set_title("Multi-Agent Exploration Animation")
    ax.set_xlabel("Grid Col")
    ax.set_ylabel("Grid Row")
    ax.invert_yaxis()

    max_agents = max(len(frame) for frame in frames)
    colors = plt.cm.tab10(np.linspace(0, 1, max(3, max_agents)))
    points = [ax.plot([], [], "o", color=colors[i], label=f"Agent {i}")[0] for i in range(max_agents)]
    ax.legend(loc="upper right", fontsize=8)

    animation_frames: List[np.ndarray] = []

    def update(frame_idx: int):
        positions = frames[frame_idx]
        for i, point in enumerate(points):
            if i < len(positions):
                r, c = positions[i]
                point.set_data([c], [r])
            else:
                point.set_data([], [])
        ax.set_title(f"Multi-Agent Exploration Animation (step {frame_idx})")
        
        # Extract frame as RGB array
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((height, width, 3))
        animation_frames.append(image)
        
        return points

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=interval_ms,
        blit=False,
        repeat=False,
    )

    # Save to GIF if path provided
    if out_gif is not None:
        try:
            ani.save(out_gif, writer="pillow", fps=max(1, 1000 // interval_ms))
            print(f"Animation saved to: {out_gif}")
        except Exception as exc:  # pragma: no cover
            print(f"Could not save GIF animation ({exc}).")
    
    plt.close(fig)
    return animation_frames

def save_animation_with_markers(
    traversable: np.ndarray,
    costs: np.ndarray,
    frames: Sequence[Sequence[GridPos]],
    robot_type: Optional[str] = None,
    out_gif: Optional[Path] = None,
    interval_ms: int = 150,
    marker_size: int = 500,
) -> List[np.ndarray]:
    """
    Create animation frames with optional custom robot marker images.
    
    Replaces colored dots with actual marker images from data/robot_markers folder
    using matplotlib's AnnotationBbox and OffsetImage.
    Falls back to colored dots if markers not found.
    
    Args:
        traversable: Binary traversability map
        costs: Cost map for visualization
        frames: List of agent position frames
        robot_type: Optional robot type name (e.g., 'spot', 'drone')
                   Looks for marker at data/robot_markers/{robot_type}.png
        out_gif: Optional path to save GIF
        interval_ms: Milliseconds per frame
        marker_size: Size of marker to render (pixels)
    
    Returns:
        List of animation frames as RGB numpy arrays (height, width, 3).
    """
    # Try to load marker image if robot_type provided
    marker_array = None
    use_markers = False
    annotation_boxes = []
    
    if robot_type is not None:
        marker_path = Path("../data/robot_markers") / f"{robot_type}.png"
        if marker_path.exists():
            try:
                marker_image = Image.open(marker_path).convert("RGBA")
                marker_image = marker_image.resize((marker_size, marker_size), Image.Resampling.LANCZOS)
                marker_array = np.array(marker_image)
                use_markers = True
                print(f"Loaded marker image from: {marker_path}")
            except Exception as e:
                print(f"Warning: Could not load marker from {marker_path}: {e}")
        else:
            print(f"Warning: Marker not found at {marker_path}, using default colors")
    
    # Create figure with cost map background
    fig, ax = plt.subplots(figsize=(7, 7))
    base = np.where(traversable, costs, np.nan)
    ax.imshow(base, cmap="viridis")
    ax.set_title("Multi-Agent Exploration Animation")
    ax.set_xlabel("Grid Col")
    ax.set_ylabel("Grid Row")
    ax.invert_yaxis()
    

    max_agents = max(len(frame) for frame in frames)
    
    if use_markers and marker_array is not None:
        # Create AnnotationBox for each agent using marker images
        for i in range(max_agents):
            imagebox = OffsetImage(marker_array, zoom=0.1, alpha=1.0)
            ab = AnnotationBbox(
                imagebox,
                (0, 0),   # start off-screen
                frameon=False,
                xycoords="data",
                annotation_clip=False,
            )
            ab.set_animated(True)
            ax.add_artist(ab)
            annotation_boxes.append(ab)
        points = None
    else:
        # Use colored dots
        colors = plt.cm.tab10(np.linspace(0, 1, max(3, max_agents)))
        points = [ax.plot([], [], "o", color=colors[i], label=f"Agent {i}", markersize=8)[0] for i in range(max_agents)]
        ax.legend(loc="upper right", fontsize=8)

    animation_frames: List[np.ndarray] = []

    def init():
        if use_markers and annotation_boxes:
            for ab in annotation_boxes:
                ab.xy = (0, 0)
            return annotation_boxes
        elif points:
            for p in points:
                p.set_data([], [])
                p.set_animated(True)
            return points
        return []

    def update(frame_idx: int):
        positions = frames[frame_idx]
        
        if use_markers and annotation_boxes:
            # Update marker positions for each agent
            for i, ab in enumerate(annotation_boxes):
                if i < len(positions):
                    r, c = positions[i]
                    # Set the position - note: AnnotationBbox uses (x, y) which maps to (col, row)
                    ab.xy = (r, c)
                    ab.set_visible(True)
                else:
                    # Move off-screen if no agent
                    ab.xy = (-1000, -1000)
                    ab.set_visible(False)
        else:
            # Update colored dots
            if points:
                for i, point in enumerate(points):
                    if i < len(positions):
                        r, c = positions[i]
                        point.set_data([c], [r])
                    else:
                        point.set_data([], [])
        
        ax.set_title(f"Multi-Agent Exploration Animation (step {frame_idx})")
        
        # Extract frame as RGB array
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((height, width, 3))
        animation_frames.append(image)
        
        return annotation_boxes if annotation_boxes else (points if points else [])

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        init_func=init,
        interval=interval_ms,
        blit=False,
        repeat=False,
    )

    # Save to GIF if path provided
    if out_gif is not None:
        try:
            ani.save(out_gif, writer="pillow", fps=max(1, 1000 // interval_ms))
            print(f"Animation saved to: {out_gif}")
        except Exception as exc:  # pragma: no cover
            print(f"Could not save GIF animation ({exc}).")
    
    plt.close(fig)
    return animation_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-agent planning/exploration from mask PNG."
    )
    parser.add_argument("mask_png", type=Path, help="Path to input mask PNG.")
    parser.add_argument(
        "--agents",
        type=int,
        default=3,
        choices=[2, 3],
        help="Number of agents (2 or 3).",
    )
    parser.add_argument(
        "--cell-size",
        type=int,
        default=20,
        help="Grid cell size in pixels (default: 20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for repeatability.",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=Path("exploration_summary.png"),
        help="Output summary figure path.",
    )
    parser.add_argument(
        "--out-gif",
        type=Path,
        default=Path("exploration_animation.gif"),
        help="Output GIF animation path.",
    )
    parser.add_argument(
        "--no-gif",
        action="store_true",
        help="Disable GIF creation.",
    )
    return parser.parse_args()

def plan_agent_exploration(mask: np.ndarray, num_agents: int, cell_size=20, seed=7, launch_pad: List[int,int] = None, robot_type: str = "spot") -> Tuple[np.ndarray, Optional[List[np.ndarray]]]:
    traversable, costs = build_grid_cost_map(mask, cell_size=cell_size)

    free_count = int(np.sum(traversable))
    if free_count == 0:
        raise ValueError("No traversable cells detected in mask.")
    if free_count < num_agents:
        raise ValueError("Free area too small for selected agent count.")
    
    start_position : GridPos = None
    if launch_pad is not None:
        start_position =  (launch_pad[0] // cell_size, launch_pad[1] // cell_size)

    agents, visited, frames, steps = run_exploration(
        traversable=traversable,
        costs=costs,
        num_agents=num_agents,
        seed=seed,
        start_position= start_position,
    )
    coverage = 100.0 * len(visited) / free_count

    print(f"Grid size: {traversable.shape[0]} rows x {traversable.shape[1]} cols")
    print(f"Traversable cells: {free_count}")
    print(f"Visited cells: {len(visited)}")
    print(f"Coverage: {coverage:.2f}%")
    print(f"Simulation steps: {steps}")

    planning_plot = plot_results(
        mask=mask,
        traversable=traversable,
        costs=costs,
        agents=agents,
        visited=visited,
        cell_size=cell_size,
    )

    animation_frames = None
    # Create output directory if it doesn't exist
    os.makedirs("../generated/planning", exist_ok=True)
    
    animation_frames = save_animation_with_markers(
        traversable=traversable,
        costs=costs,
        frames=frames,
        robot_type=robot_type,
        out_gif=Path("../generated/planning/exploration_animation.gif"),
        interval_ms=10,
    )
    print(f"Animation frames extracted: {len(animation_frames)} frames")
    return planning_plot, animation_frames


def main() -> None:
    args = parse_args()
    mask = load_mask(args.mask_png)
    traversable, costs = build_grid_cost_map(mask, cell_size=args.cell_size)

    free_count = int(np.sum(traversable))
    if free_count == 0:
        raise ValueError("No traversable cells detected in mask.")
    if free_count < args.agents:
        raise ValueError("Free area too small for selected agent count.")

    agents, visited, frames, steps = run_exploration(
        traversable=traversable,
        costs=costs,
        num_agents=args.agents,
        seed=args.seed,
    )
    coverage = 100.0 * len(visited) / free_count

    print(f"Grid size: {traversable.shape[0]} rows x {traversable.shape[1]} cols")
    print(f"Traversable cells: {free_count}")
    print(f"Visited cells: {len(visited)}")
    print(f"Coverage: {coverage:.2f}%")
    print(f"Simulation steps: {steps}")

    plot_results(
        mask=mask,
        traversable=traversable,
        costs=costs,
        agents=agents,
        visited=visited,
        cell_size=args.cell_size,
        out_png=args.out_png,
    )
    print(f"Summary image saved to: {args.out_png}")

    if not args.no_gif:
        animation_frames = save_animation(
            traversable=traversable,
            costs=costs,
            frames=frames,
            out_gif=args.out_gif,
        )
        print(f"Animation frames extracted: {len(animation_frames)} frames")


if __name__ == "__main__":
    main()
