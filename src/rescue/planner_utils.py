import numpy as np
from scipy import ndimage as ndi
from skimage.draw import line
from skimage.morphology import binary_dilation, disk

def connect_islands_with_bridges(mask, bridge_width=1, connectivity=2):
    """
    Connect all connected components in a binary mask using narrow bridges.

    Parameters
    ----------
    mask : array-like of bool or {0,1}
        Input binary mask.
    bridge_width : int
        Approximate bridge thickness in pixels.
    connectivity : int
        1 for 4-connected, 2 for 8-connected component labeling.

    Returns
    -------
    connected_mask : np.ndarray of bool
        Mask with bridge pixels added.
    """
    mask = np.asarray(mask).astype(bool)

    structure = ndi.generate_binary_structure(2, connectivity)
    labeled, n = ndi.label(mask, structure=structure)

    if n <= 1:
        return mask.copy()

    # Compute component centroids
    centroids = np.array(ndi.center_of_mass(mask, labeled, range(1, n + 1)))

    # Build MST over centroids (Prim's algorithm)
    remaining = set(range(n))
    tree = {0}
    remaining.remove(0)
    edges = []

    def dist(i, j):
        return np.linalg.norm(centroids[i] - centroids[j])

    while remaining:
        best = None
        best_d = float("inf")
        for i in tree:
            for j in remaining:
                d = dist(i, j)
                if d < best_d:
                    best_d = d
                    best = (i, j)
        i, j = best
        edges.append((i, j))
        tree.add(j)
        remaining.remove(j)

    out = mask.copy()
    r = max(1, bridge_width // 2)
    selem = disk(r)

    # Draw bridge between nearest points on each MST edge
    for i, j in edges:
        p1 = np.round(centroids[i]).astype(int)
        p2 = np.round(centroids[j]).astype(int)
        rr, cc = line(p1[0], p1[1], p2[0], p2[1])

        valid = (rr >= 0) & (rr < out.shape[0]) & (cc >= 0) & (cc < out.shape[1])
        rr, cc = rr[valid], cc[valid]
        bridge = np.zeros_like(out, dtype=bool)
        bridge[rr, cc] = True
        bridge = binary_dilation(bridge, selem)
        out |= bridge

    return out

def add_launch_pad(mask, center, side_length, value=True):
    """
    Add a small square mask centered at (row, col) on the input mask.

    Parameters
    ----------
    mask : np.ndarray (2D)
        Input mask (dtype does not need to be bool).
    center : tuple (row, col)
        Center of the square.
    side_length : int
        Side length of the square (odd or even).
    value : scalar
        Value to set inside the square (e.g. True, 1, 255).

    Returns
    -------
    mask_out : np.ndarray
        Mask with the square added.
    """
    mask = np.asarray(mask)
    r_center, c_center = center
    half = side_length // 2

    r0 = r_center - half
    r1 = r_center + half + (side_length % 2)  # handle even lengths
    c0 = c_center - half
    c1 = c_center + half + (side_length % 2)

    # Clip to image bounds
    r0 = max(r0, 0)
    r1 = min(r1, mask.shape[0])
    c0 = max(c0, 0)
    c1 = min(c1, mask.shape[1])

    # Write the square
    out = mask.copy()
    out[r0:r1, c0:c1] = value

    return out

def optimize_robot_exploration_masks(robot2traverse,drone_bridge_width=30):
    """
    Optimize traversability masks for multi-robot coordination.
    
    Allocates regions to each robot type based on traversibility and priority:
    - Finds largest traversible region touching edge in 'spot' mask as launch pad
    - If no edge-touching region, places launch pad at bottom-center
    - Adds 40-pixel bridges for drones to connect traversible regions and launch pad
    - Assigns regions based on priorities:
      * Large drones: highest priority, decreases near launch pad
      * Small drones: medium constant priority
      * Spot: lowest priority overall, but highest in 30% closest to launch pad
    
    Args:
        robot2traverse: dict with keys 'spot', 'small drone', 'large drone'
                       values are binary masks (0=not traversible, 1=traversible)
    
    Returns:
        tuple: (optimized_masks_dict, launch_pad_center)
            - optimized_masks_dict: dict with processed masks (no region duplication)
            - launch_pad_center: tuple (y, x) center of launch pad
    """
    # Get spot mask and dimensions
    spot_mask = robot2traverse['spot'].copy().astype(float)
    h, w = spot_mask.shape
    
    # Find connected components in spot mask
    labeled_array, num_features = ndi.label(spot_mask)
    
    # Find components touching edges
    edge_components = set()
    for i in range(1, num_features + 1):
        component = (labeled_array == i)
        if (component[0, :].any() or component[-1, :].any() or 
            component[:, 0].any() or component[:, -1].any()):
            edge_components.add(i)
    
    # Determine launch pad center
    if edge_components:
        # Find largest edge-touching component
        max_size = 0
        largest_component_id = None
        for comp_id in edge_components:
            size = np.sum(labeled_array == comp_id)
            if size > max_size:
                max_size = size
                largest_component_id = comp_id
        
        # Get point on edge from largest component
        component_mask = (labeled_array == largest_component_id)
        component_coords = np.where(component_mask)
        
        # Find the component point closest to an edge
        distances_to_edge = np.minimum(
            np.minimum(component_coords[0], h - 1 - component_coords[0]),
            np.minimum(component_coords[1], w - 1 - component_coords[1])
        )
        edge_point_idx = np.argmin(distances_to_edge)
        launch_y = int(component_coords[0][edge_point_idx])
        launch_x = int(component_coords[1][edge_point_idx])
    else:
        # No edge-touching region: use bottom-center
        launch_y = h - 1
        launch_x = w // 2
    
    launch_pad_center = (launch_y, launch_x)
    print(f"Launch pad center: {launch_pad_center}")
    
    # Create distance map from launch pad center
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    dist_to_launch = np.sqrt((yy - launch_y)**2 + (xx - launch_x)**2)
    
    # Normalize distance to [0, 1]
    max_dist = np.max(dist_to_launch)
    dist_normalized = dist_to_launch / max_dist if max_dist > 0 else np.zeros_like(dist_to_launch)
    
    # Calculate 30% closest region threshold (distance-based)
    close_threshold = np.percentile(dist_to_launch, 30)
    is_close_to_launch = dist_to_launch <= close_threshold
    
    # Get original traversible masks
    spot_m = robot2traverse['spot'].astype(float)
    small_drone_m = robot2traverse['small_drone'].astype(float)
    large_drone_m = robot2traverse['large_drone'].astype(float)
    
    # Create priority maps for each robot type
    # Large drones: highest priority far from launch (1.0), lower near launch (0.3)
    large_drone_priority = np.where(is_close_to_launch, 0.3, 1.0)
    
    # Small drones: medium constant priority (0.5)
    small_drone_priority = np.full((h, w), 0.5, dtype=float)
    
    # Spot: lowest overall, but highest near launch (0.8 close, 0.05-0.15 far)
    spot_priority = np.where(is_close_to_launch, 0.8, dist_normalized * 0.1 + 0.05)
    
    # Apply priority only where robot can traverse
    large_drone_p = large_drone_priority * large_drone_m
    small_drone_p = small_drone_priority * small_drone_m
    spot_p = spot_priority * spot_m
    
    # For each pixel, assign to robot with highest priority
    # Stack priorities: axis 2 order is [spot, small_drone, large_drone]
    priority_stack = np.stack([spot_p, small_drone_p, large_drone_p], axis=2)
    best_robot_idx = np.argmax(priority_stack, axis=2)
    
    spot_mask = (best_robot_idx == 0).astype(np.uint8)
    small_drone_mask = (best_robot_idx == 1).astype(np.uint8)
    large_drone_mask = (best_robot_idx == 2).astype(np.uint8)
    
    #Addling Launch zone islands
    spot_mask = add_launch_pad(spot_mask, launch_pad_center, side_length=40, value=1)
    small_drone_mask = add_launch_pad(small_drone_mask, launch_pad_center, side_length=40, value=1)
    large_drone_mask = add_launch_pad(large_drone_mask, launch_pad_center, side_length=40, value=1)
    
    # Create bridges for both drone types
    small_drone_mask_with_bridges = connect_islands_with_bridges(small_drone_mask, bridge_width=drone_bridge_width)
    large_drone_mask_with_bridges = connect_islands_with_bridges(large_drone_mask, bridge_width=drone_bridge_width)
    
    # Create output masks (mutually exclusive allocation)
    processed_masks = {
        'spot': spot_mask,
        'small_drone': small_drone_mask_with_bridges,
        'large_drone': large_drone_mask_with_bridges,
    }
    
    # Print coverage statistics
    for robot, mask in processed_masks.items():
        coverage = np.sum(mask)
        print(f"{robot}: {coverage} traversible cells allocated")
    
    return processed_masks, launch_pad_center