import pandas as pd
import numpy as np
import geopandas as gpd
import folium
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from shapely.geometry import Point, MultiPoint
import json
import argparse
from collections import Counter, deque

def build_zone_connectivity_graph(zones, zone_centroids, lat_col, lon_col):
    """
    Build a graph of zone connectivity based on closest points between zones.
    
    Two zones are considered neighbors if there is a point in zone A where the closest point 
    from another zone is a point from zone B, and vice versa.
    
    Parameters:
    -----------
    zones : dict
        Dictionary mapping zone IDs to dataframes with points in each zone
    zone_centroids : dict
        Dictionary mapping zone IDs to centroid coordinates (lat, lon)
    lat_col, lon_col : str
        Column names for latitude and longitude
        
    Returns:
    --------
    dict
        Dictionary mapping zone IDs to lists of neighboring zone IDs
    """
    import numpy as np
    from scipy.spatial import cKDTree
    import time
    
    start_time = time.time()
    print("Building zone connectivity graph...")
    
    # Initialize the neighbor dictionary
    zone_neighbors = {zone_id: set() for zone_id in zones.keys()}
    
    # Filter out empty zones
    valid_zones = {zone_id: zones[zone_id] for zone_id in zones if zone_id in zones and len(zones[zone_id]) > 0}
    valid_zone_ids = list(valid_zones.keys())
    
    if len(valid_zone_ids) <= 1:
        print("Too few valid zones for connectivity")
        return {zone_id: [] for zone_id in zones.keys()}
    
    # Step 1: Precompute KD trees and point arrays for each zone (do this only once)
    zone_trees = {}
    zone_points = {}
    
    for zone_id in valid_zone_ids:
        zone_df = valid_zones[zone_id]
        points = np.array(zone_df[[lat_col, lon_col]])
        zone_points[zone_id] = points
        zone_trees[zone_id] = cKDTree(points)
    
    # Step 2: Create a single combined KD tree with all points and track which zone each point belongs to
    all_points = []
    point_to_zone = []
    
    for zone_id in valid_zone_ids:
        points = zone_points[zone_id]
        all_points.append(points)
        point_to_zone.extend([zone_id] * len(points))
    
    if not all_points:
        print("No valid points found in any zone")
        return {zone_id: [] for zone_id in zones.keys()}
    
    all_points = np.vstack(all_points)
    all_tree = cKDTree(all_points)
    
    # Step 3: For each zone, find potential neighboring zones using a more efficient approach
    potential_neighbors = {}
    
    # First pass: Use centroids for quick filtering to identify potential neighbors
    for i, zone_a_id in enumerate(valid_zone_ids):
        # Get centroid of zone A
        centroid_a = zone_centroids.get(zone_a_id)
        if centroid_a is None:
            continue
            
        # Calculate approximate max radius of each zone
        max_radius_a = 0
        if len(zone_points[zone_a_id]) > 0:
            # Calculate distances from points to centroid
            dists = np.sqrt(np.sum((zone_points[zone_a_id] - centroid_a)**2, axis=1))
            max_radius_a = np.max(dists)
        
        potential_neighbors[zone_a_id] = []
        
        # For each other zone, check if it's potentially a neighbor
        for zone_b_id in valid_zone_ids:
            if zone_a_id == zone_b_id:
                continue
                
            centroid_b = zone_centroids.get(zone_b_id)
            if centroid_b is None:
                continue
                
            # Calculate approximate max radius of zone B
            max_radius_b = 0
            if len(zone_points[zone_b_id]) > 0:
                dists = np.sqrt(np.sum((zone_points[zone_b_id] - centroid_b)**2, axis=1))
                max_radius_b = np.max(dists)
            
            # Calculate distance between centroids
            centroid_dist = np.sqrt(np.sum((np.array(centroid_a) - np.array(centroid_b))**2))
            
            # If zones could potentially overlap or be close, consider them potential neighbors
            if centroid_dist <= (max_radius_a + max_radius_b) * 1.5:  # Add buffer for safety
                potential_neighbors[zone_a_id].append(zone_b_id)
    
    # Step 4: Use sampling to reduce computation cost 
    for zone_a_id in valid_zone_ids:
        points_a = zone_points[zone_a_id]
        
        # If too many points, use sampling for efficiency
        sample_size = min(len(points_a), 100)  # Sample up to 100 points per zone
        if len(points_a) > sample_size:
            # Stratified sampling to preserve spatial distribution
            indices = np.linspace(0, len(points_a)-1, sample_size, dtype=int)
            sample_points_a = points_a[indices]
        else:
            sample_points_a = points_a
            
        # For each sampled point in zone A, find the closest non-A point
        for point in sample_points_a:
            # Query k=2 to get the closest point (which might be self) and the next closest
            distances, indices = all_tree.query([point], k=50)  # Get 50 nearest neighbors
            
            # Filter out points from the same zone
            for idx, dist in zip(indices[0], distances[0]):
                if idx < len(point_to_zone) and point_to_zone[idx] != zone_a_id:
                    neighbor_zone = point_to_zone[idx]
                    zone_neighbors[zone_a_id].add(neighbor_zone)
                    zone_neighbors[neighbor_zone].add(zone_a_id)
                    break
    
    # Convert sets to lists for the final result
    for zone_id in zone_neighbors:
        zone_neighbors[zone_id] = list(zone_neighbors[zone_id])
    
    print(f"Zone connectivity graph built in {time.time() - start_time:.2f} seconds")
    
    # Report connectivity
    for zone_id in valid_zone_ids:
        print(f"Zone {zone_id} has neighbors: {zone_neighbors[zone_id]}")
    
    return zone_neighbors

def load_data(csv_path, lat_col="Latitude", lon_col="Longitude"):
    """Load point data from CSV file and ensure coordinates are valid"""
    try:
        df = pd.read_csv(csv_path)
        
        # Validate required columns
        if lat_col not in df.columns or lon_col not in df.columns:
            raise ValueError(f"CSV must contain {lat_col} and {lon_col} columns")
        
        # Convert coordinates to numeric and drop invalid rows
        df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
        df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
        
        invalid_rows = df[df[lat_col].isna() | df[lon_col].isna()].shape[0]
        if invalid_rows > 0:
            print(f"Dropped {invalid_rows} rows with invalid coordinates")
            df = df.dropna(subset=[lat_col, lon_col])
        
        if df.empty:
            raise ValueError("No valid coordinate data found")
        # # Create default Commune column if needed
        # if 'Commune' not in df.columns:
        #     df['Commune'] = "Unknown"
        return df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in kilometers using Haversine formula"""
    # Convert degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Earth radius in kilometers
    
    return c * r



def find_intermediate_points(pt1, pt2, all_points_dict, max_search_distance=None):
    """
    Find points from other zones that lie between two points.
    
    Parameters:
    -----------
    pt1 : tuple
        (latitude, longitude) of first point
    pt2 : tuple
        (latitude, longitude) of second point
    all_points_dict : dict
        Dictionary with zone_id as key and list of (lat, lon) tuples as values
    max_search_distance : float, optional
        Maximum distance to search for intermediate points
        
    Returns:
    --------
    int
        Count of points from other zones that lie between the two points
    """
    lat1, lon1 = pt1
    lat2, lon2 = pt2
    
    # Calculate direct distance between points
    direct_dist = haversine_distance(lat1, lon1, lat2, lon2)
    
    # Default max search distance if not specified
    if max_search_distance is None:
        max_search_distance = direct_dist * 0.5
    
    # Create bounding box for efficient filtering
    min_lat = min(lat1, lat2) - max_search_distance/111.0  # approx 1 degree = 111km
    max_lat = max(lat1, lat2) + max_search_distance/111.0
    min_lon = min(lon1, lon2) - max_search_distance/(111.0 * np.cos(np.radians((lat1 + lat2)/2)))
    max_lon = max(lon1, lon2) + max_search_distance/(111.0 * np.cos(np.radians((lat1 + lat2)/2)))
    
    # Creating vectors for line equation
    v_line = (lat2 - lat1, lon2 - lon1)
    line_length = np.sqrt(v_line[0]**2 + v_line[1]**2)
    v_unit = (v_line[0]/line_length, v_line[1]/line_length)
    
    intermediate_count = 0
    
    # Check all points from other zones
    for zone_id, points in all_points_dict.items():
        for lat, lon in points:
            # Quick bounding box check
            if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
                continue
            
            # Vector from point 1 to the candidate point
            v_to_point = (lat - lat1, lon - lon1)
            
            # Project this vector onto the line
            proj_length = v_to_point[0]*v_unit[0] + v_to_point[1]*v_unit[1]
            
            # Check if projection falls on the line segment
            if 0 < proj_length < line_length:
                # Calculate distance from point to line
                # v_to_line = v_to_point - proj_length*v_unit
                # dist_to_line = np.sqrt(v_to_line[0]**2 + v_to_line[1]**2)
                
                # Calculate actual perpendicular distance in km
                proj_point = (lat1 + proj_length*v_unit[0], lon1 + proj_length*v_unit[1])
                dist_to_line = haversine_distance(lat, lon, proj_point[0], proj_point[1])
                
                # Consider as intermediate if close enough to the line
                if dist_to_line < max_search_distance:
                    intermediate_count += 1
    
    return intermediate_count


def density_adjusted_distance(lat1, lon1, lat2, lon2, all_points_dict, own_zone_id, 
                             base_weight=1.0, intermediate_weight=0.5, max_factor=5):
    """
    Calculate a density-adjusted distance that considers points from other zones.
    
    Parameters:
    -----------
    lat1, lon1 : float
        Coordinates of first point
    lat2, lon2 : float
        Coordinates of second point
    all_points_dict : dict
        Dictionary with zone_id as key and list of (lat, lon) tuples as values
    own_zone_id : int
        ID of the zone being checked for connectivity
    base_weight : float
        Base weight for physical distance
    intermediate_weight : float
        Weight for intermediate points
    max_factor : float
        Maximum multiplier for distance adjustment
        
    Returns:
    --------
    float
        Adjusted distance considering density of points from other zones
    """
    # Get physical distance
    physical_dist = haversine_distance(lat1, lon1, lat2, lon2)
    
    # Calculate adaptive search radius based on distance
    search_radius = min(physical_dist * 0.3, 1.0)  # km
    
    # Count intermediate points from other zones
    intermediate_points = find_intermediate_points(
        (lat1, lon1), 
        (lat2, lon2), 
        {z: pts for z, pts in all_points_dict.items() if z != own_zone_id},
        max_search_distance=search_radius
    )
    
    # Penalize distance based on intermediate points
    penalty_factor = 1.0 + min(intermediate_weight * intermediate_points, max_factor - 1.0)
    
    # Calculate adjusted distance
    adjusted_dist = physical_dist * penalty_factor
    
    return adjusted_dist


def check_zone_connectivity(df, zone_id,  lat_col="Latitude", lon_col="Longitude",all_zones_dict=None):
    """
    Check if a zone is spatially connected using density-adjusted distances.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing all points with zone_id column
    zone_id : int
        ID of the zone to check
    all_zones_dict : dict, optional
        Dictionary with zone_id as key and DataFrame of points as values
    lat_col, lon_col : str
        Column names for latitude and longitude
        
    Returns:
    --------
    bool
        True if zone is connected, False otherwise
    """
    zone_df = df[df['zone_id'] == zone_id].copy()
    if len(zone_df) <= 1:
        return True
    
    # Prepare all zones dict if not provided
    if all_zones_dict is None:
        all_zones_dict = {z: df[df['zone_id'] == z] for z in df['zone_id'].unique()}
    print()
    # Convert points to more efficient format for distance calculations
    all_points_dict = {
        z: z_df[[lat_col, lon_col]].values.tolist() 
        for z, z_df in all_zones_dict.items()
    }
    
    coords = zone_df[[lat_col, lon_col]].values
    
    # Create graph for connectivity check
    connectivity_graph = nx.Graph()
    for i in range(len(coords)):
        connectivity_graph.add_node(i)
    
    # Calculate average inter-point distance within zone for adaptive threshold
    avg_dist = 0
    count = 0
    
    if len(coords) <= 100:
        # For small zones, exact calculation
        for i in range(len(coords)):
            for j in range(i+1, min(i+11, len(coords))):
                avg_dist += haversine_distance(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
                count += 1
    else:
        # For large zones, sample
        sample_indices = np.random.choice(len(coords), min(100, len(coords)), replace=False)
        for i in range(len(sample_indices)):
            for j in range(i+1, min(i+11, len(sample_indices))):
                idx1, idx2 = sample_indices[i], sample_indices[j]
                avg_dist += haversine_distance(coords[idx1][0], coords[idx1][1], 
                                               coords[idx2][0], coords[idx2][1])
                count += 1
    
    if count > 0:
        avg_dist /= count
    else:
        avg_dist = 5.0  # Default fallback
    
    # Adaptive connectivity threshold based on average distance
    connectivity_threshold = avg_dist * 3
    
    # Use KD-tree for efficient nearest neighbor search
    from scipy.spatial import cKDTree
    # Convert coordinates to radians for spherical distance
    coords_rad = np.radians(coords)
    # Swap lat/lon for KDTree
    tree_coords = np.column_stack([coords_rad[:, 1], coords_rad[:, 0]])
    tree = cKDTree(tree_coords)
    
    # Find neighbors within threshold
    for i in range(len(coords)):
        # Query point in radians (lon, lat) order for KDTree
        query_point = (np.radians(coords[i][1]), np.radians(coords[i][0]))
        
        # Find indices of points within threshold
        indices = tree.query_ball_point(query_point, connectivity_threshold / 6371.0)
        
        for j in indices:
            if i != j:
                # Calculate density-adjusted distance
                adj_dist = density_adjusted_distance(
                    coords[i][0], coords[i][1], 
                    coords[j][0], coords[j][1],
                    all_points_dict, zone_id
                )
                if adj_dist <= connectivity_threshold:
                    connectivity_graph.add_edge(i, j)
    
    # Check if graph is connected
    is_connected = nx.is_connected(connectivity_graph)
    
    if not is_connected:
        # If not connected, check if there might be just a few outliers
        components = list(nx.connected_components(connectivity_graph))
        
        # Sort components by size
        components.sort(key=len, reverse=True)
        
        # If the largest component contains at least 100% of points, consider connected
        if len(components[0]) >=  len(coords):
            is_connected = True
            print(f"Zone {zone_id}: Considering connected despite small disconnected outliers")
    
    return is_connected

def calculate_zone_workload(points, lat_col="Latitude", lon_col="Longitude"):
    points_component = len(points) * 15
    
    if len(points) <= 1:
        traversal_time = 0
    else:
        coords = points[[lat_col, lon_col]].to_numpy()
        
        if len(coords) <= 50:
            distance_matrix = np.zeros((len(coords), len(coords)))
            for i in range(len(coords)):
                for j in range(i+1, len(coords)):
                    lat1, lon1 = coords[i]
                    lat2, lon2 = coords[j]
                    dist = haversine_distance(lat1, lon1, lat2, lon2)
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist
            
            unvisited = set(range(len(coords)))
            current = unvisited.pop()
            tour_length = 0
            
            while unvisited:
                next_city = min(unvisited, key=lambda x: distance_matrix[current, x])
                tour_length += distance_matrix[current, next_city]
                current = next_city
                unvisited.remove(current)
            
            traversal_time = tour_length / 0.5
        
        else:
            sample_size = min(50, len(coords))
            sampled_points = points.sample(sample_size)
            sample_traversal_time = calculate_zone_workload(sampled_points, lat_col, lon_col) - (sample_size * 15)
            
            scaling_factor = (len(points) / sample_size) ** 0.5
            traversal_time = sample_traversal_time * scaling_factor
    
    total_workload = points_component + traversal_time
    
    return total_workload

def calculate_zone_workload_with_coefficients(points, lat_col="Latitude", lon_col="Longitude", points_coef = 1, distance_coef = 1 ):
    points_component = len(points) * 15 * points_coef
    
    if len(points) <= 1:
        traversal_time = 0
    else:
        coords = points[[lat_col, lon_col]].to_numpy()
        
        # if len(coords) <= 50:
        distance_matrix = np.zeros((len(coords), len(coords)))
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                lat1, lon1 = coords[i]
                lat2, lon2 = coords[j]
                dist = haversine_distance(lat1, lon1, lat2, lon2)
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        
        unvisited = set(range(len(coords)))
        current = unvisited.pop()
        tour_length = 0
        
        while unvisited:
            next_city = min(unvisited, key=lambda x: distance_matrix[current, x])
            tour_length += distance_matrix[current, next_city]
            current = next_city
            unvisited.remove(current)
        
        traversal_time = (tour_length / 0.5)* distance_coef
        
        # else:
        #     sample_size = min(50, len(coords))
        #     sampled_points = points.sample(sample_size)
        #     sample_traversal_time = calculate_zone_workload_with_coefficients(sampled_points, lat_col, lon_col, points_coef, distance_coef) - (sample_size * 15 * points_coef)
            
        #     scaling_factor = (len(points) / sample_size) ** 0.5
        #     traversal_time = sample_traversal_time * scaling_factor*distance_coef
    
    total_workload = points_component + traversal_time
    
    return total_workload


def force_transfer_to_lowest_zone(lowest_zone, zones, zone_workloads, zone_neighbors, 
                                  df, zone_centroids, target_workload, target_points,
                                  lat_col, lon_col, recent_transfers, blacklisted_segments, blacklisted_paths):
    """
    Attempt to force a transfer to the lowest workload zone from any available neighbor.
    Modified to try with progressively smaller numbers of points.
    
    Returns:
    --------
    bool
        True if transfer was successful, False otherwise
    """
    improved = False
    
    # Get direct neighbors of the lowest zone
    neighbors = zone_neighbors.get(lowest_zone, [])
    
    if not neighbors:
        print(f"  Zone {lowest_zone} has no neighbors - cannot force transfer")
        return False
    
    # Sort neighbors by workload (highest to lowest)
    sorted_neighbors = sorted(neighbors, key=lambda z: zone_workloads.get(z, 0), reverse=True)
    
    for neighbor in sorted_neighbors:
        # Skip if this transfer is recently attempted
        if (neighbor, lowest_zone) in recent_transfers:
            print(f"  Skipping recent transfer from Zone {neighbor} to Zone {lowest_zone}")
            continue
            
        # Skip if segment is blacklisted
        if (neighbor, lowest_zone) in blacklisted_segments:
            print(f"  Skipping blacklisted segment from Zone {neighbor} to Zone {lowest_zone}")
            continue
            
        # Skip if neighbor zone is too small
        min_zone_size = max(5, int(target_points * 0.25))
        if len(zones[neighbor]) <= min_zone_size:
            print(f"  Zone {neighbor} too small to donate points ({len(zones[neighbor])} pts) - skipping")
            continue
            
        # Find boundary points suitable for transfer
        transfer_candidates = find_boundary_points_directional(
            zones[neighbor], neighbor, lowest_zone, 
            zone_centroids, lat_col, lon_col, df
        )
        
        if not transfer_candidates:
            print(f"  No suitable boundary points found between {neighbor} and {lowest_zone}")
            continue
            
        # Calculate workload difference
        workload_diff = zone_workloads[neighbor] - zone_workloads[lowest_zone]
        
        # If workload difference is too small, skip
        if workload_diff < target_workload * 0.05:
            print(f"  Workload difference between Zone {neighbor} and Zone {lowest_zone} too small - skipping")
            continue
            
        # Calculate points to transfer 
        estimated_points = max(5, min(
            int(workload_diff / 20),  # More aggressive transfer
            len(transfer_candidates),
            int(len(zones[neighbor]) * 0.4)
        ))
        
        # NEW: Try progressively smaller point counts
        point_counts_to_try = [
            estimated_points,
            max(4, int(estimated_points * 0.75)),
            max(2, int(estimated_points * 0.5)),
            1  # Try single point as last resort
        ]
        
        transfer_success = False
        
        # Try each point count until one succeeds
        for num_points in point_counts_to_try:
            if num_points > len(transfer_candidates):
                continue
                
            print(f"  FORCE TRANSFER: Attempting to transfer {num_points} points from Zone {neighbor} to Zone {lowest_zone}")
            
            # Get indices to transfer
            transfer_indices = [idx for idx, _, _ in transfer_candidates[:num_points]]
            
            # Verify connectivity preservation
            temp_df = df.copy()
            for idx in transfer_indices:
                temp_df.at[idx, 'zone_id'] = lowest_zone
                
            source_still_connected = check_zone_connectivity(temp_df, neighbor, lat_col, lon_col)
            target_still_connected = check_zone_connectivity(temp_df, lowest_zone, lat_col, lon_col)
            
            if source_still_connected and target_still_connected:
                # Commit transfer
                for idx in transfer_indices:
                    df.at[idx, 'zone_id'] = lowest_zone
                    
                # Update tracking
                recent_transfers.add((neighbor, lowest_zone))
                if len(recent_transfers) > 20:
                    recent_transfers = set(list(recent_transfers)[-20:])
                    
                print(f"  FORCE TRANSFER: Successfully transferred {len(transfer_indices)} points")
                transfer_success = True
                improved = True
                break
            else:
                if num_points > 1:
                    print(f"  FORCE TRANSFER: Transfer of {num_points} points would disconnect zones - trying fewer points")
                else:
                    print(f"  FORCE TRANSFER: Even transferring a single point would disconnect zones - skipping")
        
        # If transfer succeeded, we can return immediately
        if transfer_success:
            return True
            
        # Otherwise, try the next neighbor
    
    return improved


def transfer_points_between_zones(df, source_zone, target_zone, num_points, 
                                 zones, zone_workloads, zone_centroids,
                                 lat_col, lon_col, force_transfer=False):
    """Transfer points from source zone to target zone while maintaining connectivity."""
    source_df = zones[source_zone]
    target_df = zones[target_zone]
    
    if len(source_df) <= num_points + 5:  # Prevent emptying a zone
        print(f"    Source zone {source_zone} has too few points ({len(source_df)}) for requested transfer ({num_points})")
        return False, None
    
    # Calculate target centroid
    target_centroid = zone_centroids[target_zone]
    
    # Find points in source zone closest to target zone
    distances = haversine_distance(
        source_df[lat_col], source_df[lon_col],
        target_centroid[0], target_centroid[1]
    )
    
    source_df = source_df.copy()
    source_df['dist_to_target'] = distances
    
    # Sort by distance to target zone
    source_df = source_df.sort_values('dist_to_target')
    
    # Try to find transferable points
    transferred_indices = []
    potential_transfers = source_df.index[:min(num_points * 3, len(source_df))]
    
    for idx in potential_transfers:
        if len(transferred_indices) >= num_points:
            break
            
        # Create temporary zone assignments for connectivity check
        temp_df = df.copy()
        
        # Change zone of this point
        temp_df.loc[idx, 'zone_id'] = target_zone
        
        # If any points were already transferred, include them
        for prev_idx in transferred_indices:
            temp_df.loc[prev_idx, 'zone_id'] = target_zone
        
        # Check if both zones remain connected after this transfer
        source_still_connected = check_zone_connectivity(temp_df, source_zone, lat_col, lon_col)
        
        # In force transfer mode, we only care about source connectivity
        if force_transfer:
            if source_still_connected:
                transferred_indices.append(idx)
        else:
            # In normal mode, check both zones' connectivity
            target_still_connected = check_zone_connectivity(temp_df, target_zone, lat_col, lon_col)
            
            if source_still_connected and target_still_connected:
                transferred_indices.append(idx)
    
    # If we couldn't find enough transferable points
    if len(transferred_indices) < min(2, num_points):
        return False, None
    
    # Actually perform the transfer
    df.loc[transferred_indices, 'zone_id'] = target_zone
    
    # Update zone workloads
    # For simplicity, we'll recalculate later in the main function
    
    return True, transferred_indices

def force_direct_transfer(highest_zone, lowest_zone, zones, zone_workloads, df, 
                          zone_centroids, target_workload, lat_col, lon_col, zone_neighbors):
    """
    Attempt a direct emergency transfer from highest to lowest zone,
    trying progressively smaller numbers of points and more aggressive transfers.
    
    Returns:
    --------
    bool
        True if transfer was successful, False otherwise
    """
    # Check if we have a direct path via zone_neighbors 
    direct_connection = lowest_zone in zone_neighbors.get(highest_zone, [])
    
    # If not directly connected and highest and lowest are not the same zone
    if not direct_connection and highest_zone != lowest_zone:
        print(f"  EMERGENCY: Zones {highest_zone} and {lowest_zone} are not directly connected")
        return False
        
    # Skip if highest zone is too small
    min_zone_size = max(10, int(len(df) / len(zone_workloads) * 0.25))
    if len(zones[highest_zone]) <= min_zone_size:
        print(f"  EMERGENCY: Zone {highest_zone} too small to donate points ({len(zones[highest_zone])} pts)")
        return False
        
    # Find boundary points - use a more aggressive approach
    print(f"  EMERGENCY: Finding viable points to transfer from Zone {highest_zone} to Zone {lowest_zone}")
    
    # Try with normal boundary finding first
    transfer_candidates = find_boundary_points_directional(
        zones[highest_zone], highest_zone, lowest_zone, 
        zone_centroids, lat_col, lon_col, df
    )
    
    # If normal approach doesn't work, try with more aggressive settings
    if not transfer_candidates:
        print(f"  EMERGENCY: No boundary points found with standard method - trying aggressive approach")
        
        # Fall back to a simpler approach - just get points closest to target zone
        if len(zones[highest_zone]) > 0:
            source_points = zones[highest_zone]
            target_centroid = zone_centroids[lowest_zone]
            
            # Calculate distances to target centroid
            distances = []
            for idx, row in source_points.iterrows():
                lat, lon = row[lat_col], row[lon_col]
                dist = ((lat - target_centroid[0])**2 + (lon - target_centroid[1])**2)**0.5
                distances.append((idx, dist, row))
                
            # Sort by distance to target
            distances.sort(key=lambda x: x[1])
            
            # Use up to 20% of points as candidates
            max_candidates = max(5, min(int(len(source_points) * 0.20), 100))
            transfer_candidates = distances[:max_candidates]
    
    # If still no candidates, we can't transfer
    if not transfer_candidates:
        print(f"  EMERGENCY: No viable points found between Zone {highest_zone} and Zone {lowest_zone}")
        return False
        
    # Calculate points to transfer - be more aggressive for emergency transfers
    workload_diff = zone_workloads[highest_zone] - zone_workloads[lowest_zone]
    estimated_points = max(3, min(
        int(workload_diff / 10),  # More aggressive than normal transfers
        len(transfer_candidates),
        int(len(zones[highest_zone]) * 0.30)  # Allow up to 30% transfer in emergency
    ))
    
    # Try progressively smaller transfers
    point_counts_to_try = [
        estimated_points,
        max(5, int(estimated_points * 0.66)),
        max(3, int(estimated_points * 0.5)),
        2,
        1  # Last resort: try just 1 point
    ]
    
    for num_points in point_counts_to_try:
        if num_points > len(transfer_candidates):
            continue
            
        print(f"  EMERGENCY: Attempting to transfer {num_points} points between zones")
        
        # Get indices to transfer
        transfer_indices = [idx for idx, _, _ in transfer_candidates[:num_points]]
        
        # Check for connectivity preservation
        temp_df = df.copy()
        for idx in transfer_indices:
            temp_df.at[idx, 'zone_id'] = lowest_zone
            
        # Verify zones remain connected
        source_still_connected = check_zone_connectivity(temp_df, highest_zone, lat_col, lon_col)
        target_still_connected = check_zone_connectivity(temp_df, lowest_zone, lat_col, lon_col)
        
        if not source_still_connected:
            print(f"  EMERGENCY: Transfer would disconnect source Zone {highest_zone}")
            
            # For very small counts, try individual points
            if num_points <= 3:
                # Try points one by one
                for single_idx, _, _ in transfer_candidates[:num_points]:
                    single_temp_df = df.copy()
                    single_temp_df.at[single_idx, 'zone_id'] = lowest_zone
                    
                    if check_zone_connectivity(single_temp_df, highest_zone, lat_col, lon_col) and \
                       check_zone_connectivity(single_temp_df, lowest_zone, lat_col, lon_col):
                        print(f"  EMERGENCY: Found single valid point to transfer")
                        df.at[single_idx, 'zone_id'] = lowest_zone
                        return True
                        
                print(f"  EMERGENCY: No single points can be transferred without disconnecting zones")
            
            continue
            
        if not target_still_connected:
            print(f"  EMERGENCY: Transfer would disconnect target Zone {lowest_zone}")
            continue
            
        # Perform transfer
        for idx in transfer_indices:
            df.at[idx, 'zone_id'] = lowest_zone
            
        print(f"  EMERGENCY: Successfully transferred {len(transfer_indices)} points")
        return True
        
    print(f"  EMERGENCY: Could not find any viable transfer configuration")
    return False




def evaluate_density_improvement(df, zone_id, neighbor_id, furthest_points, closest_points, 
                                zones, zone_workloads, zone_centroids,
                                lat_col="Latitude", lon_col="Longitude"):
    """
    Evaluate the improvement in density by swapping points between zones.
    Returns:
    - density_improvement: improvement in total distance to centroids (negative means better)
    - feasible: whether the swap is feasible
    """
    zone_df = zones[zone_id].copy()
    neighbor_df = zones[neighbor_id].copy()
    
    if len(furthest_points) == 0 or len(closest_points) == 0:
        return 0, False
    
    # Get points to swap
    to_give = df.loc[furthest_points].copy()
    to_take = df.loc[closest_points].copy()
    
    # Calculate current total distances
    current_zone_distance = calculate_distance_to_centroid(zone_df, zone_centroids[zone_id], lat_col, lon_col)
    current_neighbor_distance = calculate_distance_to_centroid(neighbor_df, zone_centroids[neighbor_id], lat_col, lon_col)
    current_total_distance = current_zone_distance + current_neighbor_distance
    
    # Simulate the swap
    new_zone_df = pd.concat([zone_df.drop(furthest_points), to_take], ignore_index=True)
    new_neighbor_df = pd.concat([neighbor_df.drop(closest_points), to_give], ignore_index=True)
    
    # Calculate new centroids
    if len(new_zone_df) > 0:
        new_zone_centroid = np.mean(new_zone_df[[lat_col, lon_col]].values, axis=0)
    else:
        new_zone_centroid = zone_centroids[zone_id]
        
    if len(new_neighbor_df) > 0:
        new_neighbor_centroid = np.mean(new_neighbor_df[[lat_col, lon_col]].values, axis=0)
    else:
        new_neighbor_centroid = zone_centroids[neighbor_id]
    
    # Calculate new total distances
    new_zone_distance = calculate_distance_to_centroid(new_zone_df, new_zone_centroid, lat_col, lon_col)
    new_neighbor_distance = calculate_distance_to_centroid(new_neighbor_df, new_neighbor_centroid, lat_col, lon_col)
    new_total_distance = new_zone_distance + new_neighbor_distance
    
    # Calculate improvement (negative means better)
    density_improvement = new_total_distance - current_total_distance
    
    # Check if both zones improve or if at least the total improves significantly
    zone_improvement = new_zone_distance < current_zone_distance
    neighbor_improvement = new_neighbor_distance < current_neighbor_distance
    total_improvement = density_improvement < 0
    
    # Swap is feasible if total distance improves and either both zones improve 
    # or one improves significantly while the other doesn't get much worse
    feasible = total_improvement and (
        (zone_improvement and neighbor_improvement) or
        (zone_improvement and (new_neighbor_distance - current_neighbor_distance) < current_neighbor_distance * 0.05) or
        (neighbor_improvement and (new_zone_distance - current_zone_distance) < current_zone_distance * 0.05)
    )
    
    return density_improvement, feasible

def identify_swap_candidates(zone_df, neighbor_df, zone_centroid, neighbor_centroid, 
                             lat_col="Latitude", lon_col="Longitude", percentage=0.2):
    """
    Identify points to swap between zones to improve density.
    Returns:
    - Points from current zone that are furthest from its centroid (candidates to give away)
    - Points from neighbor zone that are closer to current zone's centroid (candidates to receive)
    """
    if len(zone_df) <= 3 or len(neighbor_df) <= 3:
        return [], []
    
    # Calculate distances to current zone centroid for all points in the zone
    zone_distances = []
    for idx, row in zone_df.iterrows():
        point = np.array([row[lat_col], row[lon_col]])
        dist = np.sqrt(np.sum((point - zone_centroid)**2))
        zone_distances.append((idx, dist))
    
    # Sort by distance (furthest first)
    zone_distances.sort(key=lambda x: x[1], reverse=True)
    
    # Calculate distances to current zone centroid for all points in the neighbor zone
    neighbor_to_current_distances = []
    for idx, row in neighbor_df.iterrows():
        point = np.array([row[lat_col], row[lon_col]])
        dist_to_current = np.sqrt(np.sum((point - zone_centroid)**2))
        dist_to_neighbor = np.sqrt(np.sum((point - neighbor_centroid)**2))
        # We want points that are closer to current centroid than they are to their own
        if dist_to_current < dist_to_neighbor:
            neighbor_to_current_distances.append((idx, dist_to_current))
    
    # Sort by distance (closest first)
    neighbor_to_current_distances.sort(key=lambda x: x[1])
    
    # Take top percentage of points
    num_zone_points = max(1, int(len(zone_df) * percentage))
    num_neighbor_points = max(1, int(len(neighbor_df) * percentage))
    
    # Limit the number of furthest points to the number of available neighbor points
    if len(neighbor_to_current_distances) < num_zone_points:
        num_zone_points = len(neighbor_to_current_distances)
    
    # Ensure we don't take too many neighbor points
    num_neighbor_points = min(num_neighbor_points, len(neighbor_to_current_distances))
    
    furthest_points = [idx for idx, _ in zone_distances[:num_zone_points]]
    closest_points = [idx for idx, _ in neighbor_to_current_distances[:num_neighbor_points]]
    
    return furthest_points, closest_points

def calculate_distance_to_centroid(df, centroid, lat_col="Latitude", lon_col="Longitude"):
    """
    Calculate the sum of distances from all points to the centroid.
    Lower score means higher density (better).
    """
    if len(df) <= 1:
        return 0
    
    coords = df[[lat_col, lon_col]].values
    
    # Calculate distances to centroid
    distances = np.sqrt(np.sum((coords - centroid)**2, axis=1))
    
    # Return the sum of distances (total distance to centroid)
    return np.sum(distances)

def optimize_zone_density(df, zones, zone_workloads, zone_neighbors, zone_centroids, 
                           lat_col="Latitude", lon_col="Longitude", 
                           swap_percentage=0.1, max_swaps=5,
                           workload_threshold=0.15, points_coef = 1, distance_coef = 1):
    """
    Improved optimization of zone density by swapping boundary points between neighboring zones,
    considering workload differences and minimizing distance to centroid.
    Only performs swaps/transfers when they result in positive improvements.
    """
    print("\nPerforming improved density optimization by swapping boundary points...")
    num_zones = len(zones)
    swap_count = 0
    successful_swaps = 0
    
    # Track zone pairs that have already been considered
    considered_pairs = set()
    
    # Calculate average workload
    avg_workload = sum(zone_workloads.values()) / max(1, len(zone_workloads))
    
    # Track original density scores
    original_distances = {i: calculate_distance_to_centroid(zones[i], zone_centroids[i], lat_col, lon_col) 
                         for i in range(num_zones)}
    
    # Make a copy to track final distances
    final_distances = original_distances.copy()
    
    print("Original zone total distances to centroid:")
    for zone_id, distance in original_distances.items():
        print(f"Zone {zone_id}: {distance:.4f}")
    
    # Consider each zone and its neighbors
    for zone_id in range(num_zones):
        # Sort neighbors by workload difference (highest difference first)
        sorted_neighbors = []
        for neighbor_id in zone_neighbors.get(zone_id, []):
            workload_diff = zone_workloads[zone_id] - zone_workloads[neighbor_id]
            sorted_neighbors.append((neighbor_id, workload_diff))
        
        sorted_neighbors.sort(key=lambda x: abs(x[1]), reverse=True)
        
        for neighbor_id, workload_diff in sorted_neighbors:
            # Skip if already considered this pair
            if (zone_id, neighbor_id) in considered_pairs or (neighbor_id, zone_id) in considered_pairs:
                continue
            
            considered_pairs.add((zone_id, neighbor_id))
            
            # Determine if this should be a swap or one-way transfer based on workload difference
            workload_ratio = zone_workloads[zone_id] / max(0.001, zone_workloads[neighbor_id])
            relative_diff = abs(workload_diff) / avg_workload
            
            print(f"Considering zone {zone_id} (workload: {zone_workloads[zone_id]:.2f}) and "
                 f"zone {neighbor_id} (workload: {zone_workloads[neighbor_id]:.2f})")
            
            # If workload difference is significant, do a one-way transfer from higher to lower
            if relative_diff > workload_threshold:
                if workload_diff > 0:  # Current zone has higher workload
                    print(f"  Zone {zone_id} has significantly higher workload than Zone {neighbor_id}")
                    print(f"  Attempting one-way transfer from Zone {zone_id} to Zone {neighbor_id}")
                    
                    # Find points furthest from current zone centroid
                    transfer_candidates = []
                    for idx, row in zones[zone_id].iterrows():
                        point = np.array([row[lat_col], row[lon_col]])
                        dist = np.sqrt(np.sum((point - zone_centroids[zone_id])**2))
                        transfer_candidates.append((idx, dist))
                    
                    # Sort by distance (furthest first)
                    transfer_candidates.sort(key=lambda x: x[1], reverse=True)
                    
                    # Take a percentage of points to transfer
                    num_transfer_points = max(1, int(len(zones[zone_id]) * swap_percentage))
                    transfer_points = [idx for idx, _ in transfer_candidates[:num_transfer_points]]
                    
                    # Evaluate the transfer
                    if len(transfer_points) > 0:
                        # Store original distances for these zones before the transfer
                        original_zone_distance = calculate_distance_to_centroid(zones[zone_id], zone_centroids[zone_id], lat_col, lon_col)
                        original_neighbor_distance = calculate_distance_to_centroid(zones[neighbor_id], zone_centroids[neighbor_id], lat_col, lon_col)
                        
                        # Simulate the transfer
                        temp_zone_df = zones[zone_id].drop(transfer_points)
                        temp_neighbor_df = pd.concat([zones[neighbor_id], df.loc[transfer_points]], ignore_index=True)
                        
                        # Calculate new centroids
                        if len(temp_zone_df) > 0:
                            new_zone_centroid = np.mean(temp_zone_df[[lat_col, lon_col]].values, axis=0)
                        else:
                            new_zone_centroid = zone_centroids[zone_id]
                            
                        if len(temp_neighbor_df) > 0:
                            new_neighbor_centroid = np.mean(temp_neighbor_df[[lat_col, lon_col]].values, axis=0)
                        else:
                            new_neighbor_centroid = zone_centroids[neighbor_id]
                        
                        # Calculate new total distances
                        new_zone_distance = calculate_distance_to_centroid(temp_zone_df, new_zone_centroid, lat_col, lon_col)
                        new_neighbor_distance = calculate_distance_to_centroid(temp_neighbor_df, new_neighbor_centroid, lat_col, lon_col)
                        
                        # Calculate overall improvement - ADDED THIS CHECK
                        total_original_distance = original_zone_distance + original_neighbor_distance
                        total_new_distance = new_zone_distance + new_neighbor_distance
                        net_improvement = total_original_distance - total_new_distance
                        
                        # Check if transfer improves density overall
                        density_improves = net_improvement > 0
                        
                        if density_improves:  # Only proceed if there's positive improvement
                            print(f"  Transferring {len(transfer_points)} points from Zone {zone_id} to Zone {neighbor_id}")
                            print(f"  Zone {zone_id} density change: {original_zone_distance:.4f} -> {new_zone_distance:.4f}")
                            print(f"  Zone {neighbor_id} density change: {original_neighbor_distance:.4f} -> {new_neighbor_distance:.4f}")
                            print(f"  Net improvement: {net_improvement:.4f}")
                            
                            # Perform the transfer
                            df.loc[transfer_points, 'zone_id'] = neighbor_id
                            
                            # Update zones
                            zones[zone_id] = df[df['zone_id'] == zone_id]
                            zones[neighbor_id] = df[df['zone_id'] == neighbor_id]
                            
                            # Update workloads
                            try:
                                zone_workloads[zone_id] = calculate_zone_workload_with_coefficients(zones[zone_id], lat_col, lon_col, points_coef,distance_coef)
                                zone_workloads[neighbor_id] = calculate_zone_workload_with_coefficients(zones[neighbor_id], lat_col, lon_col, points_coef,distance_coef)
                                
                                # Cap extreme workloads
                                if zone_workloads[zone_id] > 100000:
                                    zone_workloads[zone_id] = 100000
                                if zone_workloads[neighbor_id] > 100000:
                                    zone_workloads[neighbor_id] = 100000
                            except Exception as e:
                                print(f"Error updating workloads: {e}")
                            
                            # Update centroids
                            zone_centroids[zone_id] = new_zone_centroid
                            zone_centroids[neighbor_id] = new_neighbor_centroid
                            
                            # Update final distances
                            final_distances[zone_id] = new_zone_distance
                            final_distances[neighbor_id] = new_neighbor_distance
                            
                            swap_count += 1
                            successful_swaps += 1
                        else:
                            print(f"  Transfer rejected: No overall density improvement (change: {net_improvement:.4f})")
                        
                elif workload_diff < 0:  # Neighbor zone has higher workload
                    print(f"  Zone {neighbor_id} has significantly higher workload than Zone {zone_id}")
                    print(f"  Attempting one-way transfer from Zone {neighbor_id} to Zone {zone_id}")
                    
                    # Find points furthest from neighbor zone centroid
                    transfer_candidates = []
                    for idx, row in zones[neighbor_id].iterrows():
                        point = np.array([row[lat_col], row[lon_col]])
                        dist = np.sqrt(np.sum((point - zone_centroids[neighbor_id])**2))
                        transfer_candidates.append((idx, dist))
                    
                    # Sort by distance (furthest first)
                    transfer_candidates.sort(key=lambda x: x[1], reverse=True)
                    
                    # Take a percentage of points to transfer
                    num_transfer_points = max(1, int(len(zones[neighbor_id]) * swap_percentage))
                    transfer_points = [idx for idx, _ in transfer_candidates[:num_transfer_points]]
                    
                    # Evaluate the transfer
                    if len(transfer_points) > 0:
                        # Store original distances for these zones before the transfer
                        original_zone_distance = calculate_distance_to_centroid(zones[zone_id], zone_centroids[zone_id], lat_col, lon_col)
                        original_neighbor_distance = calculate_distance_to_centroid(zones[neighbor_id], zone_centroids[neighbor_id], lat_col, lon_col)
                        
                        # Simulate the transfer
                        temp_neighbor_df = zones[neighbor_id].drop(transfer_points)
                        temp_zone_df = pd.concat([zones[zone_id], df.loc[transfer_points]], ignore_index=True)
                        
                        # Calculate new centroids
                        if len(temp_zone_df) > 0:
                            new_zone_centroid = np.mean(temp_zone_df[[lat_col, lon_col]].values, axis=0)
                        else:
                            new_zone_centroid = zone_centroids[zone_id]
                            
                        if len(temp_neighbor_df) > 0:
                            new_neighbor_centroid = np.mean(temp_neighbor_df[[lat_col, lon_col]].values, axis=0)
                        else:
                            new_neighbor_centroid = zone_centroids[neighbor_id]
                        
                        # Calculate new total distances
                        new_neighbor_distance = calculate_distance_to_centroid(temp_neighbor_df, new_neighbor_centroid, lat_col, lon_col)
                        new_zone_distance = calculate_distance_to_centroid(temp_zone_df, new_zone_centroid, lat_col, lon_col)
                        
                        # Calculate overall improvement - ADDED THIS CHECK
                        total_original_distance = original_zone_distance + original_neighbor_distance
                        total_new_distance = new_zone_distance + new_neighbor_distance
                        net_improvement = total_original_distance - total_new_distance
                        
                        # Check if transfer improves density overall
                        density_improves = net_improvement > 0
                        
                        if density_improves:  # Only proceed if there's positive improvement
                            print(f"  Transferring {len(transfer_points)} points from Zone {neighbor_id} to Zone {zone_id}")
                            print(f"  Zone {neighbor_id} density change: {original_neighbor_distance:.4f} -> {new_neighbor_distance:.4f}")
                            print(f"  Zone {zone_id} density change: {original_zone_distance:.4f} -> {new_zone_distance:.4f}")
                            print(f"  Net improvement: {net_improvement:.4f}")
                            
                            # Perform the transfer
                            df.loc[transfer_points, 'zone_id'] = zone_id
                            
                            # Update zones
                            zones[zone_id] = df[df['zone_id'] == zone_id]
                            zones[neighbor_id] = df[df['zone_id'] == neighbor_id]
                            
                            # Update workloads
                            try:
                                zone_workloads[zone_id] = calculate_zone_workload_with_coefficients(zones[zone_id], lat_col, lon_col, points_coef,distance_coef)
                                zone_workloads[neighbor_id] = calculate_zone_workload_with_coefficients(zones[neighbor_id], lat_col, lon_col, points_coef,distance_coef)
                                
                                # Cap extreme workloads
                                if zone_workloads[zone_id] > 100000:
                                    zone_workloads[zone_id] = 100000
                                if zone_workloads[neighbor_id] > 100000:
                                    zone_workloads[neighbor_id] = 100000
                            except Exception as e:
                                print(f"Error updating workloads: {e}")
                            
                            # Update centroids
                            zone_centroids[zone_id] = new_zone_centroid
                            zone_centroids[neighbor_id] = new_neighbor_centroid
                            
                            # Update final distances
                            final_distances[zone_id] = new_zone_distance
                            final_distances[neighbor_id] = new_neighbor_distance
                            
                            swap_count += 1
                            successful_swaps += 1
                        else:
                            print(f"  Transfer rejected: No overall density improvement (change: {net_improvement:.4f})")
            
            # If workloads are similar, perform bi-directional swap
            else:
                print(f"  Zones have similar workloads (diff: {workload_diff:.2f}) - attempting bi-directional swap")
                
                # Store original distances for these zones before the swap
                original_zone_distance = calculate_distance_to_centroid(zones[zone_id], zone_centroids[zone_id], lat_col, lon_col)
                original_neighbor_distance = calculate_distance_to_centroid(zones[neighbor_id], zone_centroids[neighbor_id], lat_col, lon_col)
                
                # Identify swap candidates
                furthest_points, closest_points = identify_swap_candidates(
                    zones[zone_id], zones[neighbor_id], 
                    zone_centroids[zone_id], zone_centroids[neighbor_id],
                    lat_col, lon_col, percentage=swap_percentage
                )
                
                # Evaluate potential benefit
                density_improvement, feasible = evaluate_density_improvement(
                    df, zone_id, neighbor_id, furthest_points, closest_points,
                    zones, zone_workloads, zone_centroids, lat_col, lon_col
                )
                
                # Only perform swap if it provides a positive improvement - MODIFIED THIS CHECK
                if feasible and density_improvement > 0 and len(furthest_points) > 0 and len(closest_points) > 0:
                    # Simulate the swap
                    to_give = df.loc[furthest_points].copy()
                    to_take = df.loc[closest_points].copy()
                    
                    new_zone_df = pd.concat([zones[zone_id].drop(furthest_points), to_take], ignore_index=True)
                    new_neighbor_df = pd.concat([zones[neighbor_id].drop(closest_points), to_give], ignore_index=True)
                    
                    # Calculate new centroids
                    if len(new_zone_df) > 0:
                        new_zone_centroid = np.mean(new_zone_df[[lat_col, lon_col]].values, axis=0)
                    else:
                        new_zone_centroid = zone_centroids[zone_id]
                        
                    if len(new_neighbor_df) > 0:
                        new_neighbor_centroid = np.mean(new_neighbor_df[[lat_col, lon_col]].values, axis=0)
                    else:
                        new_neighbor_centroid = zone_centroids[neighbor_id]
                    
                    # Calculate new distances
                    new_zone_distance = calculate_distance_to_centroid(new_zone_df, new_zone_centroid, lat_col, lon_col)
                    new_neighbor_distance = calculate_distance_to_centroid(new_neighbor_df, new_neighbor_centroid, lat_col, lon_col)
                    
                    print(f"  Swapping {len(furthest_points)} points from Zone {zone_id} with "
                         f"{len(closest_points)} points from Zone {neighbor_id}")
                    print(f"  Zone {zone_id} density change: {original_zone_distance:.4f} -> {new_zone_distance:.4f}")
                    print(f"  Zone {neighbor_id} density change: {original_neighbor_distance:.4f} -> {new_neighbor_distance:.4f}")
                    print(f"  Density improvement: {density_improvement:.4f}")
                    
                    # Perform the swap
                    df.loc[furthest_points, 'zone_id'] = neighbor_id
                    df.loc[closest_points, 'zone_id'] = zone_id
                    
                    # Update zones
                    zones[zone_id] = df[df['zone_id'] == zone_id]
                    zones[neighbor_id] = df[df['zone_id'] == neighbor_id]
                    
                    # Update workloads
                    try:
                        zone_workloads[zone_id] = calculate_zone_workload_with_coefficients(zones[zone_id], lat_col, lon_col, points_coef,distance_coef)
                        zone_workloads[neighbor_id] = calculate_zone_workload_with_coefficients(zones[neighbor_id], lat_col, lon_col, points_coef,distance_coef)
                        
                        # Cap extreme workloads
                        if zone_workloads[zone_id] > 100000:
                            zone_workloads[zone_id] = 100000
                        if zone_workloads[neighbor_id] > 100000:
                            zone_workloads[neighbor_id] = 100000
                    except Exception as e:
                        print(f"Error updating workloads after swap: {e}")
                    
                    # Update centroids
                    zone_centroids[zone_id] = new_zone_centroid
                    zone_centroids[neighbor_id] = new_neighbor_centroid
                    
                    # Update final distances
                    final_distances[zone_id] = new_zone_distance
                    final_distances[neighbor_id] = new_neighbor_distance
                    
                    swap_count += 1
                    successful_swaps += 1
                else:
                    if not feasible:
                        print("  Swap rejected: Not feasible")
                    elif density_improvement <= 0:
                        print(f"  Swap rejected: No density improvement (value: {density_improvement:.4f})")
                    elif len(furthest_points) == 0 or len(closest_points) == 0:
                        print("  Swap rejected: No suitable points to swap")
            
            # Stop if we've reached max swaps
            if swap_count >= max_swaps:
                break
        
        if swap_count >= max_swaps:
            break
    
    # After all swaps are done, recalculate the final distances for all zones
    # This ensures we report accurate before/after numbers
    for zone_id in range(num_zones):
        # Ensure we have the most up-to-date zones and centroids
        zones[zone_id] = df[df['zone_id'] == zone_id]
        if len(zones[zone_id]) > 0:
            zone_centroids[zone_id] = np.mean(zones[zone_id][[lat_col, lon_col]].values, axis=0)
            final_distances[zone_id] = calculate_distance_to_centroid(zones[zone_id], zone_centroids[zone_id], lat_col, lon_col)
    
    print(f"\nDensity optimization completed with {successful_swaps} successful swaps/transfers.")
    print("Density improvements (total distance to centroid):")
    
    total_improvement = 0
    for zone_id in range(num_zones):
        improvement = original_distances[zone_id] - final_distances[zone_id]
        percent_improvement = (improvement / original_distances[zone_id] * 100) if original_distances[zone_id] > 0 else 0
        print(f"Zone {zone_id}: {original_distances[zone_id]:.2f} -> {final_distances[zone_id]:.2f} "
             f"(change: {improvement:+.2f}, {percent_improvement:+.2f}%)")
        total_improvement += improvement
    
    print(f"Total density improvement: {total_improvement:.2f}")
    
    return df, zones, zone_workloads, zone_centroids

def create_balanced_zones(df, num_zones, lat_col="Latitude", lon_col="Longitude", max_iterations=50, points_coef = 1, distance_coef = 1):
    print(f"Creating {num_zones} balanced zones using directional transfer approach with density optimization...")
    
    # Initial clustering with KMeans
    coords = df[[lat_col, lon_col]].values
    kmeans = KMeans(n_clusters=num_zones, random_state=42, n_init=10)
    df['zone_id'] = kmeans.fit_predict(coords)
    
    # Initialize zones and metrics
    zones = {i: df[df['zone_id'] == i] for i in range(num_zones)}
    zone_points = {i: len(zones[i]) for i in range(num_zones)}
    
    # Calculate initial workloads
    zone_workloads = {}
    for i in range(num_zones):
        try:
            workload = calculate_zone_workload_with_coefficients(zones[i], lat_col, lon_col, points_coef, distance_coef)
            if workload > 100000:
                print(f"Warning: Capping extreme workload in Zone {i}: {workload:.2f} min -> 100000 min")
                workload = 100000
            zone_workloads[i] = workload
        except Exception as e:
            print(f"Error calculating workload for Zone {i}: {e}")
            zone_workloads[i] = len(zones[i]) * 15
    
    print("Initial zone distribution:")
    for i in range(num_zones):
        print(f"Zone {i}: {zone_points[i]} points, {zone_workloads[i]:.2f} min workload")
    
    # Calculate targets
    total_points = len(df)
    target_points = total_points / num_zones
    total_workload = sum(zone_workloads.values())
    target_workload = total_workload / num_zones
    
    print(f"Target points per zone: {target_points:.2f}")
    print(f"Target workload per zone: {target_workload:.2f} min")
    
    # Calculate zone centroids
    zone_centroids = {}
    for zone_id, zone_df in zones.items():
        if len(zone_df) > 0:
            zone_centroids[zone_id] = np.mean(zone_df[[lat_col, lon_col]].values, axis=0)
    
    # Build zone connectivity graph
    zone_neighbors = build_zone_connectivity_graph(zones, zone_centroids, lat_col, lon_col)
    
    # Track metrics for stopping criteria
    previous_workload_std = float('inf')
    stagnation_counter = 0
    no_improvement_counter = 0
    recent_transfers = set()
    
    # Track blacklisted paths and segments
    blacklisted_paths = set()
    blacklisted_segments = set()
    
    # Track blacklisted zone pairs for workload transfer
    blacklisted_transfers = set()
    
    # Counter for forced transfer attempts
    force_transfer_counter = 0
    max_force_transfers = 10  # Maximum number of forced transfers
    # Counter for emergency reset
    emergency_reset_counter = 0
    emergency_transfer_counter = 0
    max_emergency_transfers = 8  # Maximum number of emergency transfers
    
    # Track density optimization iterations
    last_density_optimization = -1
    density_optimization_frequency = 10  # Run density optimization every X iterations
    
    # Main optimization loop

    for iteration in range(max_iterations):
        if stagnation_counter > 10:
            break
        improved = False
        
        # Identify workload-sorted zones
        sorted_zones = sorted(zone_workloads.items(), key=lambda x: x[1], reverse=True)
        highest_zone = sorted_zones[0][0]
        lowest_zone = sorted_zones[-1][0]
        
        highest_workload = zone_workloads[highest_zone]
        lowest_workload = zone_workloads[lowest_zone]
        
        # Check if balance is already acceptable
        workload_ratio = highest_workload / max(0.001, lowest_workload)
        workload_std = np.std(list(zone_workloads.values()))
        workload_cv = workload_std / (sum(zone_workloads.values()) / len(zone_workloads))
        
        if workload_cv < 0.04 and iteration > 10:
            print(f"Excellent balance achieved (CV: {workload_cv:.3f}) after {iteration} iterations")
            break
        
        print(f"Iteration {iteration}: Highest Zone {highest_zone} ({highest_workload:.2f} min), "
              f"Lowest Zone {lowest_zone} ({lowest_workload:.2f} min), Ratio: {workload_ratio:.2f}")
        
        # Early stopping if zones are too close in workload
        if highest_workload - lowest_workload < target_workload * 0.05:
            print(f"Zones are balanced within 5% of target after {iteration} iterations")
            break
        
        # Check if it's time for density optimization
        if iteration - last_density_optimization >= density_optimization_frequency and iteration > 15:
            print(f"\nPerforming periodic density optimization at iteration {iteration}")
            max_swaps_for_periodic = 3 if iteration < 60 else 5  # Allow more swaps in later iterations
            
            df, zones, zone_workloads, zone_centroids = optimize_zone_density(
                df, zones, zone_workloads, zone_neighbors, zone_centroids, lat_col, lon_col, 
                swap_percentage=0.08, max_swaps=max_swaps_for_periodic, points_coef=points_coef, distance_coef=distance_coef
            )
            
            # Update zone neighbors after density optimization
            zone_neighbors = build_zone_connectivity_graph(zones, zone_centroids, lat_col, lon_col)
            last_density_optimization = iteration
            
            # Check balance again after density optimization
            workload_std = np.std(list(zone_workloads.values()))
            if workload_std < previous_workload_std:
                no_improvement_counter = max(0, no_improvement_counter - 1)
                
            previous_workload_std = workload_std
            
            # Continue to next iteration
            continue
        
        # Check if we're stuck and need an emergency reset
        if no_improvement_counter >=5 and emergency_transfer_counter < max_emergency_transfers:
            emergency_reset_counter += 1
            emergency_transfer_counter += 1
            print(f"\n=== EMERGENCY RESET #{emergency_transfer_counter} ===")
            print(f"Algorithm stuck for {no_improvement_counter} iterations - resetting all blacklists")
            
            # Reset all blacklists
            blacklisted_paths = set()
            blacklisted_segments = set()
            blacklisted_transfers = set()
            recent_transfers = set()
            
            print("All blacklists have been cleared")
            
            # Attempt direct transfer from highest to lowest zone
            direct_transfer_success = force_direct_transfer(
                highest_zone, lowest_zone, zones, zone_workloads, df, 
                zone_centroids, target_workload, lat_col, lon_col, zone_neighbors
            )
            
            if direct_transfer_success:
                print("Emergency transfer successful - continuing with optimization")
                improved = True
                no_improvement_counter = max(0, no_improvement_counter - 5)  # Reduce counter but don't reset completely
                
                # Update all data structures after emergency transfer
                zones = {i: df[df['zone_id'] == i] for i in range(num_zones)}
                zone_points = {i: len(zones[i]) for i in range(num_zones)}
                
                # Recalculate workloads
                for i in range(num_zones):
                    if len(zones[i]) > 0:
                        try:
                            workload = calculate_zone_workload_with_coefficients(zones[i], lat_col, lon_col, points_coef, distance_coef)
                            if workload > 100000:
                                workload = 100000
                            zone_workloads[i] = workload
                        except Exception as e:
                            print(f"Error calculating workload for Zone {i}: {e}")
                            zone_workloads[i] = len(zones[i]) * 15
                
                # Update centroids and connectivity
                for zone_id, zone_df in zones.items():
                    if len(zone_df) > 0:
                        zone_centroids[zone_id] = np.mean(zone_df[[lat_col, lon_col]].values, axis=0)
                
                # Rebuild connectivity graph after emergency transfer
                zone_neighbors = build_zone_connectivity_graph(zones, zone_centroids, lat_col, lon_col)
                
                # Continue to next iteration
                continue
            else:
                print("Emergency transfer failed - continuing with normal optimization")
        
        # Check if it's time to try forced transfers (when stuck for too long)
        if no_improvement_counter >= 3 and force_transfer_counter < max_force_transfers and emergency_transfer_counter < max_emergency_transfers:
            force_transfer_counter += 1
            print(f"  No improvement for {no_improvement_counter} iterations - attempting FORCED transfer (attempt {force_transfer_counter})")
            
            # Try to force transfer to lowest zone from any neighbor
            forced_improved = force_transfer_to_lowest_zone(
                lowest_zone, zones, zone_workloads, zone_neighbors, 
                df, zone_centroids, target_workload, target_points,
                lat_col, lon_col, recent_transfers, blacklisted_segments, blacklisted_paths
            )
            
            if forced_improved:
                improved = True
                no_improvement_counter = max(0, no_improvement_counter - 3)  # Reduce counter but don't reset completely
                print(f"  FORCE TRANSFER: Successfully improved balance through forced transfer")
                
                # Reset zone data after forced transfer
                zones = {i: df[df['zone_id'] == i] for i in range(num_zones)}
                zone_points = {i: len(zones[i]) for i in range(num_zones)}
                
                # Update workloads after forced transfer
                for i in range(num_zones):
                    if len(zones[i]) > 0:
                        try:
                            workload = calculate_zone_workload_with_coefficients(zones[i], lat_col, lon_col, points_coef, distance_coef)
                            if workload > 100000:
                                workload = 100000
                            zone_workloads[i] = workload
                        except Exception as e:
                            print(f"Error calculating workload for Zone {i}: {e}")
                            zone_workloads[i] = len(zones[i]) * 15
                
                # Update centroids
                for zone_id, zone_df in zones.items():
                    if len(zone_df) > 0:
                        zone_centroids[zone_id] = np.mean(zone_df[[lat_col, lon_col]].values, axis=0)
                
                # Update connectivity graph after forced transfer
                zone_neighbors = build_zone_connectivity_graph(zones, zone_centroids, lat_col, lon_col)
                
                # Continue to next iteration
                continue
        
        # Regular transfer logic
        # Check if transfer between highest and lowest is blacklisted
        primary_transfer_blacklisted = (highest_zone, lowest_zone) in blacklisted_transfers
        
        if primary_transfer_blacklisted:
            print(f"  Transfer from Zone {highest_zone} to Zone {lowest_zone} is blacklisted, trying alternative combinations")
            transfer_attempted = False
            
            # Try to find an alternative donor for the lowest workload zone
            for donor_idx, (donor_zone, donor_workload) in enumerate(sorted_zones[:min(5, len(sorted_zones))]):
                if donor_zone == highest_zone:
                    continue
                
                if (donor_zone, lowest_zone) in blacklisted_transfers:
                    print(f"  Transfer from Zone {donor_zone} to Zone {lowest_zone} is also blacklisted")
                    continue
                
                print(f"  Attempting transfer from alternative high workload Zone {donor_zone} to lowest Zone {lowest_zone}")
                
                # Find path from alternative donor to lowest
                transfer_path = find_shortest_path(
                    zone_neighbors, donor_zone, lowest_zone, 
                    blacklisted_paths, blacklisted_segments
                )
                
                if transfer_path:
                    transfer_attempted = True
                    print(f"  Found alternative transfer path: {transfer_path}")
                    
                    # Process this path
                    path_has_failed_segment = process_transfer_path(
                        transfer_path, df, zones, zone_workloads, zone_centroids, 
                        target_workload, target_points, lat_col, lon_col, 
                        recent_transfers, blacklisted_segments, blacklisted_paths
                    )
                    
                    if not path_has_failed_segment:
                        improved = True
                        break
                    else:
                        # If path failed, blacklist this transfer too
                        blacklisted_transfers.add((donor_zone, lowest_zone))
                        print(f"  Transfer from {donor_zone} to {lowest_zone} failed, blacklisting this combination")
                else:
                    print(f"  No valid path found between zones {donor_zone} and {lowest_zone}")
                    blacklisted_transfers.add((donor_zone, lowest_zone))
            
            # If couldn't transfer to lowest zone, find alternative recipient for highest zone
            if not improved and not transfer_attempted:
                print(f"  Could not transfer to lowest Zone {lowest_zone}, finding alternative recipient for highest Zone {highest_zone}")
                
                # Try to transfer from highest to other low workload zones
                for recipient_idx, (recipient_zone, recipient_workload) in enumerate(sorted_zones[-min(5, len(sorted_zones)):]):
                    if recipient_zone == lowest_zone:
                        continue
                    
                    if (highest_zone, recipient_zone) in blacklisted_transfers:
                        print(f"  Transfer from Zone {highest_zone} to Zone {recipient_zone} is also blacklisted")
                        continue
                    
                    print(f"  Attempting transfer from highest Zone {highest_zone} to alternative low workload Zone {recipient_zone}")
                    
                    # Find path from highest to alternative recipient
                    transfer_path = find_shortest_path(
                        zone_neighbors, highest_zone, recipient_zone, 
                        blacklisted_paths, blacklisted_segments
                    )
                    
                    if transfer_path:
                        print(f"  Found alternative transfer path: {transfer_path}")
                        
                        # Process this path
                        path_has_failed_segment = process_transfer_path(
                            transfer_path, df, zones, zone_workloads, zone_centroids, 
                            target_workload, target_points, lat_col, lon_col, 
                            recent_transfers, blacklisted_segments, blacklisted_paths
                        )
                        
                        if not path_has_failed_segment:
                            improved = True
                            break
                        else:
                            # If path failed, blacklist this transfer too
                            blacklisted_transfers.add((highest_zone, recipient_zone))
                            print(f"  Transfer from {highest_zone} to {recipient_zone} failed, blacklisting this combination")
                    else:
                        print(f"  No valid path found between zones {highest_zone} and {recipient_zone}")
                        blacklisted_transfers.add((highest_zone, recipient_zone))
        else:
            # Regular approach - find path from highest to lowest workload zone
            transfer_path = find_shortest_path(
                zone_neighbors, highest_zone, lowest_zone, 
                blacklisted_paths, blacklisted_segments
            )
            
            if transfer_path:
                print(f"  Transfer path: {transfer_path}")
                
                # Process transfers along the path
                path_has_failed_segment = process_transfer_path(
                    transfer_path, df, zones, zone_workloads, zone_centroids, 
                    target_workload, target_points, lat_col, lon_col, 
                    recent_transfers, blacklisted_segments, blacklisted_paths
                )
                
                if path_has_failed_segment:
                    # If direct path failed, blacklist this transfer combination
                    blacklisted_transfers.add((highest_zone, lowest_zone))
                    print(f"  Transfer from {highest_zone} to {lowest_zone} failed, blacklisting this combination")
                else:
                    improved = True
            else:
                print(f"  No valid path found between zones {highest_zone} and {lowest_zone}")
                # Blacklist this transfer combination
                blacklisted_transfers.add((highest_zone, lowest_zone))
                print(f"  Blacklisting transfer from {highest_zone} to {lowest_zone}")
        
        # If we're still stuck after trying all alternatives, look for ANY valid transfer between ANY zones
        if not improved and no_improvement_counter >= 5:
            print(f"  Stuck for {no_improvement_counter} iterations - attempting ANY valid transfer between zones")
            
            # Get all zones sorted by workload (high to low)
            all_sorted_zones = sorted(zone_workloads.items(), key=lambda x: x[1], reverse=True)
            
            # Try to transfer between any above-average and below-average zone
            avg_workload = sum(zone_workloads.values()) / len(zone_workloads)
            
            # Identify zones above and below average
            above_avg_zones = [z for z, w in all_sorted_zones if w > avg_workload]
            below_avg_zones = [z for z, w in all_sorted_zones if w < avg_workload]
            
            print(f"  Found {len(above_avg_zones)} zones above average and {len(below_avg_zones)} below average")
            
            # Try combinations (with some determinism by sorting)
            transfer_successful = False
            
            # Try different combinations of donor and recipient zones
            for donor_zone in above_avg_zones:
                for recipient_zone in below_avg_zones:
                    # Skip if this transfer is blacklisted
                    if (donor_zone, recipient_zone) in blacklisted_transfers:
                        continue
                    
                    # Skip if workload difference is too small
                    workload_diff = zone_workloads[donor_zone] - zone_workloads[recipient_zone]
                    if workload_diff < target_workload * 0.1:
                        continue
                    
                    print(f"  Attempting transfer from Zone {donor_zone} to Zone {recipient_zone}")
                    
                    # Find path
                    transfer_path = find_shortest_path(
                        zone_neighbors, donor_zone, recipient_zone, 
                        blacklisted_paths, blacklisted_segments
                    )
                    
                    if transfer_path:
                        print(f"  Found transfer path: {transfer_path}")
                        
                        # Process transfers along the path
                        path_has_failed_segment = process_transfer_path(
                            transfer_path, df, zones, zone_workloads, zone_centroids, 
                            target_workload, target_points, lat_col, lon_col, 
                            recent_transfers, blacklisted_segments, blacklisted_paths
                        )
                        
                        if not path_has_failed_segment:
                            transfer_successful = True
                            improved = True
                            print(f"  Successfully found alternative transfer when stuck!")
                            break
                        else:
                            blacklisted_transfers.add((donor_zone, recipient_zone))
                
                if transfer_successful:
                    break
            
            # If still no valid transfers, add more zone pairs to blacklist to avoid testing them again
            if not transfer_successful:
                print(f"  Could not find ANY valid transfers - blacklisting problematic zone combinations")
                
                # Add combinations of the highest with several lowest zones and vice versa
                for i in range(min(3, len(above_avg_zones))):
                    for j in range(min(3, len(below_avg_zones))):
                        donor = above_avg_zones[i]
                        recipient = below_avg_zones[j]
                        if (donor, recipient) not in blacklisted_transfers:
                            blacklisted_transfers.add((donor, recipient))
                            print(f"  Blacklisting combination: Zone {donor} to Zone {recipient}")
        
        # Update zone data after transfers
        zones = {i: df[df['zone_id'] == i] for i in range(num_zones)}
        zone_points = {i: len(zones[i]) for i in range(num_zones)}
        
        # Recalculate workloads
        for i in range(num_zones):
            if len(zones[i]) > 0:
                try:
                    workload = calculate_zone_workload_with_coefficients(zones[i], lat_col, lon_col, points_coef, distance_coef)
                    if workload > 100000:
                        workload = 100000
                    zone_workloads[i] = workload
                except Exception as e:
                    print(f"Error calculating workload for Zone {i}: {e}")
                    zone_workloads[i] = len(zones[i]) * 15
        
        # Update centroids
        for zone_id, zone_df in zones.items():
            if len(zone_df) > 0:
                zone_centroids[zone_id] = np.mean(zone_df[[lat_col, lon_col]].values, axis=0)
        
        # Check for improvement
        workload_std = np.std(list(zone_workloads.values()))
        std_improvement = previous_workload_std - workload_std
        
        if std_improvement < 3:
            stagnation_counter += 1
            no_improvement_counter += 1 if not improved else 0
            print(f"  Limited improvement detected, stagnation counter: {stagnation_counter}")
        else:
            stagnation_counter = 0
            no_improvement_counter = 0
        
        previous_workload_std = workload_std
        
        # Update zone connectivity graph periodically
        if iteration % 5 == 0:
            zone_neighbors = build_zone_connectivity_graph(zones, zone_centroids, lat_col, lon_col)
        
        # Check for early stopping
        if not improved:
            print(f"  No points transferred this iteration")
            no_improvement_counter += 1
            if no_improvement_counter > 15 and emergency_transfer_counter >= max_emergency_transfers:
                print(f"No improvement for 15 iterations and all emergency transfers used - stopping early")
                break
        else:
            # If we had a successful transfer, slightly reduce the no_improvement_counter
            no_improvement_counter = max(0, no_improvement_counter - 1)
        
        # Clean up blacklisted paths and segments periodically
        if iteration % 8 == 0 and iteration > 0:
            # Clean up blacklisted paths
            old_count = len(blacklisted_paths)
            blacklisted_paths = set([p for p in blacklisted_paths if len(p) > 3])  # Keep only longer paths
            new_count = len(blacklisted_paths)
            if old_count != new_count:
                print(f"  Cleared {old_count - new_count} blacklisted paths")
                
            # Clean up blacklisted segments
            if iteration > 15:
                old_segments_count = len(blacklisted_segments)
                # Keep only segments with significant zone size differences
                blacklisted_segments = set([
                    (s, t) for s, t in blacklisted_segments 
                    if abs(zone_points.get(s, 0) - zone_points.get(t, 0)) > target_points * 0.3
                ])
                new_segments_count = len(blacklisted_segments)
                if old_segments_count != new_segments_count:
                    print(f"  Cleared {old_segments_count - new_segments_count} blacklisted segments")
            
            # Clean up blacklisted transfers
            if iteration > 10:
                old_transfers_count = len(blacklisted_transfers)
                # Only keep blacklisted transfers with significant workload differences
                blacklisted_transfers = set([
                    (s, t) for s, t in blacklisted_transfers 
                    if abs(zone_workloads.get(s, 0) - zone_workloads.get(t, 0)) > target_workload * 0.2
                ])
                new_transfers_count = len(blacklisted_transfers)
                if old_transfers_count != new_transfers_count:
                    print(f"  Cleared {old_transfers_count - new_transfers_count} blacklisted transfers")
    
    # Apply final density optimization with more aggressive settings
    print("\n=== PERFORMING FINAL DENSITY OPTIMIZATION ===")
    df, zones, zone_workloads, zone_centroids = optimize_zone_density(
        df, zones, zone_workloads, zone_neighbors, zone_centroids, lat_col, lon_col, 
        swap_percentage=0.15,  # More aggressive percentage
        max_swaps=8  # Allow more swaps in final optimization
    )
    for i in range(num_zones):
                    if len(zones[i]) > 0:
                        try:
                            workload = calculate_zone_workload_with_coefficients(zones[i], lat_col, lon_col)
                            if workload > 100000:
                                workload = 100000
                            zone_workloads[i] = workload
                        except Exception as e:
                            print(f"Error calculating workload for Zone {i}: {e}")
                            zone_workloads[i] = len(zones[i]) * 15
    # Final metrics and reporting
    print("\nFinal zone distribution:")
    for i in range(num_zones):
        points_dev = zone_points[i] - target_points
        workload_dev = zone_workloads[i] - target_workload
        print(f"Zone {i}: {zone_points[i]} points ({points_dev:+.1f}), "
              f"{zone_workloads[i]:.2f} min workload ({workload_dev:+.2f} min)")
    
    # Calculate final statistics
    points_min = min(zone_points.values())
    points_max = max(zone_points.values())
    points_ratio = points_max / max(1, points_min)
    points_std = np.std(list(zone_points.values()))
    points_cv = points_std / (sum(zone_points.values()) / len(zone_points))
    
    workload_min = min(zone_workloads.values())
    workload_max = max(zone_workloads.values())
    workload_ratio = workload_max / max(0.001, workload_min)
    workload_std = np.std(list(zone_workloads.values()))
    workload_cv = workload_std / (sum(zone_workloads.values()) / len(zone_workloads))
    
    print(f"Point count min/max/ratio: {points_min}/{points_max}/{points_ratio:.2f}")
    print(f"Point count std dev: {points_std:.2f}, CV: {points_cv:.3f}")
    print(f"Workload min/max/ratio: {workload_min:.2f}/{workload_max:.2f}/{workload_ratio:.2f}")
    print(f"Workload std dev: {workload_std:.2f}, CV: {workload_cv:.3f}")
    
    # Generate commune statistics if available
    zone_communes = {}
    for zone_id, zone_df in zones.items():
        if 'Commune' in zone_df.columns:
            zone_communes[zone_id] = Counter(zone_df['Commune'])
        else:
            zone_communes[zone_id] = {"Unknown": len(zone_df)}
    
    return df, zones, zone_workloads, zone_communes

def process_transfer_path(transfer_path, df, zones, zone_workloads, zone_centroids, 
                         target_workload, target_points, lat_col, lon_col, 
                         recent_transfers, blacklisted_segments, blacklisted_paths):
    """
    Process transfers along a path, moving points from source to target zones.
    Attempts with smaller numbers of points before blacklisting a segment.
    
    Returns:
    --------
    bool
        True if path has failed segments, False if successful
    """
    # Track if any segment of this path fails
    path_has_failed_segment = False
    failed_segments = []
    
    # Process transfers along the path
    for i in range(len(transfer_path) - 1):
        source_zone = transfer_path[i]
        target_zone = transfer_path[i+1]
        
        # Skip if this segment is already blacklisted
        if (source_zone, target_zone) in blacklisted_segments:
            print(f"  Skipping blacklisted segment from {source_zone} to {target_zone}")
            path_has_failed_segment = True
            failed_segments.append((source_zone, target_zone))
            continue
        
        # Skip if recent transfer between these zones
        # if (source_zone, target_zone) in recent_transfers:
        #     print(f"  Skipping recent transfer from {source_zone} to {target_zone}")
        #     path_has_failed_segment = True
        #     failed_segments.append((source_zone, target_zone))
        #     continue
        
        # Skip if source zone is too small
        min_zone_size = max(5, int(target_points * 0.25))
        if len(zones[source_zone]) <= min_zone_size:
            print(f"  Zone {source_zone} too small to donate points ({len(zones[source_zone])} pts) - skipping")
            path_has_failed_segment = True
            failed_segments.append((source_zone, target_zone))
            blacklisted_segments.add((source_zone, target_zone))
            continue
        
        # Find boundary points suitable for transfer - using the full dataframe
        transfer_candidates = find_boundary_points_directional(
            zones[source_zone], source_zone, target_zone, 
            zone_centroids, lat_col, lon_col, df
        )
        
        if not transfer_candidates:
            print(f"  No suitable boundary points found between {source_zone} and {target_zone}")
            path_has_failed_segment = True
            failed_segments.append((source_zone, target_zone))
            blacklisted_segments.add((source_zone, target_zone))
            continue
        
        # Calculate optimal number of points to transfer
        source_excess = max(0, zone_workloads[source_zone] - target_workload)
        target_deficit = max(0, target_workload - zone_workloads[target_zone])
        
        # Adjust based on position in path
        transfer_weight = 0.5
        if i == 0:  # First transfer (from highest workload)
            transfer_weight = 0.7
        elif i == len(transfer_path) - 2:  # Last transfer (to lowest workload)
            transfer_weight = 0.6
        
        optimal_workload_transfer = min(source_excess, target_deficit) * transfer_weight
        estimated_points = max(6, min(
            int(optimal_workload_transfer / 15),
            len(transfer_candidates),
            int(len(zones[source_zone]) * 0.4)
        ))
        
        # NEW: Create a list of decreasing point counts to try
        # Start with the estimated optimal, then try progressively smaller amounts
        point_counts_to_try = [
            estimated_points,
            max(5, int(estimated_points * 0.75)),
            max(3, int(estimated_points * 0.5)),
            max(2, int(estimated_points * 0.25)),
            1  # Try transferring just 1 point as a last resort
        ]
        
        transfer_success = False
        
        # Try each point count until one succeeds
        for num_points in point_counts_to_try:
            if num_points > len(transfer_candidates):
                continue
                
            print(f"  Attempting to transfer {num_points} points from Zone {source_zone} to Zone {target_zone}")
            
            # Sort candidates by proximity to target zone
            transfer_indices = [idx for idx, _, _ in transfer_candidates[:num_points]]
            
            # Verify connectivity preservation before committing transfer
            temp_df = df.copy()
            for idx in transfer_indices:
                temp_df.at[idx, 'zone_id'] = target_zone
            
            source_still_connected = check_zone_connectivity(temp_df, source_zone, lat_col, lon_col)
            target_still_connected = check_zone_connectivity(temp_df, target_zone, lat_col, lon_col)
            
            if source_still_connected and target_still_connected:
                # Commit transfer
                for idx in transfer_indices:
                    df.at[idx, 'zone_id'] = target_zone
                
                # Update tracking
                recent_transfers.add((source_zone, target_zone))
                if len(recent_transfers) > 20:
                    recent_transfers = set(list(recent_transfers)[-20:])
                
                print(f"  Successfully transferred {len(transfer_indices)} points")
                transfer_success = True
                break
            else:
                if num_points > 1:
                    print(f"  Transfer of {num_points} points would disconnect zones - trying fewer points")
                else:
                    print(f"  Even transferring a single point would disconnect zones - skipping")
        
        # If all transfer attempts failed, mark this segment as failed
        if not transfer_success:
            path_has_failed_segment = True
            failed_segments.append((source_zone, target_zone))
            blacklisted_segments.add((source_zone, target_zone))
    
    # Blacklist the entire path if it had failed segments
    if path_has_failed_segment:
        path_key = tuple(transfer_path)
        blacklisted_paths.add(path_key)
        print(f"  Path {path_key} has been blacklisted due to failed segments: {failed_segments}")
        
        # Add individual failed segments to blacklist
        for segment in failed_segments:
            blacklisted_segments.add(segment)
            print(f"  Segment {segment} has been blacklisted")
    
    return path_has_failed_segment


def find_shortest_path(zone_neighbors, start_zone, end_zone, blacklisted_paths=None, blacklisted_segments=None):
    """
    Find shortest path from start_zone to end_zone through the zone neighbor graph,
    avoiding any blacklisted paths and segments.
    
    Parameters:
    -----------
    zone_neighbors : dict
        Dictionary mapping zone IDs to lists of neighboring zone IDs
    start_zone : int
        Starting zone ID
    end_zone : int
        Target zone ID
    blacklisted_paths : set, optional
        Set of tuples representing blacklisted paths to avoid
    blacklisted_segments : set, optional
        Set of tuples representing blacklisted segments (source, target) to avoid
        
    Returns:
    --------
    list
        List of zone IDs forming the path from start to end, or None if no path exists
    """
    if blacklisted_paths is None:
        blacklisted_paths = set()
    
    if blacklisted_segments is None:
        blacklisted_segments = set()
        
    if start_zone == end_zone:
        return [start_zone]
    
    # Use BFS to find shortest path
    queue = deque([(start_zone, [start_zone])])
    visited = set([start_zone])
    
    # Track found paths to return alternative path if best one is blacklisted
    all_found_paths = []
    
    while queue:
        current_zone, path = queue.popleft()
        
        for neighbor in zone_neighbors.get(current_zone, []):
            # Check if this segment is blacklisted
            if (current_zone, neighbor) in blacklisted_segments:
                continue
                
            new_path = path + [neighbor]
            
            # Check if this path is blacklisted
            path_tuple = tuple(new_path)
            if path_tuple in blacklisted_paths:
                continue
            
            if neighbor == end_zone:
                # Found a valid path to the destination
                all_found_paths.append(new_path)
                # Return immediately if we found a valid path
                return new_path
            
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, new_path))
    
    # If we found any paths but they were all blacklisted, return the first one
    # This is a fallback mechanism
    if all_found_paths:
        print(f"  All direct paths from Zone {start_zone} to Zone {end_zone} are blacklisted, using best available")
        return all_found_paths[0]
    
    # If direct path not found, try to find best partial path
    print(f"  No direct path from Zone {start_zone} to Zone {end_zone}, finding best partial path")
    
    # Use Dijkstra's algorithm with zone workload differential as edge weight
    shortest_paths = {}
    for zone in zone_neighbors:
        shortest_paths[zone] = float('inf')
    
    shortest_paths[start_zone] = 0
    unvisited = set(zone_neighbors.keys())
    path_prev = {}
    
    while unvisited:
        current = min(unvisited, key=lambda z: shortest_paths[z])
        
        if current == end_zone or shortest_paths[current] == float('inf'):
            break
            
        unvisited.remove(current)
        
        for neighbor in zone_neighbors.get(current, []):
            if neighbor in unvisited:
                # Check if this segment is blacklisted
                segment_blacklisted = (current, neighbor) in blacklisted_segments
                
                # Skip blacklisted segments entirely
                if segment_blacklisted:
                    continue
                
                # Use simple distance as weight
                weight = 1
                
                # Check if this edge forms part of a blacklisted path
                edge_in_blacklisted_path = False
                for path in blacklisted_paths:
                    if len(path) >= 2:
                        for i in range(len(path) - 1):
                            if path[i] == current and path[i+1] == neighbor:
                                edge_in_blacklisted_path = True
                                break
                
                # Apply a penalty if the edge is part of a blacklisted path
                if edge_in_blacklisted_path:
                    weight = 5  # Heavy penalty for blacklisted edges
                
                distance = shortest_paths[current] + weight
                if distance < shortest_paths[neighbor]:
                    shortest_paths[neighbor] = distance
                    path_prev[neighbor] = current
    
    # Reconstruct the path
    if end_zone in path_prev or end_zone == start_zone:
        path = [end_zone]
        while path[-1] != start_zone:
            if path[-1] not in path_prev:
                break
            path.append(path_prev[path[-1]])
        path.reverse()
        
        # Check if reconstructed path is blacklisted
        path_tuple = tuple(path)
        if path_tuple in blacklisted_paths:
            print(f"  Reconstructed path {path_tuple} is blacklisted, but using as last resort")
        
        # Check if path contains blacklisted segments
        for i in range(len(path) - 1):
            if (path[i], path[i+1]) in blacklisted_segments:
                print(f"  Warning: Reconstructed path contains blacklisted segment {(path[i], path[i+1])}")
        
        return path
    
    return None

def find_boundary_points_directional(source_df, source_zone, target_zone, zone_centroids,
                                    lat_col="Latitude", lon_col="Longitude", df_all=None):
    """
    Find boundary points that are candidates for transfer between zones,
    prioritizing points that are closest to the destination zone.
    
    Parameters:
    -----------
    source_df : pandas.DataFrame
        DataFrame containing points in the source zone
    source_zone : int
        ID of the source zone
    target_zone : int
        ID of the target zone
    zone_centroids : dict
        Dictionary with zone_id as key and (lat, lon) as value
    lat_col, lon_col : str
        Column names for latitude and longitude
    df_all : pandas.DataFrame, optional
        DataFrame containing all points
        
    Returns:
    --------
    list
        List of (point_index, target_zone_id, distance) tuples
    """
    source_size = len(source_df)
    if source_size == 0 or source_zone not in zone_centroids or target_zone not in zone_centroids:
        return []
    
    # Get target zone data if available
    target_df = None
    if df_all is not None:
        target_df = df_all[df_all['zone_id'] == target_zone]
    
    # Sample large zones for efficiency
    sample_size = min(source_size, 600 + int(source_size * 0.4))
    sampled_source = source_df.sample(n=sample_size) if source_size > sample_size else source_df
    
    # Get centroids
    source_centroid = zone_centroids[source_zone]
    target_centroid = zone_centroids[target_zone]
    
    # Calculate direction vector from source to target
    direction_vector = (
        target_centroid[0] - source_centroid[0],
        target_centroid[1] - source_centroid[1]
    )
    
    # Normalize direction vector
    direction_length = (direction_vector[0]**2 + direction_vector[1]**2)**0.5
    if direction_length > 0:
        direction_vector = (
            direction_vector[0] / direction_length,
            direction_vector[1] / direction_length
        )
    
    # Calculate boundary threshold adaptively
    boundary_threshold_percentile = max(30, 50 - (source_size / 80))
    
    # Calculate distances and directional scores for each point
    point_scores = {}
    distances_to_centroid = []
    
    # If we have target zone data, find nearest neighbor distances
    target_points = []
    if target_df is not None and not target_df.empty:
        target_points = target_df[[lat_col, lon_col]].values
    
    for idx, row in sampled_source.iterrows():
        # Distance to own centroid
        dist_to_own = haversine_distance(
            row[lat_col], row[lon_col],
            source_centroid[0], source_centroid[1]
        )
        distances_to_centroid.append(dist_to_own)
        
        # Distance to target centroid
        dist_to_target = haversine_distance(
            row[lat_col], row[lon_col],
            target_centroid[0], target_centroid[1]
        )
        
        # Calculate distance to nearest point in target zone
        dist_to_nearest_target_point = float('inf')
        if len(target_points) > 0:
            for target_point in target_points:
                dist = haversine_distance(
                    row[lat_col], row[lon_col],
                    target_point[0], target_point[1]
                )
                dist_to_nearest_target_point = min(dist_to_nearest_target_point, dist)
        else:
            # If no target points, use distance to centroid
            dist_to_nearest_target_point = dist_to_target
        
        # Vector from centroid to point
        point_vector = (
            row[lat_col] - source_centroid[0],
            row[lon_col] - source_centroid[1]
        )
        
        # Normalize point vector
        point_length = (point_vector[0]**2 + point_vector[1]**2)**0.5
        if point_length > 0:
            point_vector = (
                point_vector[0] / point_length,
                point_vector[1] / point_length
            )
            
        # Calculate directional alignment (dot product)
        # Higher value means better alignment with direction to target
        alignment = point_vector[0] * direction_vector[0] + point_vector[1] * direction_vector[1]
        
        # Enhance weight of proximity to target zone
        proximity_weight = 2.0  # Increase importance of being close to target zone
        
        # Combined score - prioritize points that are:
        # 1. Far from own centroid
        # 2. VERY close to target zone (increased weight)
        # 3. In the direction of target zone
        directional_score = (alignment + 1) / 2  # Convert from [-1,1] to [0,1]
        combined_score = (
            dist_to_own * directional_score / 
            max(0.001, dist_to_nearest_target_point ** proximity_weight)
        )
        
        point_scores[idx] = (
            dist_to_own, 
            dist_to_target, 
            directional_score, 
            combined_score, 
            dist_to_nearest_target_point
        )
    
    if not distances_to_centroid:
        return []
    
    # Determine boundary threshold
    boundary_threshold = np.percentile(distances_to_centroid, boundary_threshold_percentile)
    
    # Identify boundary points in the direction of target and close to target
    boundary_points = []
    
    for idx, (dist_to_own, dist_to_target, directional_score, combined_score, dist_to_nearest) in point_scores.items():
        # Check if point is in boundary region, has good directional alignment, and is close to target
        if dist_to_own >= boundary_threshold * 0.6 and directional_score > 0.5:
            boundary_points.append((idx, target_zone, combined_score))
    
    # If not enough points found, relax directional constraint but keep proximity constraint
    if len(boundary_points) < 10:
        boundary_points = []
        for idx, (dist_to_own, dist_to_target, directional_score, combined_score, dist_to_nearest) in point_scores.items():
            if dist_to_own >= boundary_threshold * 0.5 and directional_score > 0.0:
                boundary_points.append((idx, target_zone, combined_score))
    
    # If still not enough, prioritize points that are closest to the target zone
    if len(boundary_points) < 5:
        # Sort by distance to nearest target point (ascending)
        sorted_by_proximity = sorted(
            point_scores.items(), 
            key=lambda x: x[1][4]  # dist_to_nearest_target_point
        )
        boundary_points = [(idx, target_zone, 1/score[4]) for idx, score in sorted_by_proximity[:20]]
    
    # Sort by combined score (descending)
    boundary_points.sort(key=lambda x: x[2], reverse=True)
    
    # Perform spatial clustering to ensure we're transferring connected regions
    if len(boundary_points) >= 20:
        # Get point coordinates
        point_coords = {}
        for idx, _, _ in boundary_points:
            row = source_df.loc[idx]
            point_coords[idx] = (row[lat_col], row[lon_col])
        
        # Calculate adjacency threshold based on point distribution
        if len(point_coords) > 1:
            avg_dist = calculate_avg_nearest_neighbor_distance(point_coords)
            adjacency_threshold = avg_dist * 3.0
        else:
            adjacency_threshold = 1.0
        
        # Start with top points as seeds
        seed_count = min(len(boundary_points), max(3, len(boundary_points) // 10))
        seed_indices = set([boundary_points[i][0] for i in range(seed_count)])
        
        clustered_points = seed_indices.copy()
        remaining_indices = set([p[0] for p in boundary_points]) - seed_indices
        
        # Expand clusters iteratively
        for _ in range(3):
            newly_added = set()
            for idx in remaining_indices:
                if idx in clustered_points:
                    continue
                
                idx_coord = point_coords[idx]
                
                # Check if point is adjacent to already clustered points
                for cluster_idx in clustered_points:
                    cluster_coord = point_coords[cluster_idx]
                    dist = haversine_distance(
                        idx_coord[0], idx_coord[1],
                        cluster_coord[0], cluster_coord[1]
                    )
                    
                    if dist <= adjacency_threshold:
                        newly_added.add(idx)
                        break
            
            clustered_points.update(newly_added)
            remaining_indices -= newly_added
            
            if not newly_added:
                break
        
        # Filter boundary points to only include clustered points
        boundary_points = [p for p in boundary_points if p[0] in clustered_points]
    
    return boundary_points


def calculate_avg_nearest_neighbor_distance(points):
    """Calculate the average distance to nearest neighbor among points"""
    if len(points) <= 1:
        return 0.5  # Default in km
        
    # Sample points if there are too many
    point_keys = list(points.keys())
    sample_size = min(100, len(point_keys))
    
    if len(point_keys) > sample_size:
        point_keys = np.random.choice(point_keys, sample_size, replace=False)
    
    # For each sampled point, find its nearest neighbor
    nearest_distances = []
    for idx in point_keys:
        point = points[idx]
        min_dist = float('inf')
        
        for other_idx in point_keys:
            if other_idx != idx:
                other_point = points[other_idx]
                dist = haversine_distance(
                    point[0], point[1],
                    other_point[0], other_point[1]
                )
                min_dist = min(min_dist, dist)
        
        if min_dist < float('inf'):
            nearest_distances.append(min_dist)
    
    # Return average nearest neighbor distance
    if nearest_distances:
        avg_dist = sum(nearest_distances) / len(nearest_distances)
        # Avoid unrealistically large or small values
        return max(0.1, min(avg_dist, 2.0))
    else:
        return 0.5  # Default in km



import json
import numpy as np
from collections import Counter

def generate_zone_boundaries(zones, buffer_distance=0.003):
    """Generate non-overlapping polygons for zone boundaries with improved corridors and optimal padding
    
    This improved function ensures each point is contained within its zone boundary,
    all components of the same zone are unified with corridors, interior gaps are filled,
    and padding is optimized to keep points near the perimeter.
    """
    print("Generating optimized zone boundaries...")
    
    from shapely.geometry import Point, MultiPoint, Polygon, MultiPolygon, LineString
    from shapely.ops import unary_union, nearest_points, unary_union
    import pandas as pd
    import numpy as np
    import networkx as nx
    
    # PHASE 1: Create initial tight polygon shapes around points for each zone
    raw_polygons = {}
    for zone_id, zone_df in zones.items():
        if len(zone_df) < 3:  # Need at least 3 points for a polygon
            if len(zone_df) > 0:
                # For very small zones, create minimal buffer around points
                points = [Point(row['Longitude'], row['Latitude']) for _, row in zone_df.iterrows()]
                # Create minimal buffer around each point - reduced from original
                buffers = [p.buffer(buffer_distance * 1.5) for p in points]
                raw_polygons[zone_id] = unary_union(buffers)
            continue
            
        try:
            # Create points for this zone
            points = [Point(row['Longitude'], row['Latitude']) for _, row in zone_df.iterrows()]
            
            # Try to use alpha shapes instead of buffer if many points are available
            if len(points) >= 10:
                try:
                    from shapely.geometry import MultiPoint
                    # Create a concave hull (alpha shape) instead of just buffering points
                    # This creates tighter boundaries around point clusters
                    multi_point = MultiPoint(points)
                    # First try concave hull
                    try:
                        from alphashape import alphashape
                        # Use alpha shape with higher alpha for tighter fit
                        alpha_value = 8.0
                        alpha_shape = alphashape(multi_point, alpha_value)
                        if not alpha_shape.is_empty and alpha_shape.is_valid:
                            # Add minimal buffer to ensure points near edge are included
                            raw_polygons[zone_id] = alpha_shape.buffer(buffer_distance)
                            continue
                    except (ImportError, Exception) as e:
                        print(f"Alpha shape failed, falling back: {e}")
                    
                    # Fallback to convex hull with minimal buffer
                    convex_hull = multi_point.convex_hull
                    raw_polygons[zone_id] = convex_hull.buffer(buffer_distance)
                    continue
                except Exception as e:
                    print(f"Error creating alpha shape for zone {zone_id}: {e}")
            
            # Fallback to buffered points with tighter buffer
            point_buffers = [p.buffer(buffer_distance * 1.5) for p in points]
            zone_shape = unary_union(point_buffers)
            raw_polygons[zone_id] = zone_shape
            
        except Exception as e:
            print(f"Error creating polygon for zone {zone_id}: {e}")
            # Fallback: create minimal buffers around points
            if len(zone_df) > 0:
                points = [Point(row['Longitude'], row['Latitude']) for _, row in zone_df.iterrows()]
                buffers = [p.buffer(buffer_distance * 1.5) for p in points]
                raw_polygons[zone_id] = unary_union(buffers)
    
    # PHASE 2: Resolve overlaps between zones (keeping original logic)
    zone_polygons = {}
    
    if raw_polygons:
        # Create a mapping of points to their zones
        point_to_zone = {}
        for zone_id, zone_df in zones.items():
            for _, row in zone_df.iterrows():
                point_key = (row['Longitude'], row['Latitude'])
                point_to_zone[point_key] = zone_id
        
        # Process zones by size, largest first
        sorted_zones = sorted(raw_polygons.keys(), 
                            key=lambda z: raw_polygons[z].area if hasattr(raw_polygons[z], 'area') else 0,
                            reverse=True)
        
        processed_zones = set()
        
        for zone_id in sorted_zones:
            current_poly = raw_polygons[zone_id]
            
            # Check against already processed zones
            for other_id in processed_zones:
                other_poly = zone_polygons[other_id]
                
                try:
                    # Check if they overlap
                    if current_poly.intersects(other_poly):
                        overlap = current_poly.intersection(other_poly)
                        
                        if not overlap.is_empty and overlap.area > 0:
                            # Find which points are in the overlap area
                            points_in_overlap = []
                            zones_with_points = {zone_id: 0, other_id: 0}
                            
                            # Check points near the overlap area
                            overlap_bounds = overlap.bounds
                            
                            for point_coords, point_zone_id in point_to_zone.items():
                                lon, lat = point_coords
                                # Quick bounding box check before more expensive point-in-polygon check
                                if (overlap_bounds[0] <= lon <= overlap_bounds[2] and 
                                    overlap_bounds[1] <= lat <= overlap_bounds[3]):
                                    point = Point(lon, lat)
                                    if overlap.contains(point):
                                        points_in_overlap.append((point, point_zone_id))
                                        if point_zone_id in zones_with_points:
                                            zones_with_points[point_zone_id] += 1
                            
                            # If there are points in the overlap, adjust the boundaries
                            if points_in_overlap:
                                # If only one zone has points in the overlap, give it the overlap
                                if zones_with_points[zone_id] > 0 and zones_with_points[other_id] == 0:
                                    # Current zone keeps the overlap
                                    other_poly = other_poly.difference(overlap)
                                    zone_polygons[other_id] = other_poly
                                elif zones_with_points[zone_id] == 0 and zones_with_points[other_id] > 0:
                                    # Other zone keeps the overlap
                                    current_poly = current_poly.difference(overlap)
                                else:
                                    # Both zones have points in the overlap
                                    # Split the overlap based on which points belong to which zone
                                    point_buffers_current = []
                                    point_buffers_other = []
                                    
                                    for point, point_zone_id in points_in_overlap:
                                        if point_zone_id == zone_id:
                                            point_buffers_current.append(point.buffer(buffer_distance))
                                        elif point_zone_id == other_id:
                                            point_buffers_other.append(point.buffer(buffer_distance))
                                    
                                    if point_buffers_current:
                                        current_claim = unary_union(point_buffers_current)
                                        other_poly = other_poly.difference(current_claim)
                                        zone_polygons[other_id] = other_poly
                                    
                                    if point_buffers_other:
                                        other_claim = unary_union(point_buffers_other)
                                        current_poly = current_poly.difference(other_claim)
                            else:
                                # No points in overlap, use point density or distance to determine ownership
                                zone_points = len(zones[zone_id])
                                other_points = len(zones[other_id])
                                
                                # Calculate centroids
                                zone_centroid = current_poly.centroid
                                other_centroid = other_poly.centroid
                                overlap_centroid = overlap.centroid
                                
                                # Calculate distances
                                zone_dist = zone_centroid.distance(overlap_centroid)
                                other_dist = other_centroid.distance(overlap_centroid)
                                
                                # Zone with closer centroid or higher point density gets the overlap
                                if zone_dist <= other_dist or (zone_points / current_poly.area > other_points / other_poly.area):
                                    # Current zone keeps the overlap
                                    other_poly = other_poly.difference(overlap)
                                    zone_polygons[other_id] = other_poly
                                else:
                                    # Other zone keeps the overlap
                                    current_poly = current_poly.difference(overlap)
                except Exception as e:
                    print(f"Warning: Error handling overlap between zones {zone_id} and {other_id}: {e}")
                    continue
            
            # Store processed polygon
            zone_polygons[zone_id] = current_poly
            processed_zones.add(zone_id)
    
    # PHASE 3: IMPROVED CORRIDOR GENERATION - Create optimal corridors between disconnected parts
    
    # Start with polygons from phase 2
    unified_zone_polygons = {}
    
    for zone_id, zone_df in zones.items():
        # Skip empty zones
        if len(zone_df) == 0:
            continue
            
        # Get all points in this zone
        zone_points = [Point(row['Longitude'], row['Latitude']) for _, row in zone_df.iterrows()]
        
        # If we have an existing polygon for this zone, start with it
        if zone_id in zone_polygons:
            base_polygon = zone_polygons[zone_id]
        else:
            # Otherwise create a minimal polygon from the points
            if len(zone_points) < 3:
                # For fewer than 3 points, just use minimal buffer
                base_polygon = unary_union([p.buffer(buffer_distance * 1.5) for p in zone_points])
            else:
                # For 3+ points, create tighter boundary
                try:
                    multi_point = MultiPoint(zone_points)
                    base_polygon = multi_point.convex_hull.buffer(buffer_distance)
                except Exception:
                    # Fallback to buffered points
                    base_polygon = unary_union([p.buffer(buffer_distance * 1.5) for p in zone_points])
        
        # Check if polygon is MultiPolygon (disconnected parts)
        # or if points are outside the polygon
        if isinstance(base_polygon, MultiPolygon) or any(not base_polygon.contains(p) for p in zone_points):
            print(f"Zone {zone_id}: Creating optimal corridors to connect components...")
            
            # IMPROVED CORRIDOR ALGORITHM
            # 1. Identify all disconnected components
            components = []
            
            # If base_polygon is a MultiPolygon, each part is a component
            if isinstance(base_polygon, MultiPolygon):
                for poly in base_polygon.geoms:
                    components.append(poly)
            else:
                components.append(base_polygon)
            
            # 2. For each point, find which component contains it or create new component
            point_components = {}  # Maps points to their component index
            points_by_component = {i: [] for i in range(len(components))}
            orphan_points = []
            
            for i, point in enumerate(zone_points):
                found = False
                for j, component in enumerate(components):
                    if component.contains(point):
                        point_components[i] = j
                        points_by_component[j].append(i)
                        found = True
                        break
                if not found:
                    orphan_points.append(i)
            
            # 3. Create minimal components for orphan points
            for i, point_idx in enumerate(orphan_points):
                point = zone_points[point_idx]
                new_component = point.buffer(buffer_distance * 1.5)
                components.append(new_component)
                comp_idx = len(components) - 1
                point_components[point_idx] = comp_idx
                points_by_component[comp_idx] = [point_idx]
            
            # 4. Build a graph representing component connectivity
            G = nx.Graph()
            for i in range(len(components)):
                G.add_node(i, geometry=components[i], points=points_by_component.get(i, []))
            
            # 5. Find optimal connections between components using MST
            if len(components) > 1:
                # Calculate distances between all components
                for i in range(len(components) - 1):
                    for j in range(i + 1, len(components)):
                        # Find shortest distance between component boundaries
                        try:
                            p1, p2 = nearest_points(components[i].boundary, components[j].boundary)
                            dist = p1.distance(p2)
                            if dist > 0:  # Avoid self-connections
                                G.add_edge(i, j, weight=dist, points=(p1, p2))
                        except Exception as e:
                            print(f"Error finding nearest points: {e}")
                            continue
                
                # Build MST to find optimal connections
                mst_edges = list(nx.minimum_spanning_edges(G, weight='weight', data=True))
                
                # 6. Create corridors along the MST edges
                corridors = []
                for u, v, data in mst_edges:
                    try:
                        p1, p2 = data['points']
                        # Create corridor with minimal width
                        corridor = LineString([p1, p2]).buffer(buffer_distance * 1.2)
                        corridors.append(corridor)
                    except Exception as e:
                        print(f"Error creating corridor: {e}")
                        continue
                
                # 7. Union all components and corridors
                if corridors:
                    unified_poly = unary_union(components + corridors)
                else:
                    unified_poly = unary_union(components)
                
                # 8. FILL INTERNAL VOIDS - Identify and fill internal gaps
                if hasattr(unified_poly, 'interiors') and unified_poly.interiors:
                    # For simple polygons, fill all internal holes
                    unified_poly = Polygon(unified_poly.exterior)
                elif isinstance(unified_poly, MultiPolygon):
                    # For multipolygons, fill holes in each part
                    filled_parts = []
                    for part in unified_poly.geoms:
                        if hasattr(part, 'interiors') and part.interiors:
                            filled_parts.append(Polygon(part.exterior))
                        else:
                            filled_parts.append(part)
                    unified_poly = unary_union(filled_parts)
            else:
                # Only one component, just use it
                unified_poly = components[0]
                
                # Fill any holes in single polygon
                if hasattr(unified_poly, 'interiors') and unified_poly.interiors:
                    unified_poly = Polygon(unified_poly.exterior)
                
            unified_zone_polygons[zone_id] = unified_poly
        else:
            # No disconnected parts, but still fill internal voids
            if hasattr(base_polygon, 'interiors') and base_polygon.interiors:
                base_polygon = Polygon(base_polygon.exterior)
            
            unified_zone_polygons[zone_id] = base_polygon
    
    # PHASE 4: REDUCE PADDING - Shrink polygons directionally while ensuring all points are contained
    print("Optimizing zone boundaries to minimize padding with directional shrinking...")
    
    final_zone_polygons = {}
    
    # Use more granular shrink factors for better precision
    shrink_factors = [0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3]
    
    for zone_id, zone_df in zones.items():
        if zone_id not in unified_zone_polygons:
            continue
            
        original_poly = unified_zone_polygons[zone_id]
        zone_points = [Point(row['Longitude'], row['Latitude']) for _, row in zone_df.iterrows()]
        
        # Skip if polygon is already minimal or too small
        if original_poly.area < 0.0001 or len(zone_points) < 3:
            final_zone_polygons[zone_id] = original_poly
            print(f"Zone {zone_id}: Skipping shrink optimization (too small or too few points)")
            continue
        
        # Create a copy of original polygon for optimization
        poly = original_poly
        
        # Calculate the bounds of the polygon and the points
        poly_bounds = poly.bounds  # (minx, miny, maxx, maxy)
        
        # Find the actual bounds of the points
        if zone_points:
            points_x = [p.x for p in zone_points]
            points_y = [p.y for p in zone_points]
            points_bounds = (min(points_x), min(points_y), max(points_x), max(points_y))
        else:
            points_bounds = poly_bounds
        
        # Calculate padding on each side (distance from points to polygon boundary)
        padding = {
            'left': points_bounds[0] - poly_bounds[0],      # west padding
            'bottom': points_bounds[1] - poly_bounds[1],    # south padding 
            'right': poly_bounds[2] - points_bounds[2],     # east padding
            'top': poly_bounds[3] - points_bounds[3]        # north padding
        }
        
        # Determine which sides have excessive padding (above threshold)
        padding_threshold = 0.001  # Minimum padding to consider for reduction
        sides_to_shrink = [side for side, value in padding.items() if value > padding_threshold]
        
        
        # Try directional shrinking for each side with excessive padding
        try:
            optimal_shrink_results = {}
            
            # Only attempt directional shrinking if we have identified sides to shrink
            if sides_to_shrink:
                from shapely.affinity import scale, translate
                
                # For each side with excessive padding, try to shrink it
                for side in sides_to_shrink:
                    optimal_shrink_factor = 0
                    current_poly = poly  # Start with the original polygon for each side
                    
                    for shrink_factor in shrink_factors:
                        # Clone the polygon before modifying
                        test_poly = current_poly
                        
                        # Apply directional shrinking based on the side
                        if side == 'left':
                            # Shrink from the left (west)
                            shrink_amount = padding['left'] * shrink_factor
                            test_poly = translate(test_poly, xoff=shrink_amount, yoff=0)
                        elif side == 'right':
                            # Shrink from the right (east)
                            shrink_amount = padding['right'] * shrink_factor
                            test_poly = translate(test_poly, xoff=0, yoff=0)
                            # Scale horizontally from the left edge
                            scale_factor = 1 - (shrink_amount / (poly_bounds[2] - poly_bounds[0]))
                            test_poly = scale(test_poly, xfact=scale_factor, yfact=1, origin=(poly_bounds[0], 0))
                        elif side == 'bottom':
                            # Shrink from the bottom (south)
                            shrink_amount = padding['bottom'] * shrink_factor
                            test_poly = translate(test_poly, xoff=0, yoff=shrink_amount)
                        elif side == 'top':
                            # Shrink from the top (north)
                            shrink_amount = padding['top'] * shrink_factor
                            test_poly = translate(test_poly, xoff=0, yoff=0)
                            # Scale vertically from the bottom edge
                            scale_factor = 1 - (shrink_amount / (poly_bounds[3] - poly_bounds[1]))
                            test_poly = scale(test_poly, xfact=1, yfact=scale_factor, origin=(0, poly_bounds[1]))
                        
                        # Check if the test polygon is valid and contains all points
                        if test_poly.is_empty or not test_poly.is_valid:
                            break
                        
                        if all(test_poly.contains(p) for p in zone_points):
                            current_poly = test_poly  # Update current polygon
                            optimal_shrink_factor = shrink_factor  # Track the successful factor
                        else:
                            break
                    
                    # Record the optimal shrink factor for this side
                    optimal_shrink_results[side] = optimal_shrink_factor
                    
                    # Apply the optimal shrinking to the main polygon
                    if optimal_shrink_factor > 0:
                        poly = current_poly
                
                # Print the optimal shrink factors for each side
            else:
                # Try uniform shrinking as fallback if no specific sides need shrinking
                optimal_shrink_factor = 0
                
                for shrink_factor in shrink_factors:
                    # Shrink polygon uniformly
                    shrunk_poly = poly.buffer(-poly.area * shrink_factor)
                    
                    # Empty or invalid result - use previous
                    if shrunk_poly.is_empty or not shrunk_poly.is_valid:
                        break
                        
                    # Check if all points still contained
                    if all(shrunk_poly.contains(p) for p in zone_points):
                        poly = shrunk_poly  # Update to shrunk version
                        optimal_shrink_factor = shrink_factor  # Track the successful factor
                    else:
                        # Stop shrinking if points would be outside
                        break
                
            
            # Expand slightly to ensure points near border have some margin
            final_poly = poly.buffer(buffer_distance * 0.3)
            final_zone_polygons[zone_id] = final_poly
            
        except Exception as e:
            print(f"Error optimizing polygon for zone {zone_id}: {e}")
            final_zone_polygons[zone_id] = poly
    
    # PHASE 5: Final verification and cleanup
    print("Verifying final zone boundaries...")
    
    # Check that every point is in exactly one zone boundary
    for zone_id, zone_df in zones.items():
        if zone_id not in final_zone_polygons:
            continue
            
        zone_poly = final_zone_polygons[zone_id]
        points_outside = []
        
        # Check each point in this zone
        for _, row in zone_df.iterrows():
            point = Point(row['Longitude'], row['Latitude'])
            if not zone_poly.contains(point):
                points_outside.append(point)
        
        # If there are still points outside, make final adjustment
        if points_outside:
            print(f"Final check - Zone {zone_id}: {len(points_outside)} points still outside. Making final adjustment.")
            point_buffers = [p.buffer(buffer_distance) for p in points_outside]
            zone_poly = unary_union([zone_poly] + point_buffers)
            final_zone_polygons[zone_id] = zone_poly
    
    # Make sure all polygons are valid
    for zone_id in list(final_zone_polygons.keys()):
        try:
            poly = final_zone_polygons[zone_id]
            
            if poly is None or poly.is_empty:
                # Create fallback polygon centered on zone points
                if zone_id in zones and len(zones[zone_id]) > 0:
                    points = [Point(row['Longitude'], row['Latitude']) for _, row in zones[zone_id].iterrows()]
                    buffers = [p.buffer(buffer_distance * 1.5) for p in points]
                    final_zone_polygons[zone_id] = unary_union(buffers)
            elif not poly.is_valid:
                # Try to fix invalid polygon
                print(f"Fixing invalid polygon for zone {zone_id}")
                final_zone_polygons[zone_id] = poly.buffer(0)
        except Exception as e:
            print(f"Error validating polygon for zone {zone_id}: {e}")
            if zone_id in zones and len(zones[zone_id]) > 0:
                points = [Point(row['Longitude'], row['Latitude']) for _, row in zones[zone_id].iterrows()]
                buffers = [p.buffer(buffer_distance * 1.5) for p in points]
                final_zone_polygons[zone_id] = unary_union(buffers)
    
    print("Optimized zone boundary generation complete")
    return final_zone_polygons



def export_zones_to_geojson(df, zones, zone_workloads, zone_communes, zone_polygons, output_path="zones.geojson", commune_geojson=None, points_coef=1, distance_coef=1):
    """Export zone data to GeoJSON file with metadata including optimization metrics"""
    import json
    import numpy as np
    from collections import Counter
    from shapely.geometry import Point
    
    try:
        # Assign communes from GeoJSON file if provided
        if commune_geojson:
            print(f"Assigning communes from {commune_geojson}")
            df = assign_communes_from_geojson(df, commune_geojson)

            # Ensure each zone DataFrame has 'Commune' updated
            for zone_id, zone_df in zones.items():
                if 'Commune' not in zone_df.columns:
                    zone_df = zone_df.merge(df[['Latitude', 'Longitude', 'Commune']], on=['Latitude', 'Longitude'], how='left')
                    zones[zone_id] = zone_df
                else:
                    missing_communes = zone_df['Commune'].isna()
                    if missing_communes.any():
                        merged = zone_df.merge(df[['Latitude', 'Longitude', 'Commune']], on=['Latitude', 'Longitude'], how='left', suffixes=('', '_new'))
                        zone_df['Commune'] = merged['Commune_new'].combine_first(zone_df['Commune'])
                        zones[zone_id] = zone_df

            # Update zone_communes with latest assignments
            zone_communes = {}
            for zone_id, zone_df in zones.items():
                zone_communes[zone_id] = Counter(zone_df['Commune'].dropna())
        
        # Calculate boundary metrics
        boundary_metrics = {}
        for zone_id, polygon in zone_polygons.items():
            if zone_id in zones:
                zone_df = zones[zone_id]
                
                # Count points and calculate area
                points_count = len(zone_df)
                area = polygon.area if hasattr(polygon, 'area') else 0
                
                # Calculate convexity ratio (area / convex hull area)
                convexity = 1.0  # Default to perfect convexity
                if points_count >= 3:
                    try:
                        # Create a MultiPoint and get its convex hull
                        points = [Point(row['Longitude'], row['Latitude']) for _, row in zone_df.iterrows()]
                        mp = MultiPoint(points)
                        convex_hull = mp.convex_hull
                        
                        # Calculate ratio
                        if convex_hull.area > 0:
                            convexity = polygon.area / convex_hull.area
                            # Cap at 1.0 (perfect convexity)
                            convexity = min(convexity, 1.0)
                    except Exception as e:
                        print(f"Error calculating convexity for zone {zone_id}: {e}")
                
                # Calculate average distance from points to boundary
                avg_boundary_distance = 0
                if hasattr(polygon, 'boundary') and points_count > 0:
                    boundary = polygon.boundary
                    distances = []
                    
                    for _, row in zone_df.iterrows():
                        point = Point(row['Longitude'], row['Latitude'])
                        distance = point.distance(boundary)
                        distances.append(distance)
                    
                    avg_boundary_distance = np.mean(distances) if distances else 0
                
                # Store metrics
                boundary_metrics[zone_id] = {
                    "points_count": points_count,
                    "area": area,
                    "convexity": convexity,
                    "avg_boundary_distance": avg_boundary_distance,
                    "points_density": points_count / area if area > 0 else 0
                }

        # Create features for each zone
        features = []
        for zone_id, polygon in zone_polygons.items():
            if zone_id in zones:
                communes_dict = zone_communes.get(zone_id, {})
                communes_list = [f"{commune}: {count}" for commune, count in communes_dict.items()]
                
                # Include boundary metrics
                metrics = boundary_metrics.get(zone_id, {})
                
                # Calculate travel distance (in km) and workload time (in hours and minutes)
                workload_minutes = zone_workloads.get(zone_id, 0)
                points_count = len(zones[zone_id])
                
                # Calculate the travel component
                points_time_component = points_count * 15  # 15 minutes per point
                travel_time_component = workload_minutes - points_time_component
                
                # Travel distance in km (assuming 0.5 km/min speed)
                travel_distance_km = travel_time_component * 0.5
                
                # Format workload as hours and minutes
                hours = int(workload_minutes // 60)
                minutes = int(workload_minutes % 60)
                workload_formatted = f"{hours}h {minutes}m"

                properties = {
                    "zone_id": int(zone_id),
                    "points_count": int(points_count),
                    "workload_minutes": float(workload_minutes),
                    "workload_formatted": workload_formatted,
                    "travel_distance_km": float(travel_distance_km),
                    "avg_time_per_point": float(workload_minutes / max(1, points_count)),
                    "communes": communes_list,
                    "communes_count": len(communes_dict),
                    # Add boundary metrics
                    "convexity": float(metrics.get('convexity', 1.0)),
                    "avg_boundary_distance": float(metrics.get('avg_boundary_distance', 0)),
                    "area": float(metrics.get('area', 0)),
                    "points_density": float(metrics.get('points_density', 0))
                }

                geo_data = polygon.__geo_interface__ if hasattr(polygon, '__geo_interface__') else polygon
                features.append({
                    "type": "Feature",
                    "geometry": geo_data, 
                    "properties": properties
                })

        # Summary stats with boundary metrics
        all_communes = set()
        for zone_df in zones.values():
            if 'Commune' in zone_df.columns:
                all_communes.update(set(zone_df['Commune'].dropna()))
                
        # Calculate average boundary metrics
        avg_convexity = np.mean([m.get('convexity', 1.0) for m in boundary_metrics.values()]) if boundary_metrics else 1.0
        avg_boundary_distance = np.mean([m.get('avg_boundary_distance', 0) for m in boundary_metrics.values()]) if boundary_metrics else 0

        # Calculate total and average travel distances
        total_travel_distance = sum([float(f["properties"]["travel_distance_km"]) for f in features])
        avg_travel_distance = total_travel_distance / len(features) if features else 0

        summary = {
            "total_zones": len(zones),
            "total_points": len(df),
            "total_communes": len(all_communes),
            "total_travel_distance_km": float(total_travel_distance),
            "workload_min_minutes": min(zone_workloads.values()) if zone_workloads else 0,
            "workload_max_minutes": max(zone_workloads.values()) if zone_workloads else 0,
            "workload_avg_minutes": sum(zone_workloads.values()) / len(zone_workloads) if zone_workloads else 0,
            "workload_std_dev_minutes": float(np.std(list(zone_workloads.values()))) if zone_workloads else 0,
            "points_min": min([len(z) for z in zones.values()]) if zones else 0,
            "points_max": max([len(z) for z in zones.values()]) if zones else 0,
            "points_avg": len(df) / len(zones) if zones else 0,
            "points_std_dev": float(np.std([len(z) for z in zones.values()])) if zones else 0,
            # Boundary metrics summary
            "avg_convexity": float(avg_convexity),
            "avg_boundary_distance": float(avg_boundary_distance)
        }

        # Point features for individual locations
        point_features = []
        for zone_id, zone_df in zones.items():
            for _, point in zone_df.iterrows():
                commune_val = str(point['Commune']) if 'Commune' in point and pd.notna(point['Commune']) else "Unknown"
                point_features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(point['Longitude']), float(point['Latitude'])]
                    },
                    "properties": {
                        "zone_id": int(zone_id),
                        "commune": commune_val
                    }
                })

        # Final GeoJSON structure
        complete_geojson = {
            "type": "FeatureCollection",
            "metadata": summary,
            "points": {
                "type": "FeatureCollection",
                "features": point_features
            },
            "features": features 
        }
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(complete_geojson, f, indent=2)

            print(f"Exported zones to {output_path}")
        return complete_geojson

    except Exception as e:
        print(f"Error exporting to GeoJSON: {e}")
        return False



def create_map_visualization_from_geojson(geojson_path, output_file="territory_map.html"):
    """Create interactive map visualization with zone boundaries from GeoJSON file"""
    print(f"Creating map visualization at {output_file} from {geojson_path}...")
    
    try:
        # Load GeoJSON file
        with open(geojson_path, 'r') as f:
            geojson_data = json.load(f)
        
        # Extract metadata and features
        metadata = geojson_data.get('metadata', {})
        
        # Ensure zone_features is properly loaded
        zone_features = []
        raw_features = geojson_data.get('features', [])
        
        if not raw_features:
            print("Warning: No zone features found in GeoJSON")
        
        # Make sure features are properly parsed
        if isinstance(raw_features, str):
            try:
                parsed_features = json.loads(raw_features)
                if isinstance(parsed_features, list):
                    zone_features = parsed_features
                else:
                    print("Warning: Features JSON does not contain a list")
            except json.JSONDecodeError:
                print("Warning: Could not parse features as JSON")
        else:
            zone_features = raw_features
            
        # Validate zone features
        valid_zone_features = []
        for feature in zone_features:
            # Ensure feature has required properties
            if (isinstance(feature, dict) and 
                'properties' in feature and 
                'geometry' in feature and
                'zone_id' in feature.get('properties', {})):
                valid_zone_features.append(feature)
        
        if len(valid_zone_features) < len(zone_features):
            print(f"Warning: {len(zone_features) - len(valid_zone_features)} invalid zone features removed")
        
        zone_features = valid_zone_features
                
        # Parse point features with validation
        point_features = []
        points_data = geojson_data.get('points', {})
        
        if isinstance(points_data, str):
            try:
                parsed_points = json.loads(points_data)
                if isinstance(parsed_points, dict) and 'features' in parsed_points:
                    raw_point_features = parsed_points.get('features', [])
                    
                    # Validate each point feature
                    for point in raw_point_features:
                        if (isinstance(point, dict) and 
                            'geometry' in point and 
                            'coordinates' in point.get('geometry', {}) and
                            'properties' in point and
                            'zone_id' in point.get('properties', {})):
                            point_features.append(point)
                else:
                    print("Warning: Points JSON does not have expected format")
            except json.JSONDecodeError:
                print("Warning: Could not parse points as JSON")
        elif isinstance(points_data, dict) and 'features' in points_data:
            raw_point_features = points_data.get('features', [])
            
            # Validate each point feature
            for point in raw_point_features:
                if (isinstance(point, dict) and 
                    'geometry' in point and 
                    'coordinates' in point.get('geometry', {}) and
                    'properties' in point and
                    'zone_id' in point.get('properties', {})):
                    point_features.append(point)
        
        if not point_features:
            print("Warning: No valid point features found in GeoJSON")
        
        # Calculate map center from points with validation
        lats = []
        lons = []
        
        for feature in point_features:
            coords = feature.get('geometry', {}).get('coordinates', [])
            if isinstance(coords, list) and len(coords) >= 2:
                # GeoJSON coordinates are [longitude, latitude]
                lon, lat = coords[0], coords[1]
                if isinstance(lon, (int, float)) and isinstance(lat, (int, float)):
                    lons.append(lon)
                    lats.append(lat)
        
        if lats and lons:
            center_lat = sum(lats) / len(lats)
            center_lon = sum(lons) / len(lons)
        else:
            # Try to determine center from zone features if no points available
            all_coords = []
            for feature in zone_features:
                geometry = feature.get('geometry', {})
                if geometry.get('type') == 'Polygon':
                    coords = geometry.get('coordinates', [[]])
                    if coords and isinstance(coords[0], list):
                        all_coords.extend(coords[0])
                
            if all_coords:
                lons = [coord[0] for coord in all_coords if isinstance(coord, list) and len(coord) >= 2]
                lats = [coord[1] for coord in all_coords if isinstance(coord, list) and len(coord) >= 2]
                if lats and lons:
                    center_lat = sum(lats) / len(lats)
                    center_lon = sum(lons) / len(lons)
                else:
                    center_lat = 0
                    center_lon = 0
            else:
                center_lat = 0
                center_lon = 0
        
        print(f"Map center: ({center_lat}, {center_lon})")
        
        # Create base map
        mymap = folium.Map(location=[center_lat, center_lon], zoom_start=11)
        
        # Extract zone IDs and generate colors
        zone_ids = set()
        for feature in zone_features:
            zone_id = feature.get('properties', {}).get('zone_id')
            if zone_id is not None:
                zone_ids.add(zone_id)
        
        if not zone_ids:
            print("Warning: No valid zone IDs found")
        
        num_zones = len(zone_ids)
        colors = plt.cm.rainbow(np.linspace(0, 1, max(num_zones, 1)))
        zone_colors = {i: f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}' 
                      for i, (r, g, b, _) in enumerate(colors)}
        
        # Add zone polygons first (below points)
        for feature in zone_features:
            properties = feature.get('properties', {})
            zone_id = properties.get('zone_id')
            
            if zone_id in zone_colors:
                color = zone_colors[zone_id]
                
                # Get workload in hours and minutes format and travel distance
                workload_formatted = properties.get('workload_formatted', 'N/A')
                travel_distance = properties.get('travel_distance_km', 0)
                
                # Create tooltip with zone info
                communes_str = properties.get('communes', [])
                communes_html = "<br>".join(communes_str[:5]) if isinstance(communes_str, list) else ""
                
                tooltip_html = f"""
                <div style="font-family: Arial; font-size: 12px; max-width: 300px;">
                    <h4>Zone {zone_id}</h4>
                    <b>Points:</b> {properties.get('points_count', 0)}<br>
                    <b>Workload:</b> {workload_formatted}<br>
                    <b>Travel Distance:</b> {travel_distance:.2f} km<br>
                    <b>Communes ({properties.get('communes_count', 0)}):</b><br>
                    {communes_html}
                </div>
                """
                
                # Add polygon with styling (just the fill, no internal borders)
                folium.GeoJson(
                    feature,
                    style_function=lambda x, color=color: {
                        'fillColor': color,
                        'color': color,
                        'weight': 0,
                        'fillOpacity': 0.3
                    },
                    tooltip=folium.Tooltip(tooltip_html),
                    highlight_function=lambda x: {'weight': 0, 'fillOpacity': 0.5}
                ).add_to(mymap)
                
                print(f"Added zone {zone_id} polygon to map")
            else:
                print(f"Warning: Zone ID {zone_id} not in color scheme")
        
        # Add points grouped by zone with commune information
        print(f"Processing {len(point_features)} points...")
        zone_points = {}
        for feature in point_features:
            properties = feature.get('properties', {})
            zone_id = properties.get('zone_id')
            
            if zone_id is not None:
                if zone_id not in zone_points:
                    zone_points[zone_id] = []
                
                zone_points[zone_id].append(feature)
        
        # Add points for each zone
        for zone_id, points in zone_points.items():
            if zone_id in zone_colors:
                color = zone_colors[zone_id]
                
                # Create a feature group for this zone's points
                fg = folium.FeatureGroup(name=f"Zone {zone_id} Points", show=True)
                
                # Add all points for this zone
                points_added = 0
                for point in points:
                    coords = point.get('geometry', {}).get('coordinates', [])
                    if isinstance(coords, list) and len(coords) >= 2:
                        lon, lat = coords[0], coords[1]
                        
                        if not (isinstance(lat, (int, float)) and isinstance(lon, (int, float))):
                            continue
                            
                        properties = point.get('properties', {})
                        commune = properties.get('commune', 'Unknown')
                        
                        # Create popup with point info
                        popup_html = f"""
                        <div style="font-family: Arial; font-size: 12px;">
                            <b>Zone:</b> {zone_id}<br>
                            <b>Commune:</b> {commune}
                        </div>
                        """
                        
                        try:
                            folium.CircleMarker(
                                location=[lat, lon],  # GeoJSON is [lon, lat]
                                radius=2,
                                color=color,
                                fill=True,
                                fill_color=color,
                                fill_opacity=0.8,
                                weight=1,
                                popup=folium.Popup(popup_html, max_width=200)
                            ).add_to(fg)
                            points_added += 1
                        except Exception as e:
                            print(f"Error adding point: {e}")
                
                fg.add_to(mymap)
            else:
                print(f"Warning: Zone ID {zone_id} not in color scheme for points")
        
        # Add explicit zone boundaries with bold lines
        for feature in zone_features:
            properties = feature.get('properties', {})
            zone_id = properties.get('zone_id')
            
            if zone_id in zone_colors:
                # Add explicit boundary line with bold lines
                folium.GeoJson(
                    feature,
                    style_function=lambda x: {
                        'fillColor': 'none',
                        'color': '#000000',  # Black border
                        'weight': 2.5,       # Border thickness
                        'opacity': 0.7,      # Border opacity
                        'fillOpacity': 0     # No fill
                    }
                ).add_to(mymap)
        
        # Add layer control to toggle visibility
        folium.LayerControl().add_to(mymap)
        
        # Add title with stats
        num_points = metadata.get('total_points', 0)
        total_travel_distance = metadata.get('total_travel_distance_km', 0)
        
        title_html = '''
        <div style="position: fixed; 
                    top: 10px; left: 50px; width: 220px; height: 100px; 
                    background-color: white; border-radius: 5px;
                    border: 2px solid grey; z-index: 9999; padding: 10px;">
            <h4 style="margin: 0; text-align: center;">Territory Map</h4>
            <p style="margin: 5px 0 0 0; font-size: 12px; text-align: center;">
                {num_zones} Zones • {num_points} Points<br>
                Total Travel: {total_distance:.1f} km
            </p>
        </div>
        '''.format(num_zones=num_zones, num_points=num_points, total_distance=total_travel_distance)
        
        mymap.get_root().html.add_child(folium.Element(title_html))
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px;
                    background-color: white; border-radius: 5px;
                    border: 2px solid grey; z-index: 9999; padding: 10px;">
            <h5 style="margin: 0 0 5px 0;">Legend</h5>
        '''
        
        # Sort zone features by ID for consistent legend
        sorted_features = sorted(zone_features, 
                                key=lambda x: x.get('properties', {}).get('zone_id', 0))
        
        for feature in sorted_features:
            properties = feature.get('properties', {})
            zone_id = properties.get('zone_id')
            
            if zone_id in zone_colors:
                color = zone_colors[zone_id]
                points = properties.get('points_count', 0)
                workload_formatted = properties.get('workload_formatted', 'N/A')
                travel_distance = properties.get('travel_distance_km', 0)
                
                legend_html += f'''
                <div>
                    <span style="display: inline-block; width: 12px; height: 12px; 
                                background-color: {color}; margin-right: 5px;"></span>
                    <span style="font-size: 12px;">Zone {zone_id}: {points} pts, {workload_formatted}, {travel_distance:.1f} km</span>
                </div>
                '''
        
        legend_html += '</div>'
        mymap.get_root().html.add_child(folium.Element(legend_html))
        
        # Save the map
        mymap.save(output_file)
        print(f"Map visualization saved to {output_file}")
        return True
        
    except Exception as e:
        print(f"Error creating map from GeoJSON: {e}")
        import traceback
        traceback.print_exc()
        return False
    


def assign_communes_from_geojson(df, geojson_path, lat_col="Latitude", lon_col="Longitude"):
    """Assign commune to each point based on GeoJSON boundaries if not already available or marked as 'Unknown'."""
    try:
        # Ensure Commune column exists
        if 'Commune' not in df.columns:
            df['Commune'] = pd.NA
        
        # Normalize unassigned communes: treat "Unknown" as missing
        df['Commune'] = df['Commune'].replace("Unknown", pd.NA)
        
        # Quick exit if all communes are already filled
        if df['Commune'].notna().all():
            print("All points already have communes.")
            return df

        # Load GeoJSON file with commune boundaries
        communes = gpd.read_file(geojson_path)
        
        # Create GeoDataFrame
        geometry = [Point(lon, lat) for lon, lat in zip(df[lon_col], df[lat_col])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

        
        # Unassigned mask
        unassigned_mask = gdf['Commune'].isna()
        
        if unassigned_mask.any():
            unassigned_gdf = gdf[unassigned_mask]
            joined = gpd.sjoin(unassigned_gdf, communes, how="left", predicate="within")
            df.loc[unassigned_mask, 'Commune'] = joined['shapeName'].values
            
            # Nearest fallback
            still_unassigned = df['Commune'].isna()
            if still_unassigned.any():
                for idx, row in df[still_unassigned].iterrows():
                    point = Point(row[lon_col], row[lat_col])
                    communes['distance'] = communes.geometry.distance(point)
                    closest = communes.loc[communes['distance'].idxmin()]
                    df.at[idx, 'Commune'] = closest['shapeName']
        return df

    except Exception as e:
        print(f"Error assigning communes: {e}")
        df['Commune'] = df['Commune'].fillna('Placeholder_Commune')
        return df



def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Balanced Territory Zoning System')
    parser.add_argument('--csv', required=True, help='Path to input CSV file with coordinates')
    parser.add_argument('--lat', default="Latitude", help='Column name for latitude (default: Latitude)')
    parser.add_argument('--lon', default="Longitude", help='Column name for longitude (default: Longitude)')
    parser.add_argument('--zones', type=int, default=15, help='Number of zones to create (default: 15)')
    parser.add_argument('--balance', default="points", choices=['points', 'distance'], 
                        help='Balance priority: "points" or "distance" (default: points)')
    parser.add_argument('--output', default="territory_map.html", help='Output map filename (default: territory_map.html)')
    parser.add_argument('--geojson', default="zones.geojson", help='Output GeoJSON filename (default: zones.geojson)')
    parser.add_argument('--commune_geojson', help='Optional GeoJSON file with commune boundaries')
    
    args = parser.parse_args()
    
    # Load data from CSV
    print(f"Loading data from {args.csv}...")
    df = load_data(args.csv, args.lat, args.lon)
    print(f"Loaded {len(df)} points")
    
    # Create balanced zones
    df, zones, zone_workloads, zone_communes = create_balanced_zones(
        df, args.zones, args.lat, args.lon
    )
    
    # Generate zone boundaries
    zone_polygons = generate_zone_boundaries(zones)
    
    # Export zones to GeoJSON with commune assignment if provided
    export_zones_to_geojson(df, zones, zone_workloads, zone_communes, zone_polygons, 
                           args.geojson, "geoBoundaries-DZA-ADM3.geojson")
    
    # Create map visualization from the GeoJSON fileex
    create_map_visualization_from_geojson(args.geojson, args.output)
    
    print(f"Done! Map created at {args.output}")

parser = argparse.ArgumentParser(description='Balanced Territory Zoning System')
parser.add_argument('--csv', required=True, help='Path to input CSV file with coordinates')
parser.add_argument('--lat', default="Latitude", help='Column name for latitude (default: Latitude)')
parser.add_argument('--lon', default="Longitude", help='Column name for longitude (default: Longitude)')
parser.add_argument('--zones', type=int, default=15, help='Number of zones to create (default: 15)')
parser.add_argument('--balance', default="points", choices=['points', 'distance'], 
                    help='Balance priority: "points" or "distance" (default: points)')
parser.add_argument('--output', default="territory_map.html", help='Output map filename (default: territory_map.html)')
parser.add_argument('--geojson', default="zones.geojson", help='Output GeoJSON filename (default: zones.geojson)')
parser.add_argument('--commune_geojson', help='Optional GeoJSON file with commune boundaries')
parser = argparse.ArgumentParser(description='Balanced Territory Zoning System')
parser.add_argument('--csv', required=True, help='Path to input CSV file with coordinates')
parser.add_argument('--lat', default="Latitude", help='Column name for latitude (default: Latitude)')
parser.add_argument('--lon', default="Longitude", help='Column name for longitude (default: Longitude)')
parser.add_argument('--zones', type=int, default=15, help='Number of zones to create (default: 15)')
parser.add_argument('--balance', default="points", choices=['points', 'distance'], 
                    help='Balance priority: "points" or "distance" (default: points)')
parser.add_argument('--output', default="territory_map.html", help='Output map filename (default: territory_map.html)')
parser.add_argument('--geojson', default="zones.geojson", help='Output GeoJSON filename (default: zones.geojson)')
parser.add_argument('--commune_geojson', help='Optional GeoJSON file with commune boundaries')

# args = parser.parse_args()

# # Load data from CSV
# # print(f"Loading data from {args.csv}...")
# df = load_data("coordinates.csv","Latitude", "Longitude")
# print(f"Loaded {len(df)} points")

# # Create balanced zones
# df, zones, zone_workloads, zone_communes = create_balanced_zones(
#     df, 25, "Latitude", "Longitude", distance_coef=0.1, points_coef=10)
# # args = parser.parse_args()

# zone_polygons = generate_zone_boundaries(zones)
    
#     # Export zones to GeoJSON with commune assignment if provided
# export_zones_to_geojson(df, zones, zone_workloads, zone_communes, zone_polygons, 
#                         "zones.geojson", "geoBoundaries-DZA-ADM3.geojson")

# # Create map visualization from the GeoJSON file
# create_map_visualization_from_geojson("zones.geojson", "territory_map.html")
