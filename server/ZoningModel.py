import pandas as pd
import numpy as np
import geopandas as gpd
import folium
import matplotlib
import matplotlib.pyplot as plt
import os
import argparse
from shapely.geometry import Point
from collections import defaultdict
from itertools import combinations, cycle, permutations
from sklearn.cluster import KMeans
import random

# Disable multithreading for scikit-learn
os.environ["OMP_NUM_THREADS"] = "1"




def validate(df, lat_col="Latitude", lon_col="Longitude"):
    """Load point data from CSV file and ensure required columns exist"""
    try:
        # Ensure coordinates are numeric
        df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
        df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
        
        # Drop rows with invalid coordinates
        invalid_rows = df[df[lat_col].isna() | df[lon_col].isna()].shape[0]
        if invalid_rows > 0:
            print(f"Warning: Dropped {invalid_rows} rows with invalid coordinates")
            df = df.dropna(subset=[lat_col, lon_col])
        
        # Check if any data remains
        if df.empty:
            raise ValueError("No valid coordinate data found in CSV")
            
        # Create a 'Commune' column if it doesn't exist (will be populated later)
        if 'Commune' not in df.columns:
            df['Commune'] = None
            
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        raise

def assign_communes_from_geojson(df, geojson_path, lat_col="Latitude", lon_col="Longitude"):
    """Assign commune to each point based on GeoJSON boundaries"""
    try:
        # Load GeoJSON file with commune boundaries
        communes = gpd.read_file(geojson_path)
        
        # Create GeoDataFrame from DataFrame
        geometry = [Point(lon, lat) for lon, lat in zip(df[lon_col], df[lat_col])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry)
        
        # Spatial join to find commune for each point
        joined = gpd.sjoin(gdf, communes, how="left", predicate="within")
        
        # Update the original DataFrame with commune information
        df['Commune'] = joined['shapeName']
        
        # Count points with assigned communes
        assigned = df['Commune'].notna().sum()
        total = len(df)
        print(f"Assigned communes to {assigned} out of {total} points ({assigned/total*100:.1f}%)")
        
        # For unassigned points, find the nearest commune
        if assigned < total:
            print("Assigning remaining points to nearest commune...")
            unassigned = df[df['Commune'].isna()]
            
            for idx, row in unassigned.iterrows():
                point = Point(row[lon_col], row[lat_col])
                # Calculate distances to all commune polygons
                communes['distance'] = communes.geometry.apply(lambda g: g.distance(point))
                # Find the closest commune
                closest = communes.loc[communes['distance'].idxmin()]
                df.loc[idx, 'Commune'] = closest['shapeName']
                
        return df
    except Exception as e:
        print(f"Error assigning communes: {e}")
        # If GeoJSON processing fails, assign random/placeholder communes
        print("Using placeholder commune names")
        if 'Commune' not in df.columns or df['Commune'].isna().all():
            df['Commune'] = 'Placeholder_Commune'
        return df

# Import required functions from the original model
# Note: These functions are used directly from the original code
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate haversine distance between two points in kilometers with improved precision"""
    # Convert latitude and longitude from degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    
    return c * r
def pairwise_distance_matrix(cluster_df):
    """Create a matrix of pairwise distances between all points in a cluster"""
    coords = cluster_df[['Latitude', 'Longitude']].values
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            dist = haversine_distance(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
            
    return dist_matrix

def calculate_cluster_distance_mst(cluster_df):
    """Calculate estimated internal distance using Minimum Spanning Tree approach"""
    if len(cluster_df) <= 1:
        return 0
    
    # For very small clusters, use direct calculation
    if len(cluster_df) <= 3:
        coords = cluster_df[['Latitude', 'Longitude']].values
        total_dist = 0
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                total_dist += haversine_distance(coords[i][0], coords[i][1], 
                                               coords[j][0], coords[j][1])
        return total_dist / max(1, len(coords) - 1)
        
    # Get pairwise distance matrix
    dist_matrix = pairwise_distance_matrix(cluster_df)
    
    # Implement a simple MST algorithm (Prim's algorithm)
    n = len(dist_matrix)
    visited = [False] * n
    visited[0] = True
    mst_weight = 0
    
    for _ in range(n-1):
        min_edge = float('inf')
        min_i, min_j = 0, 0
        
        for i in range(n):
            if visited[i]:
                for j in range(n):
                    if not visited[j] and dist_matrix[i][j] > 0 and dist_matrix[i][j] < min_edge:
                        min_edge = dist_matrix[i][j]
                        min_i, min_j = i, j
        
        mst_weight += min_edge
        visited[min_j] = True
    
    # Add a complexity factor based on number of points and spatial distribution
    spatial_dispersion = np.std(dist_matrix.flatten())
    complexity_factor = 1.2 + 0.05 * np.log(n) + 0.1 * spatial_dispersion
    
    return mst_weight * complexity_factor

def calculate_rep_total_distance(cluster_list, inter_cluster_factor=1.5):
    """Calculate total distance in km for a representative's territory with improved routing"""
    if not cluster_list:
        return 0
    
    # Calculate internal cluster distances (improved method)
    intra_distance = sum(calculate_cluster_distance_mst(cluster) for cluster in cluster_list)
    
    # Calculate inter-cluster distances if more than one cluster
    inter_distance = 0
    if len(cluster_list) > 1:
        centroids = [cluster[['Latitude', 'Longitude']].mean().values for cluster in cluster_list]
        
        # If few clusters, try all permutations to find optimal route
        if len(centroids) <= 5:
            min_route_distance = float('inf')
            for route in permutations(range(len(centroids))):
                route_distance = 0
                for i in range(len(route) - 1):
                    c1, c2 = centroids[route[i]], centroids[route[i+1]]
                    route_distance += haversine_distance(c1[0], c1[1], c2[0], c2[1])
                if route_distance < min_route_distance:
                    min_route_distance = route_distance
            inter_distance = min_route_distance
        else:
            # For many clusters, use a greedy approach
            # Start from largest cluster
            cluster_sizes = [len(cluster) for cluster in cluster_list]
            start_idx = cluster_sizes.index(max(cluster_sizes))
            
            remaining = set(range(len(centroids)))
            current = start_idx
            remaining.remove(current)
            route_distance = 0
            
            while remaining:
                # Find nearest unvisited centroid
                min_dist = float('inf')
                nearest = -1
                for next_idx in remaining:
                    dist = haversine_distance(centroids[current][0], centroids[current][1],
                                           centroids[next_idx][0], centroids[next_idx][1])
                    if dist < min_dist:
                        min_dist = dist
                        nearest = next_idx
                
                route_distance += min_dist
                current = nearest
                remaining.remove(current)
            
            inter_distance = route_distance
        
        # Apply multiplier to account for non-direct routes and stopping time
        inter_distance *= inter_cluster_factor
    
    return intra_distance + inter_distance

def optimize_cluster_assignments(clusters_dict, num_reps, max_iterations=40):
    """
    Optimize cluster assignments to minimize workload variance and min-max ratio
    with improved performance and better results.
    """
    # Create a cache for cluster workloads to avoid recalculating
    cluster_workloads = {cid: calculate_cluster_distance_mst(cluster) 
                         for cid, cluster in clusters_dict.items()}
    
    # Sort clusters by workload (largest first) for better initial assignment
    sorted_clusters = sorted([(cid, workload) for cid, workload in cluster_workloads.items()], 
                           key=lambda x: -x[1])
    
    # Try multiple starting points and pick the best one
    best_assignments = None
    best_metrics = (float('inf'), float('inf'))  # (std_dev, max_min_ratio)
    
    # Try different initialization strategies
    for init_strategy in range(3):
        # Initialize assignments based on strategy
        current_assignments = {i: [] for i in range(1, num_reps + 1)}
        
        if init_strategy == 0:
            # Strategy 1: Greedy assignment (largest first)
            rep_workloads = {i: 0 for i in range(1, num_reps + 1)}
            for cluster_id, workload in sorted_clusters:
                min_rep = min(rep_workloads, key=rep_workloads.get)
                current_assignments[min_rep].append(cluster_id)
                rep_workloads[min_rep] += workload
                
        elif init_strategy == 1:
            # Strategy 2: Round-robin with largest clusters
            rep_cycle = cycle(range(1, num_reps + 1))
            for cluster_id, _ in sorted_clusters:
                rep = next(rep_cycle)
                current_assignments[rep].append(cluster_id)
                
        else:
            # Strategy 3: Balanced pair assignment
            # Sort reps in pairs, assign large+small to each rep
            paired_clusters = []
            half = len(sorted_clusters) // 2
            for i in range(half):
                if i < len(sorted_clusters) - i - 1:
                    paired_clusters.append((sorted_clusters[i][0], sorted_clusters[len(sorted_clusters) - i - 1][0]))
            
            # Handle odd number of clusters
            if len(sorted_clusters) % 2 == 1:
                paired_clusters.append((sorted_clusters[half][0],))
                
            rep_cycle = cycle(range(1, num_reps + 1))
            for cluster_pair in paired_clusters:
                rep = next(rep_cycle)
                for cluster_id in cluster_pair:
                    current_assignments[rep].append(cluster_id)
        
        # Calculate initial metrics for this strategy
        std_dev, max_min_ratio, current_workloads = evaluate_workload_balance(
            clusters_dict, current_assignments)
        
        # Local optimization for each initialization strategy
        current_assignments, std_dev, max_min_ratio, current_workloads = local_search_optimization(
            clusters_dict, current_assignments, cluster_workloads, max_iterations)
        
        # Update best if this strategy produced better results
        if (std_dev + max_min_ratio * 0.5) < (best_metrics[0] + best_metrics[1] * 0.5):
            best_assignments = {k: v.copy() for k, v in current_assignments.items()}
            best_metrics = (std_dev, max_min_ratio)
    
    return best_assignments

def local_search_optimization(clusters_dict, initial_assignments, cluster_workloads, max_iterations):
    """
    Perform local search optimization with enhanced move strategies and faster evaluation.
    """
    current_assignments = {k: v.copy() for k, v in initial_assignments.items()}
    
    # Calculate initial workloads more efficiently
    rep_cluster_lists = {
        rep: [clusters_dict[cid] for cid in cluster_ids if cid in clusters_dict]
        for rep, cluster_ids in current_assignments.items()
    }
    current_workloads = {
        rep: calculate_rep_total_distance(clusters) 
        for rep, clusters in rep_cluster_lists.items()
    }
    
    std_dev = np.std(list(current_workloads.values())) if current_workloads else float('inf')
    max_workload = max(current_workloads.values()) if current_workloads else 0
    min_workload = min(current_workloads.values()) if current_workloads else 0
    max_min_ratio = max_workload / max(0.001, min_workload)
    
    # Track improvement
    last_improvement = 0
    best_std_dev = std_dev
    best_max_min_ratio = max_min_ratio
    best_assignments = {k: v.copy() for k, v in current_assignments.items()}
    
    # Fast approximation for estimating rep workload changes
    def estimate_workload_change(rep_id, add_cluster_id=None, remove_cluster_id=None):
        """Quickly estimate workload change without full recalculation"""
        # This is just an estimation - accurate calculation happens when moves are committed
        if add_cluster_id is not None and remove_cluster_id is not None:
            # Swap case - this is just an approximation
            return cluster_workloads.get(add_cluster_id, 0) - cluster_workloads.get(remove_cluster_id, 0)
        elif add_cluster_id is not None:
            # Add case
            return cluster_workloads.get(add_cluster_id, 0)
        elif remove_cluster_id is not None:
            # Remove case
            return -cluster_workloads.get(remove_cluster_id, 0)
        return 0
    
    # Main optimization loop
    for iteration in range(max_iterations):
        improvement_found = False
        
        # Sort reps by workload for better move selection
        sorted_reps = sorted(current_workloads.items(), key=lambda x: x[1])
        min_rep, min_workload = sorted_reps[0]
        max_rep, max_workload = sorted_reps[-1]
        
        # Early termination if already well-balanced
        current_max_min_ratio = max_workload / max(0.001, min_workload)
        if std_dev < 0.3 and current_max_min_ratio < 1.1:
            break
            
        # If no improvements for several iterations, stop
        if iteration - last_improvement > max(5, max_iterations // 8):
            break
        
        # PHASE 1: Try moves from most loaded to least loaded rep
        workload_diff = max_workload - min_workload
        
        # Only consider significant clusters (not too small)
        min_impact = workload_diff * 0.05  # 5% of the difference
        significant_clusters = []
        
        for cluster_id in current_assignments[max_rep]:
            if cluster_id in cluster_workloads:
                cluster_workload = cluster_workloads[cluster_id]
                if cluster_workload >= min_impact and cluster_workload <= workload_diff * 0.9:
                    significant_clusters.append((cluster_id, cluster_workload))
        
        # Sort clusters by how well they'd balance the workload
        # We want clusters that would make the workloads as close as possible
        target_transfer = workload_diff / 2
        significant_clusters.sort(key=lambda x: abs(x[1] - target_transfer))
        
        # Try most promising moves first
        for cluster_id, cluster_workload in significant_clusters[:min(5, len(significant_clusters))]:
            # Estimate new workloads if we move this cluster
            est_new_max = max_workload - cluster_workload
            est_new_min = min_workload + cluster_workload
            
            # Only evaluate if it looks promising
            if est_new_max >= est_new_min:  # Don't over-correct
                # Perform the move
                current_assignments[max_rep].remove(cluster_id)
                current_assignments[min_rep].append(cluster_id)
                
                # Calculate true new workloads (more accurate than estimates)
                new_max_clusters = [clusters_dict[c] for c in current_assignments[max_rep] if c in clusters_dict]
                new_min_clusters = [clusters_dict[c] for c in current_assignments[min_rep] if c in clusters_dict]
                
                new_max_workload = calculate_rep_total_distance(new_max_clusters)
                new_min_workload = calculate_rep_total_distance(new_min_clusters)
                
                # Update workloads dictionary
                current_workloads[max_rep] = new_max_workload
                current_workloads[min_rep] = new_min_workload
                
                # Calculate new metrics
                new_std_dev = np.std(list(current_workloads.values()))
                new_max = max(current_workloads.values())
                new_min = min(current_workloads.values())
                new_max_min_ratio = new_max / max(0.001, new_min)
                
                # Accept if better, otherwise revert
                if new_std_dev <= std_dev or new_max_min_ratio < max_min_ratio:
                    std_dev = new_std_dev
                    max_min_ratio = new_max_min_ratio
                    last_improvement = iteration
                    improvement_found = True
                    
                    # Keep track of best solution seen
                    if (new_std_dev + new_max_min_ratio * 0.5) < (best_std_dev + best_max_min_ratio * 0.5):
                        best_std_dev = new_std_dev
                        best_max_min_ratio = new_max_min_ratio
                        best_assignments = {k: v.copy() for k, v in current_assignments.items()}
                    
                    break
                else:
                    # Revert the move
                    current_assignments[min_rep].remove(cluster_id)
                    current_assignments[max_rep].append(cluster_id)
                    current_workloads[max_rep] = max_workload
                    current_workloads[min_rep] = min_workload
        
        # PHASE 2: If no simple move worked, try swaps between very unbalanced pairs
        if not improvement_found:
            # Look at the most unbalanced pairs first
            rep_pairs = []
            for i in range(len(sorted_reps) - 1):
                for j in range(i + 1, len(sorted_reps)):
                    rep1, workload1 = sorted_reps[i]
                    rep2, workload2 = sorted_reps[j]
                    imbalance = abs(workload1 - workload2)
                    if imbalance >= min_impact:
                        rep_pairs.append((rep1, rep2, imbalance))
            
            # Sort by imbalance (highest first)
            rep_pairs.sort(key=lambda x: -x[2])
            
            # Try swaps on the most promising pairs
            for rep1, rep2, _ in rep_pairs[:min(3, len(rep_pairs))]:
                workload1 = current_workloads[rep1]
                workload2 = current_workloads[rep2]
                target_transfer = abs(workload1 - workload2) / 2
                
                # Find best possible swap
                best_swap = None
                best_swap_score = float('inf')
                
                # Limit candidates for performance
                candidates1 = sorted([(c, cluster_workloads.get(c, 0)) for c in current_assignments[rep1] 
                                   if c in cluster_workloads], key=lambda x: -x[1])[:4]
                candidates2 = sorted([(c, cluster_workloads.get(c, 0)) for c in current_assignments[rep2] 
                                   if c in cluster_workloads], key=lambda x: -x[1])[:4]
                
                for (cluster1, w1) in candidates1:
                    for (cluster2, w2) in candidates2:
                        # Skip if the workloads are very similar
                        delta_workload = w1 - w2
                        balance_impact = abs((workload1 - w1 + w2) - (workload2 - w2 + w1))
                        
                        # Score this swap by how well it balances
                        swap_score = balance_impact
                        
                        if swap_score < best_swap_score:
                            best_swap = (cluster1, cluster2)
                            best_swap_score = swap_score
                
                # If found a good swap, apply it
                if best_swap and best_swap_score < abs(workload1 - workload2):
                    cluster1, cluster2 = best_swap
                    
                    # Execute swap
                    current_assignments[rep1].remove(cluster1)
                    current_assignments[rep1].append(cluster2)
                    current_assignments[rep2].remove(cluster2)
                    current_assignments[rep2].append(cluster1)
                    
                    # Recalculate workloads
                    rep1_clusters = [clusters_dict[c] for c in current_assignments[rep1] if c in clusters_dict]
                    rep2_clusters = [clusters_dict[c] for c in current_assignments[rep2] if c in clusters_dict]
                    current_workloads[rep1] = calculate_rep_total_distance(rep1_clusters)
                    current_workloads[rep2] = calculate_rep_total_distance(rep2_clusters)
                    
                    # Calculate new metrics
                    new_std_dev = np.std(list(current_workloads.values()))
                    new_max = max(current_workloads.values())
                    new_min = min(current_workloads.values())
                    new_max_min_ratio = new_max / max(0.001, new_min)
                    
                    if new_std_dev <= std_dev or new_max_min_ratio < max_min_ratio:
                        std_dev = new_std_dev
                        max_min_ratio = new_max_min_ratio
                        last_improvement = iteration
                        improvement_found = True
                        
                        # Keep track of best solution
                        if (new_std_dev + new_max_min_ratio * 0.5) < (best_std_dev + best_max_min_ratio * 0.5):
                            best_std_dev = new_std_dev
                            best_max_min_ratio = new_max_min_ratio
                            best_assignments = {k: v.copy() for k, v in current_assignments.items()}
                        
                        break
                    else:
                        # Revert the swap
                        current_assignments[rep1].remove(cluster2)
                        current_assignments[rep1].append(cluster1)
                        current_assignments[rep2].remove(cluster1)
                        current_assignments[rep2].append(cluster2)
                        current_workloads[rep1] = workload1
                        current_workloads[rep2] = workload2
            
                if improvement_found:
                    break
        
        # PHASE 3: If we still see no improvement, try occasional randomization to escape local optima
        if not improvement_found and iteration > 5 and iteration % 5 == 0:
            # Choose a random high-workload rep and low-workload rep
            high_reps = sorted_reps[-min(3, len(sorted_reps)):]
            low_reps = sorted_reps[:min(3, len(sorted_reps))]
            
            if high_reps and low_reps:
                high_rep, high_workload = random.choice(high_reps)
                low_rep, low_workload = random.choice(low_reps)
                
                # Try to find a random cluster from high_rep that's not too large
                high_rep_clusters = [(cid, cluster_workloads.get(cid, 0)) 
                                  for cid in current_assignments[high_rep] 
                                  if cid in cluster_workloads]
                
                if high_rep_clusters:
                    # Filter to reasonably sized clusters
                    candidates = [c for c in high_rep_clusters 
                               if c[1] <= (high_workload - low_workload) * 0.9]
                    
                    if candidates:
                        # Pick a random candidate
                        cluster_id, _ = random.choice(candidates)
                        
                        # Move it
                        current_assignments[high_rep].remove(cluster_id)
                        current_assignments[low_rep].append(cluster_id)
                        
                        # Recalculate workloads
                        high_rep_clusters = [clusters_dict[c] for c in current_assignments[high_rep] if c in clusters_dict]
                        low_rep_clusters = [clusters_dict[c] for c in current_assignments[low_rep] if c in clusters_dict]
                        current_workloads[high_rep] = calculate_rep_total_distance(high_rep_clusters)
                        current_workloads[low_rep] = calculate_rep_total_distance(low_rep_clusters)
                        
                        # Calculate new metrics
                        std_dev = np.std(list(current_workloads.values()))
                        max_workload = max(current_workloads.values())
                        min_workload = min(current_workloads.values())
                        max_min_ratio = max_workload / max(0.001, min_workload)
    
    # Return the best solution found
    std_dev, max_min_ratio, current_workloads = evaluate_workload_balance(
        clusters_dict, best_assignments)
    
    return best_assignments, std_dev, max_min_ratio, current_workloads

def greedy_assign_clusters(clusters_dict, num_reps):
    """Assign clusters to representatives using a greedy approach"""
    if num_reps <= 0:
        return {}
    
    # Calculate workload for each cluster
    cluster_workloads = {cid: calculate_cluster_distance_mst(cluster) for cid, cluster in clusters_dict.items()}
    
    # Initialize empty assignments for each rep
    rep_workloads = {i: 0 for i in range(1, num_reps + 1)}
    rep_clusters = {i: [] for i in range(1, num_reps + 1)}
    
    # Sort clusters by workload (descending)
    sorted_clusters = sorted(cluster_workloads.items(), key=lambda x: -x[1])
    
    # Assign clusters using a greedy approach (assign to rep with lowest workload)
    for cluster_id, _ in sorted_clusters:
        min_workload_rep = min(rep_workloads, key=rep_workloads.get)
        rep_clusters[min_workload_rep].append(cluster_id)
        rep_workloads[min_workload_rep] += cluster_workloads[cluster_id]
    
    return rep_clusters

def evaluate_workload_balance(clusters_dict, rep_clusters):
    """Evaluate the balance of workload among representatives"""
    workloads = {}
    
    for rep_id, cluster_ids in rep_clusters.items():
        rep_cluster_list = [clusters_dict[cid] for cid in cluster_ids if cid in clusters_dict]
        workloads[rep_id] = calculate_rep_total_distance(rep_cluster_list)
    
    workload_values = list(workloads.values())
    if not workload_values or max(workload_values) == 0:
        return float('inf'), float('inf'), workloads
    
    std_dev = np.std(workload_values)
    max_min_ratio = max(workload_values) / max(0.001, min(workload_values))
    
    return std_dev, max_min_ratio, workloads

def split_cluster(cluster_df, num_clusters=2, max_attempts=3):
    """
    Split a cluster into smaller clusters using KMeans with multiple attempts to get better splits.
    Returns a list of sub-clusters.
    """
    # If the cluster is too small, don't split
    if len(cluster_df) < 10:
        return [cluster_df]
    
    # Extract coordinates for clustering
    coords = cluster_df[['Latitude', 'Longitude']].values
    
    best_split = None
    best_variance_ratio = float('inf')
    
    # Try multiple times with different random seeds
    for attempt in range(max_attempts):
        # Use KMeans with k-means++ initialization
        random_seed = 42 + attempt
        kmeans = KMeans(n_clusters=num_clusters, random_state=random_seed, n_init=3, init='k-means++')
        labels = kmeans.fit_predict(coords)
        
        # Create sub-clusters based on labels
        sub_clusters = []
        for i in range(num_clusters):
            cluster_mask = labels == i
            if sum(cluster_mask) > 0:  # Ensure we don't create empty clusters
                sub_clusters.append(cluster_df.iloc[cluster_mask])
        
        # Evaluate split quality by looking at size variance
        if len(sub_clusters) > 1:
            sizes = [len(sc) for sc in sub_clusters]
            mean_size = np.mean(sizes)
            variance_ratio = np.std(sizes) / mean_size if mean_size > 0 else float('inf')
            
            # Calculate spatial compactness of each sub-cluster
            compactness_scores = []
            for sc in sub_clusters:
                sc_coords = sc[['Latitude', 'Longitude']].values
                if len(sc_coords) > 1:
                    centroid = np.mean(sc_coords, axis=0)
                    distances = [haversine_distance(centroid[0], centroid[1], lat, lon) for lat, lon in sc_coords]
                    compactness = np.mean(distances)
                    compactness_scores.append(compactness)
            
            # Combine size variance and compactness into a single score
            if compactness_scores:
                avg_compactness = np.mean(compactness_scores)
                combined_score = variance_ratio * (1 + avg_compactness)
                
                if combined_score < best_variance_ratio:
                    best_variance_ratio = combined_score
                    best_split = sub_clusters
    
    return best_split if best_split else [cluster_df]

def find_cluster_to_split(clusters_dict, rep_clusters, workloads, already_split=None):
    """
    Find the best cluster to split based on workload imbalance and cluster characteristics.
    Considers history of previously split clusters to allow multiple splits of the same cluster.
    """
    if already_split is None:
        already_split = {}
    
    # Find rep with highest workload
    max_workload_rep = max(workloads, key=workloads.get)
    
    if not rep_clusters[max_workload_rep]:
        return None, None
    
    # Calculate threshold for splitting based on average cluster size
    avg_cluster_size = np.mean([len(cluster) for cluster in clusters_dict.values()])
    min_split_size = max(10, int(avg_cluster_size * 0.5))
    
    # Find best cluster to split based on size, spatial dispersion, and split history
    candidate_clusters = []
    
    for cluster_id in rep_clusters[max_workload_rep]:
        if cluster_id in clusters_dict:
            cluster_df = clusters_dict[cluster_id]
            cluster_size = len(cluster_df)
            
            # Count how many times this cluster has been split before
            split_count = already_split.get(cluster_id, 0)
            
            if cluster_size >= min_split_size:
                # Calculate spatial dispersion
                if cluster_size > 1:
                    coords = cluster_df[['Latitude', 'Longitude']].values
                    centroid = np.mean(coords, axis=0)
                    distances = np.array([haversine_distance(centroid[0], centroid[1], lat, lon) 
                                        for lat, lon in coords])
                    dispersion = np.std(distances)
                    
                    # Score based on size, dispersion, and split history
                    # Penalty for clusters that have been split many times
                    score = cluster_size * dispersion / (1 + split_count * 0.5)
                    
                    candidate_clusters.append((cluster_id, score, cluster_size, dispersion, split_count))
    
    # Sort candidates by score (highest first)
    candidate_clusters.sort(key=lambda x: -x[1])
    
    # Take the best candidate
    if candidate_clusters:
        best_cluster_id = candidate_clusters[0][0]
        return max_workload_rep, best_cluster_id
    
    return None, None

def generate_distinct_colors(n):
    """Generate visually distinct colors for clusters"""
    if n <= 10:
        # Use ColorBrewer qualitative color schemes for smaller numbers
        color_schemes = [
            ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],  # Default
            ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', 
             '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5']   # Set1 + Set3
        ]
        return color_schemes[0][:n]
    
    # For many colors, use HSV space with maximally distant hues
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7 + (i % 3) * 0.1  # Vary saturation slightly
        value = 0.7 + ((i + 1) % 3) * 0.1  # Vary value slightly
        rgb = matplotlib.colors.hsv_to_rgb([hue, saturation, value])
        hex_color = matplotlib.colors.rgb2hex(rgb)
        colors.append(hex_color)
    
    return colors

def assign_clusters_to_reps(clusters_dict, num_reps):
    """Assign clusters to representatives to balance workload"""
    if num_reps <= 0:
        return {}
    
    # Calculate workload for each cluster
    cluster_workloads = {cid: calculate_cluster_distance_mst(cluster) for cid, cluster in clusters_dict.items()}
    
    # Initialize empty assignments for each rep
    rep_workloads = {i: 0 for i in range(1, num_reps + 1)}
    rep_clusters = {i: [] for i in range(1, num_reps + 1)}
    
    # Sort clusters by workload (descending)
    sorted_clusters = sorted(cluster_workloads.items(), key=lambda x: -x[1])
    
    # Assign clusters using a greedy approach (assign to rep with lowest workload)
    for cluster_id, workload in sorted_clusters:
        min_workload_rep = min(rep_workloads, key=rep_workloads.get)
        rep_clusters[min_workload_rep].append(cluster_id)
        rep_workloads[min_workload_rep] += workload
    
    return rep_clusters

def main(data,reps):
    df=data
    df=df.rename(columns={'lattitude':'Latitude','longitude':'Longitude'})
    print(f"Loading data from ...")
    df = validate()
    print(f"Loaded {len(df)} points")
   
    print(f"Points belong to {df['Commune'].nunique()} communes")
    
    # Create a dictionary of clusters (starting with commune-based clusters)
    clusters_dict = {}
    for i, (commune_name, commune_df) in enumerate(df.groupby('Commune')):
        clusters_dict[i] = commune_df.copy()
    
    print(f"Created {len(clusters_dict)} initial clusters")
    
    # Number of representatives
    num_reps = reps
    
    # Initialize the assignment with optimized algorithm
    print("\nOptimizing initial cluster assignments...")
    rep_clusters = optimize_cluster_assignments(clusters_dict, num_reps)
    std_dev, max_min_ratio, workloads = evaluate_workload_balance(clusters_dict, rep_clusters)
    
    print("\nInitial workload balance (after optimization):")
    print(f"  Standard Deviation: {std_dev:.2f} km")
    print(f"  Max/Min Ratio: {max_min_ratio:.2f}")
    
    # Parameters for iterative improvement
    max_iterations = 30
    target_std_dev = 0.1  # Target standard deviation in km
    target_max_min_ratio = 1.2  # Target max/min ratio
    next_cluster_id = len(clusters_dict)  # For assigning IDs to new clusters
    
    # Track clusters that have already been split and how many times
    already_split = {}
    
    # Iteratively improve workload balance by splitting overloaded clusters
    iteration = 0
    while (iteration < max_iterations and 
           (std_dev > target_std_dev or max_min_ratio > target_max_min_ratio)):
        
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")
        
        # Find cluster to split, considering history of splits
        rep_to_split, cluster_to_split = find_cluster_to_split(clusters_dict, rep_clusters, workloads, already_split)
        
        if rep_to_split is None or cluster_to_split is None:
            print("No suitable cluster found for splitting")
            break
        
        # Update split history
        already_split[cluster_to_split] = already_split.get(cluster_to_split, 0) + 1
        split_count = already_split[cluster_to_split]
        
        # Split the cluster with potentially more sub-clusters for clusters that have been split before
        if cluster_to_split in clusters_dict:
            # Determine number of sub-clusters based on split history and size
            cluster_size = len(clusters_dict[cluster_to_split])
            base_num_clusters = 2
            
            # For larger clusters or clusters that have been split multiple times, create more sub-clusters
            if split_count > 1 or cluster_size > 50:
                base_num_clusters = min(3 + split_count, 5)  # Limit to max 5 sub-clusters
            
            print(f"Splitting cluster {cluster_to_split} from Rep {rep_to_split} into {base_num_clusters} sub-clusters (split #{split_count})")
            
            # Try splitting with multiple attempts
            sub_clusters = split_cluster(clusters_dict[cluster_to_split], num_clusters=base_num_clusters, max_attempts=3)
            
            if len(sub_clusters) <= 1:
                print("Cluster couldn't be split further")
                # Mark as not splittable to avoid trying again
                already_split[cluster_to_split] = -1
                continue
                
            # Remove original cluster from the dictionary
            clusters_dict.pop(cluster_to_split)
            
            # Add new sub-clusters to dictionary
            new_cluster_ids = []
            for sub_cluster in sub_clusters:
                if len(sub_cluster) > 0:  # Avoid empty clusters
                    clusters_dict[next_cluster_id] = sub_cluster
                    new_cluster_ids.append(next_cluster_id)
                    next_cluster_id += 1
            
            print(f"  Created {len(new_cluster_ids)} new clusters with IDs: {new_cluster_ids}")
            
            # Remove the original cluster ID from rep assignments
            for rep_id in rep_clusters:
                if cluster_to_split in rep_clusters[rep_id]:
                    rep_clusters[rep_id].remove(cluster_to_split)
            
            # Optimize assignments of all clusters
            print("  Re-optimizing cluster assignments...")
            rep_clusters = optimize_cluster_assignments(clusters_dict, num_reps)
                
            # Re-evaluate workload balance
            prev_std_dev, prev_max_min_ratio = std_dev, max_min_ratio
            std_dev, max_min_ratio, workloads = evaluate_workload_balance(clusters_dict, rep_clusters)
            
            improvement = (prev_std_dev - std_dev) / prev_std_dev * 100 if prev_std_dev > 0 else 0
            print(f"  New Standard Deviation: {std_dev:.2f} km (was {prev_std_dev:.2f}, improved by {improvement:.1f}%)")
            print(f"  New Max/Min Ratio: {max_min_ratio:.2f} (was {prev_max_min_ratio:.2f})")
            
            # Early stopping if improvement is very small
            if improvement < 1.0 and iteration > 5:
                print("  Minimal improvement, stopping iterations")
                break
        else:
            print(f"Cluster {cluster_to_split} not found in dictionary")
    
    # Display final results
    print("\nFinal Representative Workload Distribution:")
    workload_values = []
    for rep_id, distance in sorted(workloads.items()):
        cluster_ids = rep_clusters[rep_id]
        valid_cluster_ids = [cid for cid in cluster_ids if cid in clusters_dict]
        total_points = sum(len(clusters_dict[cid]) for cid in valid_cluster_ids)
        commune_counts = defaultdict(int)
        
        for cid in valid_cluster_ids:
            cluster_df = clusters_dict[cid]
            for commune in cluster_df['Commune'].unique():
                commune_counts[commune] += sum(cluster_df['Commune'] == commune)
        
        rep_communes = [f"{commune} ({count})" for commune, count in commune_counts.items()]
        
        print(f"Rep {rep_id}: {distance:.2f} km, {total_points} points, {len(valid_cluster_ids)} clusters")
        print(f"  Communes: {', '.join(rep_communes)}")
        workload_values.append(distance)
    
    # Print workload statistics
    if workload_values:
        print("\nWorkload Statistics (in kilometers):")
        print(f"  Minimum: {min(workload_values):.2f} km")
        print(f"  Maximum: {max(workload_values):.2f} km")
        print(f"  Average: {sum(workload_values)/len(workload_values):.2f} km")
        print(f"  Std Dev: {np.std(workload_values):.2f} km")
        if min(workload_values) > 0:
            print(f"  Max/Min Ratio: {max(workload_values)/min(workload_values):.2f}")
        else:
            print("  Max/Min Ratio: N/A (minimum workload is zero)")
    
    # Create Geojson file


if __name__ == "__main__":
    main()