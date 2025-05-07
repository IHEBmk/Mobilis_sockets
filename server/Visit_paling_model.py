# import math
# import pulp
# import numpy as np

# # Helper: Haversine formula to compute distance between two lat/lon points
# def haversine(lon1, lat1, lon2, lat2):
#     R = 6371  # Earth radius in km
#     lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
#     dlon, dlat = lon2 - lon1, lat2 - lat1
#     a = (math.sin(dlat / 2) ** 2 +
#          math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
#     return R * 2 * math.asin(math.sqrt(a))

# # Plan route for a single day (or time chunk)
# def plan_route(points, speed_kmph, time_limit_minutes, visited_points, visit_cost_minutes=15):
#     N = len(points)

#     # Create subset of unvisited points, ensuring start point is always included
#     unvisited_points = [point for point in points if point['name'] not in visited_points and point['name']!='Start']
#     unvisited_points.insert(0, points[0])  # Always start at the same location
#     N_unvisited = len(unvisited_points)

#     # Compute travel time matrix (in minutes)
#     travel_time = np.zeros((N_unvisited, N_unvisited))
#     for i in range(N_unvisited):
#         for j in range(N_unvisited):
#             if i != j:
#                 dist = haversine(unvisited_points[i]['longitude'], unvisited_points[i]['latitude'],
#                                  unvisited_points[j]['longitude'], unvisited_points[j]['latitude'])
#                 time = (dist / speed_kmph) * 60
#                 travel_time[i][j] = time

#     # Define integer programming model
#     model = pulp.LpProblem("Orienteering", pulp.LpMaximize)

#     x = pulp.LpVariable.dicts("x", ((i, j) for i in range(N_unvisited) for j in range(N_unvisited) if i != j), cat="Binary")
#     u = pulp.LpVariable.dicts("u", (i for i in range(N_unvisited)), lowBound=0, upBound=N_unvisited - 1, cat="Continuous")

#     # Objective: maximize number of visits (excluding return to start)
#     model += pulp.lpSum(x[i, j] for i in range(N_unvisited) for j in range(N_unvisited) if i != j and j != 0)

#     # Constraints
#     model += pulp.lpSum(x[0, j] for j in range(1, N_unvisited)) == 1
#     model += pulp.lpSum(x[j, 0] for j in range(1, N_unvisited)) == 0

#     for i in range(1, N_unvisited):
#         model += pulp.lpSum(x[j, i] for j in range(N_unvisited) if j != i) <= 1
#         model += pulp.lpSum(x[i, j] for j in range(N_unvisited) if j != i) <= 1

#     model += pulp.lpSum(x[i, j] * (travel_time[i][j] + visit_cost_minutes)
#                         for i in range(N_unvisited) for j in range(N_unvisited) if i != j) <= time_limit_minutes

#     # Subtour elimination (MTZ)
#     for i in range(1, N_unvisited):
#         for j in range(1, N_unvisited):
#             if i != j:
#                 model += u[i] - u[j] + (N_unvisited - 1) * x[i, j] <= N_unvisited - 2

#     model.solve()

#     # Extract solution
#     solution_edges = [(i, j) for i in range(N_unvisited) for j in range(N_unvisited)
#                       if i != j and pulp.value(x[i, j]) == 1]

#     # Build ordered route
#     solution_order = [unvisited_points[0]['name']]
#     current = 0
#     while True:
#         next_nodes = [j for i, j in solution_edges if i == current]
#         if not next_nodes:
#             break
#         current = next_nodes[0]
#         solution_order.append(unvisited_points[current]['name'])

#     estimated_time = sum(travel_time[i][j] + visit_cost_minutes for i, j in solution_edges)

#     return solution_order, solution_edges, unvisited_points, estimated_time

# # Multi-day planning
# def plan_multiple_days(points, total_minutes, daily_limit_minutes, speed_kmph):
#     visited_points = []
#     all_routes = []
#     all_edges = []
#     all_estimates = []

#     num_days = math.ceil(total_minutes / daily_limit_minutes)

#     for day in range(num_days):
#         route, edges, unvisited_points, estimated_time = plan_route(
#             points, speed_kmph=speed_kmph, time_limit_minutes=daily_limit_minutes, visited_points=visited_points
#         )

#         if len(route) <= 1:  # No progress
#             break

#         all_routes.append(route)
#         all_edges.append(edges)
#         all_estimates.append(estimated_time)
#         visited_points.extend(route[1:])  # Skip start point

#         if len(visited_points) >= len(points) - 1:
#             break

#     return all_routes, all_edges, all_estimates





import math
import pulp
import numpy as np

# Helper: Haversine formula to compute distance between two lat/lon points
def haversine(lon1, lat1, lon2, lat2):
    R = 6371  # Earth radius in km
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))

# Plan route for a single day (or time chunk)
def plan_route(points, speed_kmph, time_limit_minutes, visited_points, visit_cost_minutes=15):
    N = len(points)

    # Create subset of unvisited points, ensuring start point is always included
    unvisited_points = [point for point in points if point['name'] not in visited_points and point['name'] != 'Start']
    unvisited_points.insert(0, points[0])  # Always start at the same location
    N_unvisited = len(unvisited_points)

    # Compute travel time matrix (in minutes)
    travel_time = np.zeros((N_unvisited, N_unvisited))
    for i in range(N_unvisited):
        for j in range(N_unvisited):
            if i != j:
                dist = haversine(unvisited_points[i]['longitude'], unvisited_points[i]['latitude'],
                                 unvisited_points[j]['longitude'], unvisited_points[j]['latitude'])
                time = (dist / speed_kmph) * 60
                travel_time[i][j] = time

    # Define integer programming model
    model = pulp.LpProblem("Orienteering", pulp.LpMaximize)

    x = pulp.LpVariable.dicts("x", ((i, j) for i in range(N_unvisited) for j in range(N_unvisited) if i != j), cat="Binary")
    u = pulp.LpVariable.dicts("u", (i for i in range(N_unvisited)), lowBound=0, upBound=N_unvisited - 1, cat="Continuous")

    # Objective: maximize number of visits (excluding return to start)
    model += pulp.lpSum(x[i, j] for i in range(N_unvisited) for j in range(N_unvisited) if i != j and j != 0)

    # Constraints
    model += pulp.lpSum(x[0, j] for j in range(1, N_unvisited)) == 1
    model += pulp.lpSum(x[j, 0] for j in range(1, N_unvisited)) == 0

    for i in range(1, N_unvisited):
        model += pulp.lpSum(x[j, i] for j in range(N_unvisited) if j != i) <= 1
        model += pulp.lpSum(x[i, j] for j in range(N_unvisited) if j != i) <= 1

    model += pulp.lpSum(x[i, j] * (travel_time[i][j] + visit_cost_minutes)
                        for i in range(N_unvisited) for j in range(N_unvisited) if i != j) <= time_limit_minutes

    # Subtour elimination (MTZ)
    for i in range(1, N_unvisited):
        for j in range(1, N_unvisited):
            if i != j:
                model += u[i] - u[j] + (N_unvisited - 1) * x[i, j] <= N_unvisited - 2

    model.solve()

    # Extract solution
    solution_edges = [(i, j) for i in range(N_unvisited) for j in range(N_unvisited)
                      if i != j and pulp.value(x[i, j]) == 1]

    # Build ordered route
    solution_order = [unvisited_points[0]['name']]
    current = 0
    while True:
        next_nodes = [j for i, j in solution_edges if i == current]
        if not next_nodes:
            break
        current = next_nodes[0]
        solution_order.append(unvisited_points[current]['name'])

    estimated_time = sum(travel_time[i][j] + visit_cost_minutes for i, j in solution_edges)

    return solution_order, solution_edges, unvisited_points, estimated_time

# Multi-day planning until all points are visited
def plan_multiple_days(points, daily_limit_minutes, speed_kmph):
    visited_points = []
    all_routes = []
    all_edges = []
    all_estimates = []

    while len(visited_points) < len(points) - 1:  # Exclude start point
        route, edges, unvisited_points, estimated_time = plan_route(
            points, speed_kmph=speed_kmph, time_limit_minutes=daily_limit_minutes, visited_points=visited_points
        )

        if len(route) <= 1:  # No progress
            break

        all_routes.append(route)
        all_edges.append(edges)
        all_estimates.append(estimated_time)
        visited_points.extend(route[1:])  # Skip start point

    return all_routes, all_edges, all_estimates



