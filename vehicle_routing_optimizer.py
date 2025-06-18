import pandas as pd
import numpy as np
import requests
import folium
import streamlit as st
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import tempfile
import polyline

# Constants
BUS_CAPACITY = 63
MAX_DISTANCE_KM = 150
MAX_DURATION_MIN = 180
LATEST_DROP_time = "08:45"
API_KEY = st.secrets["API_KEY"]

def load_data(file):
    df = pd.read_excel(file)
    df["Latest Drop Time (HH:MM)"] = "08:45"
    return df

def build_location_index(df):
    df_pick = df[["Pickup Latitude", "Pickup Longitude", "Pickup Location Name"]].rename(columns={"Pickup Latitude": "lat", "Pickup Longitude": "lng", "Pickup Location Name": "name"})
    df_drop = df[["Drop Latitude", "Drop Longitude", "Drop Location Name"]].rename(columns={"Drop Latitude": "lat", "Drop Longitude": "lng", "Drop Location Name": "name"})
    locations = pd.concat([df_pick, df_drop]).drop_duplicates().reset_index(drop=True)
    return locations

def fetch_distance_matrix(locations):
    n = len(locations)
    dist_matrix = np.zeros((n, n))
    dur_matrix = np.zeros((n, n))
    for i in range(n):
        origins = f"{locations.iloc[i]['lat']},{locations.iloc[i]['lng']}"
        destinations = "|".join([f"{row.lat},{row.lng}" for _, row in locations.iterrows()])
        url = f"https://maps.googleapis.com/maps/api/distancematrix/json?origins={origins}&destinations={destinations}&key={API_KEY}"
        response = requests.get(url).json()
        for j, element in enumerate(response['rows'][0]['elements']):
            dist_matrix[i][j] = element['distance']['value'] / 1000
            dur_matrix[i][j] = element['duration']['value'] / 60
    return dist_matrix, dur_matrix

def solve_routing(dist_matrix, dur_matrix, demands, num_vehicles, depot):
    manager = pywrapcp.RoutingIndexManager(len(dist_matrix), num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(dist_matrix[from_node][to_node] * 1000)

    def duration_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(dur_matrix[from_node][to_node])

    routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(distance_callback))
    demand_callback_index = routing.RegisterUnaryTransitCallback(lambda idx: demands[manager.IndexToNode(idx)])
    routing.AddDimensionWithVehicleCapacity(demand_callback_index, 0, [BUS_CAPACITY] * num_vehicles, True, 'Capacity')
    duration_callback_index = routing.RegisterTransitCallback(duration_callback)
    routing.AddDimension(duration_callback_index, 0, MAX_DURATION_MIN, True, 'Duration')

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.FromSeconds(60)

    solution = routing.SolveWithParameters(search_parameters)
    if not solution:
        return []

    routes = []
    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        route = []
        load = 0
        distance = 0
        duration = 0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(node)
            load += demands[node]
            next_index = solution.Value(routing.NextVar(index))
            if not routing.IsEnd(next_index):
                distance += dist_matrix[node][manager.IndexToNode(next_index)]
                duration += dur_matrix[node][manager.IndexToNode(next_index)]
            index = next_index
        if len(route) > 1:
            routes.append({
                "vehicle_id": vehicle_id,
                "nodes": route,
                "load": load,
                "distance_km": round(distance, 2),
                "duration_min": round(duration, 2),
                "utilization": round((load / BUS_CAPACITY) * 100, 1)
            })
    return routes

def get_snapped_polyline(coords):
    origin = f"{coords[0][0]},{coords[0][1]}"
    destination = f"{coords[-1][0]},{coords[-1][1]}"
    waypoints = "|".join([f"{lat},{lng}" for lat, lng in coords[1:-1]])
    url = f"https://maps.googleapis.com/maps/api/directions/json?origin={origin}&destination={destination}&waypoints={waypoints}&key={API_KEY}"
    response = requests.get(url).json()
    if response['status'] == 'OK':
        points = response['routes'][0]['overview_polyline']['points']
        return polyline.decode(points)
    return coords

def visualize_routes(routes, locations):
    route_map = folium.Map(location=[locations.iloc[0].lat, locations.iloc[0].lng], zoom_start=10)
    colors = ['blue', 'green', 'purple', 'orange', 'red', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple']
    for i, route in enumerate(routes):
        color = colors[i % len(colors)]
        coords = [(locations.iloc[n].lat, locations.iloc[n].lng) for n in route['nodes']]
        snapped = get_snapped_polyline(coords)
        folium.PolyLine(snapped, color=color, weight=5, opacity=0.7).add_to(route_map)
        folium.Marker(snapped[0], popup="Start").add_to(route_map)
        folium.Marker(snapped[-1], popup="End").add_to(route_map)
    return route_map

st.set_page_config(page_title="Vehicle Routing Optimizer", layout="wide")
st.title("🚌 Vehicle Routing Optimizer")

uploaded_file = st.file_uploader("Upload Routing Excel", type=["xlsx"])
if uploaded_file:
    with st.spinner("Processing routes..."):
        df = load_data(uploaded_file)
        locations = build_location_index(df)
        dist_matrix, dur_matrix = fetch_distance_matrix(locations)
        demands = [0] * len(locations)
        for _, row in df.iterrows():
            for i, loc in locations.iterrows():
                if loc.lat == row["Pickup Latitude"] and loc.lng == row["Pickup Longitude"]:
                    demands[i] += row["Employee Count"]
        routes = solve_routing(dist_matrix, dur_matrix, demands, 100, 0)
        if routes:
            st.success(f"Found {len(routes)} routes!")
            for i, route in enumerate(routes, start=1):
                stop_names = [locations.iloc[n].name for n in route['nodes']]
                st.markdown(f"### 🗺 Route {i}")
                st.write("**Stops:** " + " → ".join(str(name) for name in stop_names))
                st.write(f"**Distance:** {route['distance_km']} km")
                st.write(f"**Duration:** {route['duration_min']} min")
                st.write(f"**Load:** {route['load']} / {BUS_CAPACITY} → **Utilization:** {route['utilization']}%")
            st.components.v1.html(visualize_routes(routes, locations)._repr_html_(), height=600)
        else:
            st.error("No routes found. Please check input data or constraints.")
