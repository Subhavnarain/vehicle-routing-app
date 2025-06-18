
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
LATEST_DROP_TIME = "08:45"
API_KEY = st.secrets["API_KEY"]

def load_data(file):
    df = pd.read_excel(file)
    df["Latest Drop Time (HH:MM)"] = "08:45"
    return df

def build_location_index(df):
    locations = pd.concat([
        df[["Pickup Latitude", "Pickup Longitude"]].rename(columns={"Pickup Latitude": "lat", "Pickup Longitude": "lng"}),
        df[["Drop Latitude", "Drop Longitude"]].rename(columns={"Drop Latitude": "lat", "Drop Longitude": "lng"})
    ]).drop_duplicates().reset_index(drop=True)
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

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    demand_callback_index = routing.RegisterUnaryTransitCallback(lambda idx: demands[manager.IndexToNode(idx)])
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index, 0, [BUS_CAPACITY] * num_vehicles, True, 'Capacity'
    )

    duration_callback_index = routing.RegisterTransitCallback(duration_callback)
    routing.AddDimension(
        duration_callback_index, 0, MAX_DURATION_MIN, True, 'Duration'
    )

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
        route_load = 0
        route_distance = 0
        route_duration = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route.append(node_index)
            route_load += demands[node_index]
            next_index = solution.Value(routing.NextVar(index))
            if not routing.IsEnd(next_index):
                route_distance += dist_matrix[node_index][manager.IndexToNode(next_index)]
                route_duration += dur_matrix[node_index][manager.IndexToNode(next_index)]
            index = next_index
        if len(route) > 1:
            utilization = round((route_load / BUS_CAPACITY) * 100, 1)
            routes.append({
                "vehicle_id": vehicle_id,
                "nodes": route,
                "load": route_load,
                "distance_km": round(route_distance, 2),
                "duration_min": round(route_duration, 2),
                "utilization": utilization
            })
    return routes

def get_snapped_polyline(start, end):
    url = f"https://maps.googleapis.com/maps/api/directions/json?origin={start}&destination={end}&key={API_KEY}"
    response = requests.get(url).json()
    if response['status'] == 'OK':
        points = response['routes'][0]['overview_polyline']['points']
        return polyline.decode(points)
    return [tuple(map(float, start.split(','))), tuple(map(float, end.split(',')))]

def visualize_routes(routes, locations):
    route_map = folium.Map(location=[locations.iloc[0].lat, locations.iloc[0].lng], zoom_start=10)
    colors = ['blue', 'green', 'purple', 'orange', 'red', 'darkred', 'lightred',
              'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple']

    for i, route in enumerate(routes):
        color = colors[i % len(colors)]
        coords = []
        for j in range(len(route['nodes']) - 1):
            a = locations.iloc[route['nodes'][j]]
            b = locations.iloc[route['nodes'][j + 1]]
            snapped = get_snapped_polyline(f"{a.lat},{a.lng}", f"{b.lat},{b.lng}")
            coords += snapped
        folium.PolyLine(coords, color=color, weight=5, opacity=0.7).add_to(route_map)
        folium.Marker(coords[0], popup=f"Start").add_to(route_map)
        folium.Marker(coords[-1], popup=f"End").add_to(route_map)

    return route_map

# --- Streamlit App ---
st.set_page_config(page_title="Vehicle Routing Optimizer", layout="wide")
st.title("🚌 Vehicle Routing Optimizer")

uploaded_file = st.file_uploader("Upload Routing Excel", type=["xlsx"])

if uploaded_file:
    with st.spinner("Processing routes..."):
        df = load_data(uploaded_file)
        locations = build_location_index(df)
        dist_matrix, dur_matrix = fetch_distance_matrix(locations)
        demands = [0] * len(locations)  # placeholder logic
        routes = solve_routing(dist_matrix, dur_matrix, demands, 100, 0)

        if routes:
            st.success(f"Found {len(routes)} routes!")
            for i, route in enumerate(routes, start=1):
                st.markdown(f"### 🗺 Route {i}")
                stops = " → ".join([f"Stop {n}" for n in route['nodes']])
                st.write(f"**Stops:** {stops}")
                st.write(f"**Distance:** {route['distance_km']} km")
                st.write(f"**Duration:** {route['duration_min']} min")
                st.write(f"**Load:** {route['load']} / {BUS_CAPACITY} → **Utilization:** {route['utilization']}%")

            map_object = visualize_routes(routes, locations)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
                map_object.save(f.name)
                with open(f.name, "r") as file:
                    map_html = file.read()
                st.components.v1.html(map_html, height=600)
        else:
            st.error("No routes found. Please check input data or constraints.")
