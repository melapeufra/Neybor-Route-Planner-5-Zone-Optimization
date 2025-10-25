from typing import List, Dict, Tuple, Optional
from math import radians, sin, cos, sqrt, atan2
import random
import csv
import argparse

HOUSES = [
    {"name": "Duden 26", "address": "Rue Egide Walschaerts 26, Saint-Gilles 1060", "lat": 50.826828311543565, "lon": 4.33677586917812, "is_airbnb": True},
    {"name": "Louise 32", "address": "Rue Jourdan 32, Saint-Gilles 1060", "lat": 50.83428780731322, "lon": 4.354435782671721, "is_airbnb": True},
    {"name": "Duden 3", "address": "Avenue Reine Marie-Henriette 3, Forest 1190", "lat": 50.82345878988992, "lon": 4.33436130965701, "is_airbnb": True},
    {"name": "Congres 46", "address": "Rue de la Croix de Fer 46, Brussels 1000", "lat": 50.848541593328044, "lon": 4.365480126851019, "is_airbnb": True},
    {"name": "Congres 22", "address": "Rue de la Tribune 22, Brussels 1000", "lat": 50.84854488903282, "lon": 4.365122150137942, "is_airbnb": True},
    {"name": "Ambiorix 46", "address": "Rue des Deux Tours 46, 1210 Saint-Josse-ten-Noode", "lat": 50.85105495407479, "lon": 4.3782560115086575, "is_airbnb": False},
    {"name": "Artan 112", "address": "Rue Artan 112, 1030 Schaerbeek", "lat": 50.85364680806313, "lon": 4.383660167330543, "is_airbnb": False},
    {"name": "Botanique 31", "address": "Chau. de Haecht 31, 1210 Saint-Josse-ten-Noode", "lat": 50.85592825310391, "lon": 4.367923365481105, "is_airbnb": False},
    {"name": "Botanique 42", "address": "Rue du Méridien 42, 1210 Saint-Josse-ten-Noode", "lat": 50.854565830029024, "lon": 4.369158338494917, "is_airbnb": False},
    {"name": "Brugman 53", "address": "Av. Louis Lepoutre 53, 1050 Ixelles", "lat": 50.81912136605574, "lon": 4.357256511506246, "is_airbnb": False},
    {"name": "Chatelin 63", "address": "Rue de Florence 63, 1060 Saint-Gilles", "lat": 50.82881048643311, "lon": 4.35739712684958, "is_airbnb": False},
    {"name": "Colignon 31", "address": "Rue Herman 31, 1030 Schaerbeek", "lat": 50.86468744645891, "lon": 4.377417184523553, "is_airbnb": False},
    {"name": "Colignon 39", "address": "Rue Emmanuel Hiel 39, 1030 Schaerbeek", "lat": 50.86635253901819, "lon": 4.371428511509731, "is_airbnb": False},
    {"name": "Fernand 12", "address": "Rue Mercelis 12, 1050 Ixelles", "lat": 50.832579262484806, "lon": 4.3660458673289915, "is_airbnb": False},
    {"name": "Fernand 3", "address": "Rue du Viaduc 3, 1050 Ixelles", "lat": 50.832376723699724, "lon": 4.3676680961646195, "is_airbnb": False},
    {"name": "Flagey 16", "address": "Rue Maes 16, 1050 Ixelles", "lat": 50.831473175030396, "lon": 4.369378880821944, "is_airbnb": False},
    {"name": "Flagey 21", "address": "Rue du Serpentin 21, 1050 Ixelles", "lat": 50.82793441321475, "lon": 4.374789453835611, "is_airbnb": False},
    {"name": "Flagey 33", "address": "Rue Wéry 33, 1050 Ixelles", "lat": 50.83133348701857, "lon": 4.3755055384932175, "is_airbnb": False},
    {"name": "Louise 13", "address": "Rue d'Ecosse 13, 1060 Saint-Gilles", "lat": 50.83328226728295, "lon": 4.352356296164707, "is_airbnb": False},
    {"name": "Louise 65", "address": "Rue Mercelis 65, 1050 Ixelles", "lat": 50.83107722398169, "lon": 4.363488094314956, "is_airbnb": False},
    {"name": "Montgomery 17", "address": "Rue de la Duchesse 17, 1150 Woluwe-Saint-Pierre", "lat": 50.838650744291066, "lon": 4.406603980822448, "is_airbnb": False},
    {"name": "Parvis 4", "address": "Rue de la Filature 4, 1060 Saint-Gilles", "lat": 50.83203942132771, "lon": 4.345137138493248, "is_airbnb": False},
    {"name": "Leopold 1", "address": "Rue Vandenbroeck 1", "lat": 50.835676118178, "lon": 4.374911192465664, "is_airbnb": False},
    {"name": "Leopold 55", "address": "Rue du Chateau 55", "lat": 50.83131861580924, "lon": 4.380596035899892, "is_airbnb": False},
    {"name": "Neybor Office", "address": "Chaussée de Boondael 365, Ixelles 1050", "lat": 50.8177318, "lon": 4.3864221, "is_airbnb": False},
]

#distance & TSP

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def pair_distance_km(a: Dict, b: Dict) -> float:
    return haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])

def build_distance_matrix(items: List[Dict]) -> List[List[float]]:
    n = len(items)
    D = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            d = pair_distance_km(items[i], items[j])
            D[i][j] = D[j][i] = d
    return D

def nearest_neighbor(D: List[List[float]], start: int) -> List[int]:
    n = len(D)
    unv = set(range(n))
    route = [start]
    unv.remove(start)
    cur = start
    while unv:
        nxt = min(unv, key=lambda j: D[cur][j])
        route.append(nxt)
        unv.remove(nxt)
        cur = nxt
    return route

def route_length(D: List[List[float]], route: List[int], cycle: bool=False) -> float:
    L = sum(D[route[i]][route[i+1]] for i in range(len(route)-1))
    if cycle and route:
        L += D[route[-1]][route[0]]
    return L

def two_opt(route: List[int], D: List[List[float]], cycle: bool=False) -> List[int]:
    improved = True
    best = route[:]
    best_len = route_length(D, best, cycle=cycle)
    n = len(best)
    if n < 4:
        return best
    while improved:
        improved = False
        for i in range(1, n-2):
            for k in range(i+1, n-1):
                new_route = best[:i] + best[i:k+1][::-1] + best[k+1:]
                new_len = route_length(D, new_route, cycle=cycle)
                if new_len + 1e-9 < best_len:
                    best, best_len = new_route, new_len
                    improved = True
    return best

def kmeans_plus_plus_init(points: List[Tuple[float, float]], k: int, seed: int=42) -> List[Tuple[float, float]]:
    rnd = random.Random(seed)
    centroids = [points[rnd.randrange(len(points))]]
    while len(centroids) < k:
        d2 = []
        for (x, y) in points:
            m = min((x-cx)**2 + (y-cy)**2 for (cx, cy) in centroids)
            d2.append(m)
        total = sum(d2)
        r = rnd.random() * total
        cum = 0.0
        for idx, w in enumerate(d2):
            cum += w
            if cum >= r:
                centroids.append(points[idx])
                break
    return centroids

def kmeans_5(houses: List[Dict], max_iter: int=200, seed: int=42) -> List[List[int]]:
    k = 5
    pts = [(h["lat"], h["lon"]) for h in houses]
    cents = kmeans_plus_plus_init(pts, k, seed=seed)

    def d2(a, b): return (a[0]-b[0])**2 + (a[1]-b[1])**2

    for _ in range(max_iter):
        clusters = [[] for _ in range(k)]
        for i, p in enumerate(pts):
            j = min(range(k), key=lambda c: d2(p, cents[c]))
            clusters[j].append(i)
        new_cents = []
        for idxs in clusters:
            if idxs:
                lat = sum(pts[i][0] for i in idxs)/len(idxs)
                lon = sum(pts[i][1] for i in idxs)/len(idxs)
                new_cents.append((lat, lon))
            else:
                new_cents.append(pts[random.randrange(len(pts))])
        if new_cents == cents:
            break
        cents = new_cents
    return clusters


def filter_houses(houses: List[Dict], exclude_airbnb: bool) -> List[Dict]:
    return [h for h in houses if not (exclude_airbnb and h.get("is_airbnb", False))]

def plan_5zones(houses: List[Dict]) -> Dict[str, Dict]:
    """Returns dict: {Zone_n: {houses:[...], order:[...], total_km:float}}"""
    from collections import defaultdict
    clusters = kmeans_5(houses)
    results = {}

    for zi, idxs in enumerate(clusters):
        zone_key = f"Zone_{zi+1}"
        zone_houses = [houses[i] for i in idxs]
        if not zone_houses:
            results[zone_key] = {"houses": [], "order": [], "total_km": 0.0}
            continue

        cx = sum(h["lat"] for h in zone_houses)/len(zone_houses)
        cy = sum(h["lon"] for h in zone_houses)/len(zone_houses)
        start_idx = min(range(len(zone_houses)), key=lambda t: (zone_houses[t]["lat"]-cx)**2 + (zone_houses[t]["lon"]-cy)**2)

        D = build_distance_matrix(zone_houses)
        route = nearest_neighbor(D, start=start_idx)
        route = two_opt(route, D, cycle=True) 
        total = route_length(D, route, cycle=True)

        ordered = [zone_houses[i] for i in route]
        results[zone_key] = {"houses": ordered, "order": route, "total_km": total}
    return results

def export_csv(results: Dict[str, Dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["zone", "order", "house_name", "address", "lat", "lon"])
        for zone, info in results.items():
            for i, h in enumerate(info["houses"], start=1):
                w.writerow([zone, i, h["name"], h["address"], h["lat"], h["lon"]])

def print_routes(results: Dict[str, Dict]) -> None:
    for zone, info in sorted(results.items(), key=lambda kv: kv[0]):
        H = info["houses"]
        print(f"\n=== {zone} ===")
        print(f"Stops (incl. cycle back to start): {len(H)} | Route (open): {len(H)} steps | Cycle length ~ {info['total_km']:.2f} km")
        for i, h in enumerate(H, start=1):
            print(f"{i:2d}. {h['name']}  [{h['address']}]")

def save_map(results: Dict[str, Dict], path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is required for --save-map")
        return
    plt.figure(figsize=(8,8))
    zones = list(sorted(results.keys()))
    colors = plt.cm.get_cmap("tab10", len(zones))
    for i, z in enumerate(zones):
        H = results[z]["houses"]
        if not H:
            continue
        xs = [h["lon"] for h in H] + [H[0]["lon"]]
        ys = [h["lat"] for h in H] + [H[0]["lat"]]
        plt.plot(xs, ys, '-', alpha=0.7, label=z)  
        plt.scatter([h["lon"] for h in H], [h["lat"] for h in H])
        for h in H:
            plt.text(h["lon"], h["lat"], h["name"], fontsize=7)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Five Zones with Cyclic Shortest Routes")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    print(f"Map saved to: {path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude-airbnb", action="store_true", help="Exclude AirBnB houses")
    ap.add_argument("--export-csv", type=str, default=None, help="Path to export CSV")
    ap.add_argument("--save-map", type=str, default=None, help="Path to save PNG map")
    args = ap.parse_args()

    houses = filter_houses(HOUSES, args.exclude_airbnb)
    results = plan_5zones(houses)
    print_routes(results)

    if args.export_csv:
        export_csv(results, args.export_csv)
        print(f"\nCSV exported to: {args.export_csv}")
    if args.save_map:
        save_map(results, args.save_map)

if __name__ == "__main__":
    main()
