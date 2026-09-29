import os
import requests

def _nominatim_geocode(place: str) -> dict:
    url = "https://nominatim.openstreetmap.org/search"
    headers = {"User-Agent": "FairyAI/1.0"}
    params = {"q": place, "format": "json", "limit": 1}
    resp = requests.get(url, params=params, headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return {"ok": False, "error": f"Place not found: {place}"}
    return {"ok": True, "name": data[0]["display_name"], "lat": float(data[0]["lat"]), "lon": float(data[0]["lon"])}

def _osrm_directions(lat1, lon1, lat2, lon2) -> dict:
    url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}"
    params = {"overview": "false", "steps": "true"}
    resp = requests.get(url, params=params, timeout=15)
    data = resp.json()
    if data["code"] != "Ok":
        return {"ok": False, "error": "Could not calculate route."}
    route = data["routes"][0]
    return {"ok": True, "distance_km": round(route["distance"] / 1000, 1), "duration_min": round(route["duration"] / 60, 1), "steps": [step["name"] for step in route["legs"][0]["steps"] if step["name"]]}

def get_location(place: str) -> str:
    result = _nominatim_geocode(place)
    if not result["ok"]:
        return f"Location error: {result['error']}"
    return f"[Location] {result['name']}\n   Latitude: {result['lat']}\n   Longitude: {result['lon']}"

def get_directions(origin: str, destination: str) -> str:
    o = _nominatim_geocode(origin)
    d = _nominatim_geocode(destination)
    if not o["ok"]:
        return f"Could not find origin: {o.get('error', origin)}"
    if not d["ok"]:
        return f"Could not find destination: {d.get('error', destination)}"
    route = _osrm_directions(o["lat"], o["lon"], d["lat"], d["lon"])
    if not route["ok"]:
        return f"Route error: {route['error']}"
    steps_text = "\n   ".join(f"-> {s}" for s in route["steps"][:8])
    return f"[Route] {origin} -> {destination}\n   Distance: {route['distance_km']} km\n   Duration: {route['duration_min']} min\n\n   {steps_text}"

def search_nearby(location: str, query: str) -> str:
    loc = _nominatim_geocode(location)
    if not loc["ok"]:
        return f"Could not find location: {loc.get('error', location)}"
    url = "https://nominatim.openstreetmap.org/search"
    headers = {"User-Agent": "FairyAI/1.0"}
    params = {"q": f"{query} near {location}", "format": "json", "limit": 5}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        data = resp.json()
        if not data:
            return f"No results for '{query}' near {location}."
        lines = [f"[Search] {query} near {location}:"]
        for i, r in enumerate(data, 1):
            lines.append(f"   {i}. {r['display_name']}")
        return "\n".join(lines)
    except Exception as e:
        return f"Search error: {e}"
