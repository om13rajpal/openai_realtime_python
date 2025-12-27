"""
Quick test script for API endpoints
Run with: python test_endpoints.py
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    print("\n=== Testing Health Endpoint ===")
    r = requests.get(f"{BASE_URL}/health")
    print(f"Status: {r.status_code}")
    print(f"Response: {r.json()}")
    return r.status_code == 200

def test_weather_query():
    print("\n=== Testing Weather Query ===")
    payload = {
        "message": "What's the weather in Miami?",
        "user_location": "New York"
    }
    r = requests.post(f"{BASE_URL}/maritime-chat", json=payload)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"Response Type: {data.get('response', {}).get('type')}")
        print(f"Has Report: {'report' in data.get('response', {})}")
        print(json.dumps(data, indent=2)[:500] + "...")
    return r.status_code == 200

def test_assistance_query():
    print("\n=== Testing Local Assistance Query ===")
    payload = {
        "message": "I need local maritime assistance in Miami",
        "user_location": "Miami"
    }
    r = requests.post(f"{BASE_URL}/maritime-chat", json=payload)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"Response Type: {data.get('response', {}).get('type')}")
        print(f"Has Assistance: {'local_assistance' in data.get('response', {})}")
    return r.status_code == 200

def test_route_query():
    print("\n=== Testing Route Planning Query ===")
    payload = {
        "message": "Plan a route from Miami to Dubai",
        "user_location": "Miami",
        "vessels": [{"make": "Catalina", "model": "315", "year": "2018"}]
    }
    r = requests.post(f"{BASE_URL}/maritime-chat", json=payload)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"Response Type: {data.get('response', {}).get('type')}")
        print(f"Has Trip Plan: {'trip_plan' in data.get('response', {})}")
    return r.status_code == 200

def test_normal_query():
    print("\n=== Testing Normal Query ===")
    payload = {
        "message": "Hello, how can you help me?",
    }
    r = requests.post(f"{BASE_URL}/maritime-chat", json=payload)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"Response Type: {data.get('response', {}).get('type')}")
        print(f"Message: {data.get('response', {}).get('message', '')[:200]}...")
    return r.status_code == 200

if __name__ == "__main__":
    print("=" * 50)
    print("API ENDPOINT TESTS")
    print("=" * 50)
    print("\nMake sure the backend is running at", BASE_URL)
    
    try:
        results = {
            "Health": test_health(),
            "Weather": test_weather_query(),
            "Assistance": test_assistance_query(),
            "Route": test_route_query(),
            "Normal": test_normal_query(),
        }
        
        print("\n" + "=" * 50)
        print("TEST RESULTS")
        print("=" * 50)
        for test, passed in results.items():
            status = "PASS" if passed else "FAIL"
            print(f"{test}: {status}")
    except requests.exceptions.ConnectionError:
        print("\nERROR: Cannot connect to backend. Make sure it's running!")
        print("Run: python -m uvicorn app.main:app --reload")
