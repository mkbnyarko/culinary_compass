import urllib.request
import urllib.error
import json

BASE_URL = "http://127.0.0.1:8000"

def test_404():
    print("\n--- Testing 404 (Not Found) ---")
    try:
        urllib.request.urlopen(f"{BASE_URL}/non-existent-endpoint")
        print("FAILURE: Expected 404, got 200 OK")
    except urllib.error.HTTPError as e:
        print(f"Status Code: {e.code}")
        print(f"Response: {e.read().decode()}")
        if e.code == 404:
            print("SUCCESS: Received 404")
        else:
            print(f"FAILURE: Expected 404, got {e.code}")
    except Exception as e:
        print(f"FAILURE: Exception occurred: {e}")

def test_422():
    print("\n--- Testing 422 (Validation Error) ---")
    req = urllib.request.Request(
        f"{BASE_URL}/recommend",
        data=json.dumps({"wrong_field": "oops"}).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    try:
        urllib.request.urlopen(req)
        print("FAILURE: Expected 422, got 200 OK")
    except urllib.error.HTTPError as e:
        print(f"Status Code: {e.code}")
        print(f"Response: {e.read().decode()}")
        if e.code == 422:
             print("SUCCESS: Received 422")
        else:
            print(f"FAILURE: Expected 422, got {e.code}")
    except Exception as e:
        print(f"FAILURE: Exception occurred: {e}")

if __name__ == "__main__":
    test_404()
    test_422()
