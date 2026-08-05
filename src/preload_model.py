from llama_server import health_check

STARTUP_TIMEOUT = 10 * 60
PORT = 1234

# MODEL_NAME = "gemma-4-E2B"
MODEL_NAME = "Qwen3.5-4B"
LLAMA_API_KEY = "MY_API_KEY"
URL = f"http://localhost:{PORT}"

def start_server():
    import subprocess

    subprocess.Popen([
        "llama", "serve",
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--models-max", "1",
        # "--api-key", str(LLAMA_API_KEY),
        "--models-preset", "/home/sujyot/containers/llama-server/models.ini",
    ])

def load_model(url: str, api_key: str, model_name: str):
    import requests

    try:
        payload = {"model": model_name}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        print(f"Sending load request for model {model_name}")
        res = requests.post(f"{url}/models/load", json=payload, headers=headers)
        if not res.ok:
            raise RuntimeError(f"Failed to initiate model load: {res.status_code} - {res.text}")

        print(f"PreLoading '{model_name}'...")
        model_status(url, api_key, model_name)
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except RuntimeError as e:
        print(e)


def model_status(url: str, api_key: str, model_name: str):
    import time
    import requests

    deadline = time.time() + STARTUP_TIMEOUT
    while time.time() < deadline:
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }

            res = requests.get(f"{url}/models", headers=headers)

            if not res.ok:
                raise RuntimeError(f"Failed to get models list: {res.status_code} - {res.text}")

            status = get_model_status(res.json(), model_name)
            if status == None:
                raise RuntimeError(f"Model not found: {model_name}")

            if status == "loaded":
                print(f"Model loaded: {model_name}")
                break
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
        except RuntimeError as e:
            print(e)

        time.sleep(5)


def get_model_status(jsonRes, model_id):
    for model in jsonRes.get("data", []):
        if model["id"] == model_id:
            return model["status"]["value"]
    return None  # model not found


def main():
    start_server()
    health_check(URL, LLAMA_API_KEY)
    load_model(URL, LLAMA_API_KEY, MODEL_NAME)

if __name__ == "__main__":
    main()