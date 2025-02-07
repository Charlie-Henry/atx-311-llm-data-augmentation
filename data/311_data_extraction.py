# Extract 10 of each 311 service type and write it to a CSV file
import requests
import os
import time
from config import REQUEST_TYPES
import pandas as pd

API_KEY = os.getenv("OPEN_311_API_KEY")
BASE_URL = "https://311.austintexas.gov/open311/v2"


def download_requests(service_id):
    params = {"service_code": service_id, "page_size": 10, "extensions": True}
    headers = {"Authorization": f"Bearer {API_KEY}"}
    response = requests.get(f"{BASE_URL}/requests.json", headers=headers, params=params)
    return response.json()


def main():
    data = []
    for service_type in REQUEST_TYPES.keys():
        data += download_requests(service_type)
        # By default, all users of the Open311 API are limited to 10 requests per minute.
        time.sleep(7)
    pd.DataFrame(data).to_csv("311_data_sample.csv", index=False)


if __name__ == "__main__":
    main()
