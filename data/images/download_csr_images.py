import random

import requests
import os
import time
import pandas as pd
import re


SO_USER = os.getenv("SO_USER")
SO_PASSWORD = os.getenv("SO_PASSWORD")

DATASET_URL = "https://datahub.austintexas.gov/resource/bayp-2ymg.json"
OPEN_311_URL = "https://311.austintexas.gov/open311/v2"
OPEN_311_API_KEY = os.getenv("OPEN_311_API_KEY")


def get_image_list():
    data = requests.get(
        DATASET_URL + "?$limit=9999&$order=created_date%20DESC",
        auth=(SO_USER, SO_PASSWORD),
    )
    return data.json()


def download_descriptions(service_request_id):
    params = {"extensions": True}
    headers = {"Authorization": f"Bearer {OPEN_311_API_KEY}"}
    response = requests.get(
        f"{OPEN_311_URL}/requests/{service_request_id}.json",
        headers=headers,
        params=params,
    )
    return response.json()[0]


def download_image(url, save_path):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for request errors
        with open(save_path, "wb") as file:
            file.write(response.content)
        print(f"Image successfully saved to {save_path}")
    except requests.RequestException as e:
        print(f"Error downloading the image: {e}")


def main():
    # First, download the list of CSRs with images
    images = get_image_list()
    output = []
    # Shuffling order of requests so we get a random sample
    random.shuffle(images)
    for sr in images:
        data = download_descriptions(sr["service_request_sr_number"])
        # By default, all users of the Open311 API are limited to 10 requests per minute.
        time.sleep(7)
        if "description" in data:
            # some images are in a string list, we only care about the first one for now
            image_url = sr[
                "servicerequestsflexnotestransportationandpublicworks_flex_answer"
            ]
            image_url = re.search(r"\[(https?://[^\]]+)\]", image_url)
            image_url = image_url.group(1)

            download_image(
                image_url,
                save_path=f"csr_images/{sr['service_request_sr_number']}.jpg",
            )
            output.append(data)
            if len(output) >= 30:
                break

    pd.DataFrame(output).to_csv("311_with_images.csv", index=False)


if __name__ == "__main__":
    main()
