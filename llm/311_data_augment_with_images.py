import dspy
import pandas as pd

from config import REQUEST_TYPES


def load_llm():
    lm = dspy.LM("ollama/llava:7b")
    dspy.configure(lm=lm)
    return lm


def load_test_data():
    df = pd.read_csv("../data/images/311_with_images.csv")
    # require for now, that a description is provided
    df = df.dropna(subset=["description"])
    return df.to_dict(orient="records")


def process_request_types():
    output = []
    for service_id in REQUEST_TYPES:
        name = REQUEST_TYPES[service_id]["name"]
        desc = REQUEST_TYPES[service_id]["description"]
        output.append(f"Name: {name}, Description {desc};")
    return output


class Augment311Request(dspy.Signature):
    """
    You are an assistant with the City of Austin's Transportation Public Works Department. We receive several service requests
    from the public, and we need your help prioritizing them and providing further context. You will help us find service
    requests that are not assigned to the correct department to save us time and money.

    Background on city terms:
    - A traffic signal (or traffic light) is a signaling devices at intersections which direct traffic using colored lights.
    - A sign is usually attached to a pole and along the side of the road (such as a STOP sign) these are different than traffic signals

    Rules:
    - Always use a professional tone suitable for the workplace.
    - If there is not enough context provided to confidently classify the service_type then keep the original.
    - Only flag service requests as emergencies if they are a true immediate impact to public safety
    - Rate a typical service request as 5, a low priority as 0, emergencies should always be a 10.

    Your tasks:
    1. Check the selected service_type and compare it to all possible service_types to see if the correct one was selected.
    2. Check if this service request should be flagged as an emergency that poses a threat to the public
    3. Provide a short summary of the request.
    4. Provide recommendations for the staff who will address this service request on what to do next.
    5. A score from 0 to 10 rating the priority of this service request, based on its potential impact to the community.
    6. Report issues that the city can address that you see with the attached image
    7. Is the image relevant to the request?
    8. Summarize the image
    """

    service_type_definitions: list = dspy.InputField()
    description: str = dspy.InputField()
    service_type: str = dspy.InputField()
    address: str = dspy.InputField()
    lat: float = dspy.InputField()
    lon: float = dspy.InputField()
    image: dspy.Image = dspy.InputField()

    service_type_output: str = dspy.OutputField(
        description="The appropriate service_type for this service request"
    )
    is_emergency: bool = dspy.OutputField(
        description="Does this service request pose a threat to the public and needs to be addressed immediately?"
    )
    summary: str = dspy.OutputField(description="Summary of the service request.")
    recommendations: str = dspy.OutputField(
        description="Recommendations for the city's staff to address this service request."
    )
    rating: int = dspy.OutputField(
        description="A score from 0 to 10 rating the priority of this service request, based on its potential impact "
        "to the community."
    )
    issues_image_found: str = dspy.OutputField(
        description="Report issues that the city can address that you see with the attached image."
    )
    image_relevant_request: bool = dspy.OutputField(
        description="Does the image support the request?"
    )
    image_summary: str = dspy.OutputField(description="Summarize the image")


def eval():
    lm = load_llm()
    data = load_test_data()

    for entry in data:
        # basic QA
        predict = dspy.Predict(Augment311Request)
        # chain of thought
        file_location = f"../data/images/csr_images/{entry['service_request_id']}.jpg"
        # predict = dspy.ChainOfThought(Augment311Request)
        prediction = predict(
            service_type_definitions=process_request_types(),
            description=entry["description"],
            service_type=entry["service_name"],
            address=entry["address"],
            lat=entry["lat"],
            lon=entry["long"],
            image=dspy.Image.from_file(file_location),
        )
        print(lm.inspect_history(n=1))
        entry["service_type_output"] = prediction.service_type_output
        entry["is_emergency"] = prediction.is_emergency
        entry["recommendations"] = prediction.recommendations
        entry["summary"] = prediction.summary
        entry["rating"] = prediction.rating
        entry["issues_image_found"] = prediction.issues_image_found
        entry["image_relevant_request"] = prediction.image_relevant_request
        entry["image_summary"] = prediction.image_summary

    pd.DataFrame(data).to_csv("test_results_images.csv", index=False)


if __name__ == "__main__":
    eval()
