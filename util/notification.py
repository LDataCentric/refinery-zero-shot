import requests
import os
from submodules.model.business_objects import project


def send_project_update(project_id, message, is_global=False):
    endpoint = os.getenv("WS_NOTIFY_ENDPOINT")
    if not endpoint:
        print(
            "- WS_NOTIFY_ENDPOINT not set -- did you run the start script?", flush=True
        )
        return

    message = f"GLOBAL:{message}" if is_global else f"{project_id}:{message}"
    project_item = project.get(project_id)
    organization_id = str(project_item.organization_id)
    req = requests.post(
        f"{endpoint}/notify",
        json={
            "organization": organization_id,
            "message": message,
        },
    )
    if req.status_code != 200:
        print("Could not send notification update", flush=True)
