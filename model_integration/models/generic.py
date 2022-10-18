import torch
from transformers import Pipeline, pipeline
from typing import List, Dict, Any, Optional
from ..util import lookup_hypothesis
from util.config_handler import get_config_value
from util.connector import get_model_path
from util.notification import send_project_update

__classifier = {}


def get_labels_for_text(
    project_id: str,
    config: str,
    text: str,
    labels: List[str],
    information_source_id: Optional[str] = None,
) -> Dict[str, Any]:
    if __has_classifier(config):
        classifier = __get_classifier(config)
    else:
        classifier = __get_classifier_with_web_socket_update(
            project_id, information_source_id, config
        )
    hypothesis = lookup_hypothesis(config)
    if hypothesis:
        return classifier(
            text,
            labels,
            hypothesis_template=hypothesis,
        )

    return classifier(text, labels)


def __has_classifier(config: str) -> bool:
    return config in __classifier


def __get_classifier(config: str) -> Pipeline:
    global __classifier
    if config not in __classifier:
        if get_config_value("is_managed"):
            model = get_model_path(config)
        else:
            model = config

        if torch.cuda.is_available():
            __classifier[config] = pipeline(
                "zero-shot-classification", model=model, device=0
            )
        else:
            __classifier[config] = pipeline("zero-shot-classification", model=model)
    return __classifier[config]


def __get_classifier_with_web_socket_update(
    project_id: str, information_source_id: str, config: str
) -> Pipeline:
    if not project_id or not config:
        raise ValueError(
            f"Invalid project id or config for zero shot execution given. Project Id: {project_id}; Config: {config}."
        )
    information_source_id_str = (
        (f":{information_source_id}") if information_source_id else ""
    )
    send_project_update(
        project_id, f"zero_shot_download:started{information_source_id_str}"
    )
    classifier = __get_classifier(config)
    send_project_update(
        project_id, f"zero_shot_download:finished{information_source_id_str}"
    )
    return classifier
