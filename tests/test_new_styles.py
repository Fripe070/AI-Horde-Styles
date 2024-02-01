import json
import os
import unittest
from pathlib import Path
from typing import Any, Annotated

import requests
from horde_sdk.ai_horde_api.apimodels import ImageGenerateAsyncRequest
from pydantic import BaseModel, StringConstraints, field_validator, Field, model_validator, ConfigDict, ValidationError


# noinspection PyNestedDecorators
class StyleImageGenerationRequest(ImageGenerateAsyncRequest):
    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, prompt: str) -> None:
        positive_prompt, *negative_prompt = prompt.split("###")
        positive_prompt: str
        negative_prompt: str | None = "###".join(negative_prompt) or None

        assert positive_prompt.count("{p}") == 1, "Positive prompt must contain exactly one {p} placeholder"
        assert positive_prompt.count("{np}") == 0, "Positive prompt must not contain {np}"
        if negative_prompt is not None:
            assert negative_prompt.count("{np}") == 1, "Negative prompt must contain exactly one {np} placeholder"
            assert negative_prompt.count("{p}") == 0, "Negative prompt must not contain {p}"


# noinspection PyTypeHints
class Style(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Annotated[str, StringConstraints(min_length=1)]
    tags: list[Annotated[str, StringConstraints(min_length=1, pattern=r"^[a-z\-_]+$")]] = Field(default_factory=list)
    apply_enhancements: bool = False

    request: StyleImageGenerationRequest

    @model_validator(mode="after")
    def validate_model(self) -> "Style":
        models = self.request.models
        reference_models = []
        for model in models:
            reference_model = model_reference.get(model)
            assert reference_model is not None, f"Unknown model {model}"
            reference_models.append(reference_model)

        if not self.apply_enhancements:
            return self

        assert len(models) > 0, "At least one model must be specified in order to apply enhancements"
        unique_baselines = set(reference_model["baseline"] for reference_model in reference_models)
        assert len(unique_baselines) == 1, (
            f"All models must use the same baseline, but got {len(unique_baselines)} baselines: "
            f"{', '.join(unique_baselines)}"
        )

        return self


def get_github_json_file(url: str) -> dict[str, Any]:
    response = requests.get(url, headers={"Accept": "application/vnd.github.raw+json"})
    response.raise_for_status()
    return response.json()


model_reference: dict[str, dict[str, Any]] = get_github_json_file(
    # "https://api.github.com/repos/Haidra-Org/AI-Horde-image-model-reference/contents/stable_diffusion.json"
    "https://github.com/Haidra-Org/AI-Horde-image-model-reference/raw/main/stable_diffusion.json"
)

ENHANCEMENTS_PATH = Path("enhancements.json")
STYLES_PATH = Path("new-styles.json")

with open(ENHANCEMENTS_PATH, "r", encoding="utf-8") as file:
    enhancements: dict[str, dict[str, Any]] = json.load(file)


class TestNewStyles(unittest.TestCase):
    def setUp(self):
        with open(STYLES_PATH, "r", encoding="utf-8") as file:
            self.styles: list[dict[str, Any]] = json.load(file)

    def test_styles(self) -> None:
        print(f"Validating {len(self.styles)} styles from {STYLES_PATH.as_posix()}")

        self.assertTrue(isinstance(self.styles, list))
        self.assertTrue(all(isinstance(style, dict) for style in self.styles))

        for style in self.styles:
            with self.subTest(style=style.get("name", style)):
                Style.model_validate(style, strict=True)

        if len(self.styles) != len(set(style["name"].lower() for style in self.styles)):
            self.fail("Style names must be unique, ignoring case")
