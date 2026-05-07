from typing import Self

from pydantic import BaseModel, ConfigDict, create_model, field_validator, model_validator

from app.model import feature_cols


EXPECTED_FEATURE_COUNT = 44


class _BearingFeaturesBase(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                feature_name: 0.0
                for feature_name in feature_cols
            }
        },
    )

    @field_validator("*", mode="before", check_fields=False)
    @classmethod
    def reject_boolean_features(cls, feature_value: object) -> object:
        if isinstance(feature_value, bool):
            raise ValueError("Feature values must be numeric.")
        return feature_value

    @model_validator(mode="after")
    def validate_feature_count(self) -> Self:
        received_feature_count = len(self.model_dump())
        if received_feature_count != EXPECTED_FEATURE_COUNT:
            raise ValueError("Exactly 44 feature values are required.")
        return self


BearingFeatures = create_model(
    "BearingFeatures",
    __base__=_BearingFeaturesBase,
    __module__=__name__,
    **{feature_name: (float, ...) for feature_name in feature_cols},
)


class RULPrediction(BaseModel):
    predicted_rul_seconds: float
    predicted_rul_minutes: float
    n_features_received: int
