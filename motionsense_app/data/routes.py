from typing import Annotated, Literal

import numpy as np
from fastapi import APIRouter, Query

from motionsense_app.data.schema import DatasetInfo, FeatureList, Sample, SamplePage
from motionsense_app.dependencies import StoreDependency, require_dataset

router = APIRouter(prefix="/api/data")


@router.get("", response_model=DatasetInfo)
def data_info(store: StoreDependency) -> dict:
    return store.info()


@router.get("/features", response_model=FeatureList)
def features(store: StoreDependency) -> dict:
    require_dataset(store)
    return {"items": store.features()}


@router.get("/samples", response_model=SamplePage)
def samples(
    store: StoreDependency,
    split: Literal["train", "test"] = "test",
    subject_id: Annotated[int | None, Query(gt=0)] = None,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> dict:
    _, labels, subjects = store.arrays(split)
    indexes = (range(len(labels)) if subject_id is None
               else np.flatnonzero(subjects == subject_id))
    return {
        "items": [
            {"sample_id": f"{split}:{index + 1:06d}", "subject_id": int(subjects[index]),
             "split": split, "actual_label": int(labels[index])}
            for index in indexes[offset:offset + limit]
        ],
        "offset": offset, "limit": limit, "total": len(indexes),
    }


@router.get("/samples/{sample_id}", response_model=Sample)
def sample(sample_id: str, store: StoreDependency) -> dict:
    return store.sample(sample_id)
