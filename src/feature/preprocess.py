from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# try adding indicator for missing columns
# try filling in missing values by prediction (knn, random forest)
# imputer with mean, median, most_frequent
# add back in many category cols

numerical_cols = [
    "amount_tsh",
    "gps_height",
    "longitude",
    "latitude",
    "num_private",
    "population",
    "construction_year",
]

many_category_cols = [
    "funder",  # has nan
    "installer",  # has nan
    "wpt_name",
    "subvillage",  # has nan
    "ward",
    "scheme_name",
]

category_cols = [
    "basin",
    "region",
    "lga",
    "public_meeting",
    "scheme_management",
    "permit",
    "extraction_type",
    "extraction_type_group",
    "extraction_type_class",
    "management",
    "management_group",
    "payment",
    "water_quality",
    "quality_group",
    "quantity",
    "quantity_group",
    "source",
    "source_type",
    "source_class",
    "waterpoint_type",
    "region_code",
    "district_code",
]


def create_pipeline(model) -> Pipeline:
    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preparation = ColumnTransformer(
        [
            ("numerical", "passthrough", numerical_cols),
            ("categorical", cat_pipeline, category_cols),
        ],
    )

    pipeline = Pipeline([("preparation", preparation), ("model", model)])

    return pipeline
