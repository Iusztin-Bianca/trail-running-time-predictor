import pandas as pd
from pathlib import Path
from app.feature_engineering.features import FeatureExtractor

DATASET_PATH = Path("data/features.parquet")
RAW_GPX_DIR = Path("data/raw_gpx")

rows = []
extractor = FeatureExtractor()

for gpx_file in Path("data/raw_gpx").glob("*.gpx"):
    with open(gpx_file, "rb") as f:
        features = extractor.extract_from_gpx(f.read())
        features["activity_id"] = gpx_file.stem
        rows.append(features)

df = pd.DataFrame(rows)

df.to_parquet("data/features.parquet", index = False)