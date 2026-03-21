from pathlib import Path

from app.feature_engineering.features import FeatureExtractor

GPX_PATH = Path("data/Siria_Trail_Run_.gpx")

def main():
    with open(GPX_PATH, "rb") as f:
        file_bytes = f.read()

    extractor = FeatureExtractor()
    features = extractor.extract_from_gpx(file_bytes)
    print("Extracted features: ")
    for k,v in features.items():
       print(f"{k}: {v}")
    

if __name__ == "__main__":
    main()