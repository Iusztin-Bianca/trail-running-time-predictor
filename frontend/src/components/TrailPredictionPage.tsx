import { useState } from "react";
import { api } from "../api/client";
import trailImage from "../images/trail_running_path.png";
import ElevationProfile from "./ElevationProfile";
import { parseGpx, type GpxStats } from "../utils/gpxParser";
import "./TrailPredictionPage.css";

export default function TrailPredictionPage() {
  const [gpxFile, setGpxFile] = useState<File | null>(null);
  const [gpxStats, setGpxStats] = useState<GpxStats | null>(null);
  const [predictedTime, setPredictedTime] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showModal, setShowModal] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;

    const file = e.target.files[0];

    if (!file.name.endsWith(".gpx")) {
      setError("Please upload a valid GPX file.");
      setGpxFile(null);
      return;
    }

    setError(null);
    setGpxFile(file);

    const text = await file.text();
    setGpxStats(parseGpx(text));
  };

  const submit = async () => {
    if (!gpxFile) {
      setError("Please upload a GPX file first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", gpxFile);

    setLoading(true);
    setError(null);

    try {
      const response = await api.post("/predict-from-gpx", formData, {
        params: { is_race: 1, is_easy: 0 },
      });
      setPredictedTime(response.data.predicted_time_formatted);
      setShowModal(true);
    } catch (err: any) {
      setError(err.response?.data?.detail ?? "Prediction failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <img src={trailImage} alt="Trail runner" className="heroImage" />
      <div className="heroOverlay" />

      {/* Title above card */}
      <div className="heroText">
        <h1 className="heroTitle">Predict Your Trail Race Time</h1>
        <p className="heroSubtitle">Upload a .gpx file to get your finish time estimate</p>
      </div>

      {/* Main Card */}
      <div className="card">
        <label className="uploadBox">
          <input
            type="file"
            accept=".gpx"
            style={{ display: "none" }}
            onChange={handleFileChange}
          />
          <div className="uploadContent">
            <div className="icon">📁</div>
            <span>{gpxFile ? gpxFile.name : "Upload a GPX file"}</span>
          </div>
        </label>

        {error && <p className="errorText">{error}</p>}

        <button className="button" onClick={submit} disabled={loading}>
          {loading ? "Predicting..." : "Predict your race time!"}
        </button>
      </div>

      {/* Result modal */}
      {showModal && (
        <div className="modalOverlay" onClick={() => setShowModal(false)}>
          <div className="modalCard" onClick={(e) => e.stopPropagation()}>

            <div className="modalHeader">
              <p className="modalTitle">Predicted Race Time</p>
              <p className="modalTime">{predictedTime}</p>
            </div>

            <div className="modalBody">
              {gpxStats && (
                <div className="modalStats">
                  <div className="statPill">
                    <span className="statValue">{gpxStats.totalDistanceKm.toFixed(1)} km</span>
                    <span className="statLabel">Distance</span>
                  </div>
                  <div className="statPill">
                    <span className="statValue">+{gpxStats.elevationGainM} m</span>
                    <span className="statLabel">Elevation gain</span>
                  </div>
                </div>
              )}

              {gpxStats && gpxStats.elevationProfile.length > 1 && (
                <div className="elevationContainer">
                  <p className="elevationLabel">Elevation profile</p>
                  <ElevationProfile profile={gpxStats.elevationProfile} />
                </div>
              )}
            </div>

            <div className="modalFooter">
              <button className="button" onClick={() => setShowModal(false)}>
                Close
              </button>
            </div>

          </div>
        </div>
      )}
    </div>
  );
}
