import { useState } from "react";
import { api } from "../api/client";
import trailImage from "../assets/trail.jpg";
import "./PredictionForm.css";

export default function PredictionForm() {
  const [gpxFile, setGpxFile] = useState<File | null>(null);
  const [predictedTime, setPredictedTime] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showModal, setShowModal] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;

    const file = e.target.files[0];

    if (!file.name.endsWith(".gpx")) {
      setError("Please upload a valid GPX file.");
      setGpxFile(null);
      return;
    }

    setError(null);
    setGpxFile(file);
  };

  const submit = async () => {
    if (!gpxFile) {
      setError("Please upload a GPX file first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", gpxFile);
  
    try {
      const response = await api.post("/predict-from-gpx", formData);
      console.log("AAAA: ", response)
      setPredictedTime(response.data.predicted_time_minutes);
      setShowModal(true);
    } catch (err: any) {
      console.error("Backend error:", err.response?.data);
      setError(err.response?.data?.detail ?? "Prediction failed.");
    }
  };

  return (
    <div className="page">
      <h1 className="title">Trail Running Race Prediction</h1>

      <div className="content">
         {/* Left side */}
         <div className="left">
          <img src={trailImage} alt="Trail running" className="image" />
        </div>
        {/* Right side */}
        <div className="right">
          <div className="label">
            <label className="label">Upload a GPX file</label>
          </div>
          <div className="inputFile">
            <input
              type="file"
              accept=".gpx"
              onChange={handleFileChange}
              className="fileInput"
            />
          </div> 

          <button onClick={submit} className="predictButton">
            Predict race time
          </button>

          {error && <p className="error">{error}</p>}
        </div>
      </div>

      {showModal && predictedTime !== null && (
        <div className="modalOverlay">
          <div className="modal">
            <h2>Prediction Result</h2>
            <p>{predictedTime} minutes</p>
            <button
              onClick={() => setShowModal(false)}
              className="closeButton"
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
