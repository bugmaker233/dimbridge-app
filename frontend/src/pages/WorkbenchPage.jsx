import {useEffect, useState} from "react";

import AnalysisWorkspace from "../components/AnalysisWorkspace";
import PredicateResultView from "../components/PredicateResultView";
import UploadPanel from "../components/UploadPanel";
import {fetchHealth, uploadDataset} from "../services/api";

export default function WorkbenchPage() {
  const [health, setHealth] = useState("checking");
  const [dataset, setDataset] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    fetchHealth()
      .then(() => setHealth("online"))
      .catch(() => setHealth("offline"));
  }, []);

  const handleUpload = async (file) => {
    setBusy(true);
    setError("");
    try {
      const uploaded = await uploadDataset(file);
      setDataset(uploaded);
      setAnalysis(null);
      setResult(null);
    } catch (uploadError) {
      setError(uploadError.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">DimBridge Standalone</p>
          <h1>Interactive Predicate Analysis Workbench</h1>
          <p className="hero-copy">
            This app scaffold replaces the Jupyter widget shell with a standard
            frontend/backend architecture while preserving the original DimBridge analysis flow.
          </p>
        </div>
        <div className={`status-pill ${health}`}>
          Backend: {health === "online" ? "Online" : health === "offline" ? "Offline" : "Checking"}
        </div>
      </header>

      {error ? <p className="error-banner">{error}</p> : null}

      <UploadPanel busy={busy} dataset={dataset} onUpload={handleUpload} />

      <AnalysisWorkspace
        analysis={analysis}
        dataset={dataset}
        onProjectionChange={setAnalysis}
        onResult={setResult}
      />

      <section className="panel result-panel">
        <div className="panel-header">
          <h2>API Result</h2>
          <span className="badge">Step 3</span>
        </div>
        <p className="muted">
          Predicate responses are shown against the current projected factor space rather than
          the raw uploaded table.
        </p>
        <PredicateResultView analysis={analysis} result={result} />
        <pre className="result-block">
          {result ? JSON.stringify(result, null, 2) : "No predicate result yet."}
        </pre>
      </section>
    </main>
  );
}
