import {useState} from "react";

export default function UploadPanel({busy, dataset, onUpload}) {
  const [file, setFile] = useState(null);

  const submit = async (event) => {
    event.preventDefault();
    if (!file || busy) {
      return;
    }
    await onUpload(file);
  };

  return (
    <section className="panel intake-panel">
      <div className="panel-header">
        <h2>Data Input</h2>
        <span className={dataset ? "badge success-badge" : "badge"}>
          {dataset ? "Loaded" : "Step 1"}
        </span>
      </div>
      <p className="muted">
        Upload a CSV or XLSX dataset. The backend stores the dataset in memory and returns the
        schema needed for the analysis workspace.
      </p>
      <form className="upload-form" onSubmit={submit}>
        <label className="file-picker">
          <span>{file ? file.name : "Choose dataset file"}</span>
          <input
            accept=".csv,.xlsx,.xls"
            onChange={(event) => setFile(event.target.files?.[0] ?? null)}
            type="file"
          />
        </label>
        <button disabled={!file || busy} type="submit">
          {busy ? "Uploading..." : "Upload Dataset"}
        </button>
      </form>

      <div className="intake-divider" />

      {dataset ? (
        <div className="dataset-summary-block">
          <div className="panel-header panel-subheader">
            <h3>Dataset Summary</h3>
            <span className="badge success-badge">Ready</span>
          </div>
          <div className="summary-grid">
            <div>
              <strong>File</strong>
              <p>{dataset.filename}</p>
            </div>
            <div>
              <strong>Rows</strong>
              <p>{dataset.row_count}</p>
            </div>
            <div>
              <strong>Columns</strong>
              <p>{dataset.column_count}</p>
            </div>
            <div>
              <strong>Numeric Columns</strong>
              <p>{dataset.numeric_columns.join(", ") || "None"}</p>
            </div>
            <div>
              <strong>Renderable Numeric Rows</strong>
              <p>{dataset.renderable_row_count}</p>
            </div>
          </div>
          <div className="preview-table-wrapper">
            <table className="preview-table">
              <thead>
                <tr>
                  {dataset.columns.map((column) => (
                    <th key={column}>{column}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {dataset.preview.map((row, rowIndex) => (
                  <tr key={rowIndex}>
                    {dataset.columns.map((column) => (
                      <td key={column}>{String(row[column] ?? "")}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        <div className="empty-summary">
          <div className="panel-header panel-subheader">
            <h3>Dataset Summary</h3>
            <span className="badge muted-badge">Waiting</span>
          </div>
          <p className="muted">Upload a dataset to inspect fields and enable predicate runs.</p>
        </div>
      )}
    </section>
  );
}
