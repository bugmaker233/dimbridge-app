export default function DatasetSummary({dataset}) {
  if (!dataset) {
    return (
      <section className="panel">
        <div className="panel-header">
          <h2>Dataset Summary</h2>
          <span className="badge muted-badge">Waiting</span>
        </div>
        <p className="muted">Upload a dataset to inspect fields and enable predicate runs.</p>
      </section>
    );
  }

  return (
    <section className="panel">
      <div className="panel-header">
        <h2>Dataset Summary</h2>
        <span className="badge success-badge">Loaded</span>
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
    </section>
  );
}
