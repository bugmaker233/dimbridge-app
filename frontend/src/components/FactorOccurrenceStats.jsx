import {useMemo} from "react";

import {buildFactorSummary} from "./factorReport";

function formatPercent(value) {
  return Number.isFinite(value) ? (value * 100).toFixed(1) + "%" : "-";
}

function formatRank(value) {
  return Number.isFinite(value) ? value.toFixed(2) : "-";
}

export default function FactorOccurrenceStats({history, onExport, onReset}) {
  const rows = useMemo(
    function () {
      return buildFactorSummary(history);
    },
    [history],
  );

  return (
    <section className="factor-stats-panel">
      <div className="factor-picker-header">
        <div>
          <strong>Predicate Factor Test Statistics</strong>
          <p className="muted">
            Cumulative occurrences from completed brush and objective-slider predicate subsets.
          </p>
        </div>
        <div className="factor-stats-actions">
          <span className="badge">{history.length} subsets</span>
          <button
            className="secondary-button"
            disabled={history.length === 0}
            onClick={onExport}
            type="button"
          >
            Export Report
          </button>
          <button
            className="secondary-button"
            disabled={history.length === 0}
            onClick={onReset}
            type="button"
          >
            Reset Statistics
          </button>
        </div>
      </div>

      {rows.length > 0 ? (
        <div className="factor-stats-table-wrap">
          <table className="factor-stats-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Factor</th>
                <th>Occurrences</th>
                <th>Hit Rate</th>
                <th>Average Predicate Rank</th>
                <th>Best Rank</th>
                <th>Average F1 Drop</th>
                <th>Average Range Reduction</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(function (row, index) {
                return (
                  <tr key={row.factor}>
                    <td>{row.overall_rank}</td>
                    <td>{row.factor}</td>
                    <td>{row.occurrence_count}</td>
                    <td>{formatPercent(row.hit_rate)}</td>
                    <td>{formatRank(row.average_predicate_rank)}</td>
                    <td>{row.best_rank}</td>
                    <td>{formatRank(row.average_importance_f1_drop)}</td>
                    <td>{formatPercent(row.average_range_reduction)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="muted factor-stats-empty">
          Select a sample subset and wait for predicate regression to populate statistics.
        </p>
      )}
    </section>
  );
}
