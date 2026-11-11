function formatNumber(value) {
  return Number.isFinite(value) ? value.toFixed(3) : "-";
}

function formatPercent(value) {
  return Number.isFinite(value) ? (value * 100).toFixed(1) + "%" : "-";
}

const MAX_VISIBLE_FACTORS = 20;

function formatAlgorithm(algorithm) {
  const labels = {
    "data-extent": "Data Extent",
    "legacy-predicate-regression": "Predicate Regression (Legacy)",
    "paper-predicate-regression": "Predicate Regression (Paper)",
    "recursive-predicate-induction": "RPI (Exact Observed Intervals)",
  };
  return labels[algorithm] || algorithm || "Predicate Result";
}

function formatQuality(quality) {
  if (!quality) {
    return "No quality score";
  }
  if (Number.isFinite(Number(quality.predicate_f1))) {
    return `Proxy F1 ${formatNumber(Number(quality.f1))} | Rule F1 ${formatNumber(
      Number(quality.predicate_f1),
    )}`;
  }
  return `F1 ${formatNumber(Number(quality.f1))}`;
}

function normalizeInterval(interval, extent) {
  if (!interval || !extent) {
    return {start: 0, width: 100};
  }

  const globalMin = extent[0];
  const globalMax = extent[1];
  const span = globalMax - globalMin || 1;
  const start = ((interval[0] - globalMin) / span) * 100;
  const end = ((interval[1] - globalMin) / span) * 100;
  return {
    start: Math.max(0, Math.min(100, start)),
    width: Math.max(2, Math.min(100, end - start)),
  };
}

export default function PredicateResultView({analysis, result}) {
  if (!result) {
    return (
      <div className="selection-summary">
        <strong>Predicate Result</strong>
        <p className="muted">Brush inside the generated projection to populate interval explanations.</p>
      </div>
    );
  }

  const factorColumns = analysis ? analysis.factor_columns || [] : [];
  const records = analysis ? analysis.records || [] : [];
  const candidateSolutions = Array.isArray(result.candidate_solutions)
    ? result.candidate_solutions
    : [];
  const searchWasTruncated = Boolean(
    result.diagnostics?.brushes?.some(function (brush) {
      return brush.truncated;
    }),
  );
  const solutionsWereLimited = Boolean(
    result.diagnostics?.brushes?.some(function (brush) {
      return brush.solutions_limited;
    }),
  );
  const depthWasLimited = Boolean(
    result.diagnostics?.brushes?.some(function (brush) {
      return brush.depth_limited;
    }),
  );
  const beamWasLimited = Boolean(
    result.diagnostics?.brushes?.some(function (brush) {
      return brush.beam_limited;
    }),
  );
  const intervalBranchesWereLimited = Boolean(
    result.diagnostics?.brushes?.some(function (brush) {
      return brush.interval_branch_limited;
    }),
  );
  const extents = Object.fromEntries(
    factorColumns.map(function (column) {
      const values = records
        .map(function (record) {
          return Number(record[column]);
        })
        .filter(function (value) {
          return Number.isFinite(value);
        });

      if (!values.length) {
        return [column, [0, 1]];
      }

      const min = Math.min.apply(null, values);
      const max = Math.max.apply(null, values);
      return [column, [min, max]];
    }),
  );

  return (
    <div className="predicate-result">
      <div className="predicate-result-meta">
        <strong>{formatAlgorithm(result.algorithm)}</strong>
        {searchWasTruncated ? (
          <span className="warning-text">Search budget reached; candidates are incomplete.</span>
        ) : depthWasLimited ? (
          <span className="warning-text">Search stopped at the configured maximum depth.</span>
        ) : intervalBranchesWereLimited ? (
          <span className="muted">Per-factor interval limits pruned recursive branches.</span>
        ) : beamWasLimited ? (
          <span className="muted">Beam width limited recursive branches.</span>
        ) : solutionsWereLimited ? (
          <span className="muted">Additional candidates were omitted by the solution limit.</span>
        ) : null}
      </div>

      {(result.predicates || []).map(function (predicate, predicateIndex) {
        const rankedClauses = predicate
          .slice()
          .sort(function (first, second) {
            return (
              (Number(second.importance) || 0) - (Number(first.importance) || 0) ||
              (Number(second.range_reduction) || 0) -
                (Number(first.range_reduction) || 0) ||
              (Number(first.dim) || 0) - (Number(second.dim) || 0)
            );
          });
        const visibleClauses = rankedClauses.slice(0, MAX_VISIBLE_FACTORS);
        return (
          <section className="predicate-group" key={"predicate-" + predicateIndex}>
            <div className="predicate-group-header">
              <strong>Brush {predicateIndex + 1}</strong>
              <span className="predicate-quality">
                {predicate.length > MAX_VISIBLE_FACTORS
                  ? `Top ${MAX_VISIBLE_FACTORS} of ${predicate.length} factors | `
                  : `${predicate.length} factors | `}
                {formatQuality(result.qualities?.[predicateIndex])}
              </span>
            </div>

            {visibleClauses.length > 0 ? (
              visibleClauses.map(function (clause, clauseIndex) {
                const extent = extents[clause.attribute];
                const bar = normalizeInterval(clause.interval, extent);
                const importance = Number(clause.importance);
                const rangeReduction = Number(clause.range_reduction);
                return (
                  <div className="predicate-clause" key={predicateIndex + "-" + clause.attribute}>
                    <div className="predicate-clause-header">
                      <span>
                        #{Number.isFinite(Number(clause.rank)) ? clause.rank : clauseIndex + 1}{" "}
                        {clause.attribute}
                      </span>
                      <span>
                        {formatNumber(clause.interval[0])} to {formatNumber(clause.interval[1])}
                      </span>
                    </div>
                    {Number.isFinite(importance) || Number.isFinite(rangeReduction) ? (
                      <div className="predicate-clause-meta">
                        <span>Importance (F1 drop) {formatNumber(importance)}</span>
                        <span>Range reduction {formatPercent(rangeReduction)}</span>
                      </div>
                    ) : null}
                    <div className="predicate-track">
                      <div
                        className="predicate-bar"
                        style={{left: bar.start + "%", width: bar.width + "%"}}
                      />
                    </div>
                  </div>
                );
              })
            ) : (
              <p className="muted">No interval clauses returned for this brush.</p>
            )}
          </section>
        );
      })}

      {candidateSolutions.length ? (
        <section className="predicate-alternatives">
          <div className="predicate-group-header">
            <strong>RPI Candidate Solutions</strong>
            <span className="predicate-quality">
              The first candidate drives the coordinated DimBridge views.
            </span>
          </div>
          {candidateSolutions.map(function (brushCandidates, brushIndex) {
            return (
              <details className="candidate-brush" key={"candidate-brush-" + brushIndex} open>
                <summary>
                  Brush {brushIndex + 1}: {brushCandidates.length} candidate
                  {brushCandidates.length === 1 ? "" : "s"}
                </summary>
                <div className="candidate-grid">
                  {brushCandidates.map(function (candidate) {
                    const quality = candidate.quality || {};
                    return (
                      <article
                        className={candidate.rank === 1 ? "candidate-card primary" : "candidate-card"}
                        key={candidate.signature || candidate.rank}
                      >
                        <div className="candidate-card-header">
                          <strong>#{candidate.rank}</strong>
                          <span>F1 {formatNumber(quality.f1)}</span>
                        </div>
                        <div className="candidate-metrics">
                          <span>Precision {formatNumber(quality.precision)}</span>
                          <span>Recall {formatNumber(quality.recall)}</span>
                          <span>
                            TP {candidate.true_positive_count} / Predicted {candidate.predicted_count}
                          </span>
                        </div>
                        <div className="candidate-clauses">
                          {(candidate.predicate || []).map(function (clause) {
                            return (
                              <span
                                className="candidate-clause"
                                key={`${clause.interval_id || clause.dim}:${clause.attribute}`}
                              >
                                {clause.attribute}: {formatNumber(clause.interval?.[0])} to{" "}
                                {formatNumber(clause.interval?.[1])}
                              </span>
                            );
                          })}
                          {!candidate.predicate?.length ? (
                            <span className="muted">TRUE (no discriminating clause)</span>
                          ) : null}
                        </div>
                      </article>
                    );
                  })}
                </div>
              </details>
            );
          })}
        </section>
      ) : null}
    </div>
  );
}
