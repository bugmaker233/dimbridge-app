function formatNumber(value) {
  return Number.isFinite(value) ? value.toFixed(3) : "-";
}

function formatPercent(value) {
  return Number.isFinite(value) ? (value * 100).toFixed(1) + "%" : "-";
}

const MAX_VISIBLE_FACTORS = 20;

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
      {result.predicates.map(function (predicate, predicateIndex) {
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
                {result.qualities && result.qualities[predicateIndex]
                  ? `F1 ${formatNumber(result.qualities[predicateIndex].f1)}`
                  : "No quality score"}
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
    </div>
  );
}
