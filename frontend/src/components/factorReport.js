export function buildFactorSummary(history) {
  const totalSubsets = history.length;
  const factorMap = new Map();

  history.forEach(function (event) {
    const seen = new Set();
    (event.factors || []).forEach(function (factor, index) {
      if (!factor.attribute || seen.has(factor.attribute)) {
        return;
      }
      seen.add(factor.attribute);
      const rank = Number.isFinite(Number(factor.rank)) ? Number(factor.rank) : index + 1;
      const importance = Number.isFinite(Number(factor.importance)) ? Number(factor.importance) : 0;
      const rangeReduction = Number.isFinite(Number(factor.range_reduction))
        ? Number(factor.range_reduction)
        : 0;
      const current = factorMap.get(factor.attribute) || {
        attribute: factor.attribute,
        count: 0,
        rankSum: 0,
        bestRank: rank,
        importanceSum: 0,
        maxImportance: importance,
        rangeReductionSum: 0,
      };
      current.count += 1;
      current.rankSum += rank;
      current.bestRank = Math.min(current.bestRank, rank);
      current.importanceSum += importance;
      current.maxImportance = Math.max(current.maxImportance, importance);
      current.rangeReductionSum += rangeReduction;
      factorMap.set(factor.attribute, current);
    });
  });

  return Array.from(factorMap.values())
    .map(function (factor) {
      return {
        factor: factor.attribute,
        occurrence_count: factor.count,
        hit_rate: totalSubsets > 0 ? factor.count / totalSubsets : 0,
        average_predicate_rank: factor.count > 0 ? factor.rankSum / factor.count : null,
        best_rank: factor.bestRank,
        average_importance_f1_drop:
          factor.count > 0 ? factor.importanceSum / factor.count : null,
        max_importance_f1_drop: factor.maxImportance,
        average_range_reduction:
          factor.count > 0 ? factor.rangeReductionSum / factor.count : null,
      };
    })
    .sort(function (first, second) {
      return (
        second.occurrence_count - first.occurrence_count ||
        second.average_importance_f1_drop - first.average_importance_f1_drop ||
        first.average_predicate_rank - second.average_predicate_rank ||
        first.factor.localeCompare(second.factor)
      );
    })
    .map(function (factor, index) {
      return {overall_rank: index + 1, ...factor};
    });
}

function safeFilename(value) {
  return String(value || "dataset")
    .replace(/\.[^.]+$/, "")
    .replace(/[^\p{L}\p{N}._-]+/gu, "-")
    .replace(/^-+|-+$/g, "") || "dataset";
}

function compactTimestamp(date) {
  return date.toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z").replace("T", "-");
}

export function downloadFactorReport({analysis, dataset, history, sessionStartedAt}) {
  const generatedAt = new Date();
  const report = {
    schema_version: 1,
    session: {
      dataset_id: analysis?.dataset_id || dataset?.dataset_id || null,
      dataset_filename: dataset?.filename || null,
      started_at: sessionStartedAt,
      generated_at: generatedAt.toISOString(),
      completed_subset_count: history.length,
      projection: analysis
        ? {
            method: analysis.method,
            supervision: analysis.supervision,
            standardize: analysis.standardize,
            factor_columns: analysis.factor_columns || [],
            objective_columns: analysis.objective_columns || [],
            target_column: analysis.target_column || null,
          }
        : null,
    },
    selection_history: history,
    factor_summary: buildFactorSummary(history),
  };
  const blob = new Blob([JSON.stringify(report, null, 2)], {type: "application/json"});
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `critical-factor-report_${safeFilename(dataset?.filename)}_${compactTimestamp(generatedAt)}.json`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}
