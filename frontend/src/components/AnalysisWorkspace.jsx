import {useCallback, useEffect, useMemo, useState} from "react";

import CorrelationHeatmap from "./CorrelationHeatmap";
import FactorOccurrenceStats from "./FactorOccurrenceStats";
import {downloadFactorReport} from "./factorReport";
import LegacyDimbridgeWidget from "./LegacyDimbridgeWidget";
import {runProjection} from "../services/api";

function buildDefaultConfig(dataset) {
  const numericColumns = dataset ? dataset.numeric_columns : [];
  if (!numericColumns.length) {
    return {
      factorColumns: [],
      targetColumn: "",
      projectionMode: "umap",
    };
  }

  if (numericColumns.length >= 3) {
    return {
      factorColumns: numericColumns.slice(0, numericColumns.length - 1),
      targetColumn: numericColumns[numericColumns.length - 1],
      projectionMode: "supervised-umap",
    };
  }

  return {
    factorColumns: numericColumns.slice(),
    targetColumn: "",
    projectionMode: "umap",
  };
}

function projectionRequestConfig(projectionMode) {
  if (projectionMode === "supervised-umap") {
    return {method: "umap", supervision: "supervised"};
  }
  return {method: projectionMode, supervision: "unsupervised"};
}

function formatProjectionMode(projectionMode) {
  if (projectionMode === "pca") {
    return "PCA";
  }
  if (projectionMode === "tsne") {
    return "t-SNE";
  }
  return projectionMode === "supervised-umap" ? "Supervised UMAP" : "UMAP";
}

function formatMetricValue(value) {
  if (!Number.isFinite(value)) {
    return "-";
  }
  if (Math.abs(value) >= 100) {
    return value.toFixed(1);
  }
  if (Math.abs(value) >= 10) {
    return value.toFixed(2);
  }
  return value.toFixed(3);
}

function quantile(values, probability) {
  const sorted = values
    .filter(function (value) {
      return Number.isFinite(value);
    })
    .slice()
    .sort(function (a, b) {
      return a - b;
    });

  if (!sorted.length) {
    return null;
  }
  if (sorted.length === 1) {
    return sorted[0];
  }

  const position = (sorted.length - 1) * probability;
  const lowerIndex = Math.floor(position);
  const upperIndex = Math.ceil(position);
  const weight = position - lowerIndex;
  return sorted[lowerIndex] * (1 - weight) + sorted[upperIndex] * weight;
}

function buildObjectiveExtents(analysis) {
  if (!analysis) {
    return {};
  }

  const records = analysis.records || [];
  const objectiveColumns = analysis.objective_columns || [];
  return Object.fromEntries(
    objectiveColumns
      .map(function (column) {
        const values = records
          .map(function (record) {
            return Number(record[column]);
          })
          .filter(function (value) {
            return Number.isFinite(value);
          });

        if (!values.length) {
          return null;
        }

        const min = Math.min.apply(null, values);
        const max = Math.max.apply(null, values);
        const span = max - min;
        return [
          column,
          {
            min,
            max,
            values,
            step: span > 0 ? span / 100 : 1,
          },
        ];
      })
      .filter(Boolean),
  );
}

export default function AnalysisWorkspace({analysis, dataset, onProjectionChange, onResult}) {
  const [predicateMode, setPredicateMode] = useState("regression");
  const [predicateFactorLimit, setPredicateFactorLimit] = useState(20);
  const [selectionHistory, setSelectionHistory] = useState([]);
  const [sessionStartedAt, setSessionStartedAt] = useState(() => new Date().toISOString());
  const [brushMode, setBrushMode] = useState("contrastive");
  const [resetToken, setResetToken] = useState(0);
  const [error, setError] = useState("");
  const [projectionError, setProjectionError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [projectionBusy, setProjectionBusy] = useState(false);
  const [widgetMeta, setWidgetMeta] = useState({
    totalRows: 0,
    validRows: 0,
    droppedRows: 0,
    hasRenderableData: false,
  });

  const defaultConfig = useMemo(() => buildDefaultConfig(dataset), [dataset]);

  const [factorColumns, setFactorColumns] = useState([]);
  const [objectiveColumns, setObjectiveColumns] = useState([]);
  const [objectiveFilters, setObjectiveFilters] = useState({});
  const [objectiveDirections, setObjectiveDirections] = useState({});
  const [objectiveSegmentCounts, setObjectiveSegmentCounts] = useState({});
  const [objectiveActiveSegments, setObjectiveActiveSegments] = useState({});
  const [objectiveSelectionSource, setObjectiveSelectionSource] = useState("objective-slider");
  const [currentSelectionMask, setCurrentSelectionMask] = useState(null);
  const [correlationPairs, setCorrelationPairs] = useState([]);
  const [correlationPredicateSelection, setCorrelationPredicateSelection] = useState({
    columns: [],
    version: 0,
  });
  const [targetColumn, setTargetColumn] = useState("");
  const [projectionMode, setProjectionMode] = useState("umap");
  const [standardize, setStandardize] = useState(true);
  const [nNeighbors, setNNeighbors] = useState(20);
  const [minDist, setMinDist] = useState(0.1);
  const [targetWeight, setTargetWeight] = useState(0.5);
  const [perplexity, setPerplexity] = useState(30);
  const [learningRate, setLearningRate] = useState(200);
  const [maxIterations, setMaxIterations] = useState(1000);
  const [needsProjection, setNeedsProjection] = useState(false);

  useEffect(() => {
    setFactorColumns(defaultConfig.factorColumns);
    setObjectiveColumns([]);
    setObjectiveFilters({});
    setObjectiveDirections({});
    setObjectiveSegmentCounts({});
    setObjectiveActiveSegments({});
    setTargetColumn(defaultConfig.targetColumn);
    setProjectionMode(defaultConfig.projectionMode);
    setStandardize(true);
    setNNeighbors(20);
    setMinDist(0.1);
    setTargetWeight(0.5);
    setPerplexity(30);
    setLearningRate(200);
    setMaxIterations(1000);
    setNeedsProjection(Boolean(dataset));
    setProjectionError("");
    setError("");
    setIsLoading(false);
    setProjectionBusy(false);
    setSelectionHistory([]);
    setSessionStartedAt(new Date().toISOString());
    setResetToken(function (previous) {
      return previous + 1;
    });
    setWidgetMeta({
      totalRows: 0,
      validRows: 0,
      droppedRows: 0,
      hasRenderableData: false,
    });
    onProjectionChange(null);
    onResult(null);
  }, [dataset, defaultConfig, onProjectionChange, onResult]);

  useEffect(() => {
    setError("");
    setIsLoading(false);
    onResult(null);
  }, [predicateMode, brushMode, onResult]);

  useEffect(() => {
    setCurrentSelectionMask(null);
    setCorrelationPairs([]);
    setCorrelationPredicateSelection(function (previous) {
      return {columns: [], version: previous.version + 1};
    });
  }, [analysis]);

  const focusedCorrelationAttributes = useMemo(
    function () {
      return Array.from(
        new Set(
          correlationPairs.flatMap(function (pair) {
            return pair;
          }),
        ),
      );
    },
    [correlationPairs],
  );

  const targetOptions = dataset ? dataset.columns : [];
  const numericColumns = dataset ? dataset.numeric_columns : [];

  const handlePredicateResult = useCallback(
    function (payload) {
      onResult(payload);
      if (predicateMode !== "regression" || !Array.isArray(payload?.predicates)) {
        return;
      }

      const rankedFactors = new Map();
      payload.predicates.flat().forEach(function (clause, index) {
        const attribute = clause?.attribute;
        if (typeof attribute !== "string" || !attribute) {
          return;
        }
        const importance = Number.isFinite(Number(clause.importance))
          ? Number(clause.importance)
          : 0;
        const rangeReduction = Number.isFinite(Number(clause.range_reduction))
          ? Number(clause.range_reduction)
          : 0;
        const rank = Number.isFinite(Number(clause.rank)) ? Number(clause.rank) : index + 1;
        const previous = rankedFactors.get(attribute);
        if (!previous) {
          rankedFactors.set(attribute, {importance, rangeReduction, rank, index});
          return;
        }
        previous.importance = Math.max(previous.importance, importance);
        previous.rangeReduction = Math.max(previous.rangeReduction, rangeReduction);
        previous.rank = Math.min(previous.rank, rank);
      });
      const inferredColumns = Array.from(rankedFactors.entries())
        .sort(function ([firstAttribute, first], [secondAttribute, second]) {
          return (
            second.importance - first.importance ||
            second.rangeReduction - first.rangeReduction ||
            first.rank - second.rank ||
            first.index - second.index ||
            firstAttribute.localeCompare(secondAttribute)
          );
        })
        .map(function ([attribute]) {
          return attribute;
        });
      setCorrelationPredicateSelection(function (previous) {
        return {columns: inferredColumns, version: previous.version + 1};
      });
      setCorrelationPairs([]);

      const eventMeta = payload.event_meta || {};
      const events = payload.predicates.map(function (predicate, subsetIndex) {
        const quality = Array.isArray(payload.qualities) ? payload.qualities[subsetIndex] : null;
        return {
          event_id: `${eventMeta.completed_at || new Date().toISOString()}#${eventMeta.request_id || 0}-${subsetIndex + 1}`,
          source: eventMeta.source || "brush",
          requested_at: eventMeta.requested_at || null,
          completed_at: eventMeta.completed_at || new Date().toISOString(),
          subset_index: subsetIndex,
          selected_count: eventMeta.selected_counts?.[subsetIndex] ?? null,
          total_count: eventMeta.total_count ?? null,
          quality: quality
            ? {
                accuracy: quality.accuracy,
                precision: quality.precision,
                recall: quality.recall,
                f1: quality.f1,
              }
            : null,
          factors: (Array.isArray(predicate) ? predicate : []).map(function (clause, index) {
            return {
              attribute: clause.attribute,
              rank: Number.isFinite(Number(clause.rank)) ? Number(clause.rank) : index + 1,
              interval: Array.isArray(clause.interval) ? clause.interval.slice() : null,
              importance: Number.isFinite(Number(clause.importance))
                ? Number(clause.importance)
                : 0,
              f1_without: Number.isFinite(Number(clause.f1_without))
                ? Number(clause.f1_without)
                : null,
              range_reduction: Number.isFinite(Number(clause.range_reduction))
                ? Number(clause.range_reduction)
                : 0,
            };
          }),
        };
      });
      setSelectionHistory(function (previous) {
        return previous.concat(events);
      });
    },
    [onResult, predicateMode],
  );

  const handleFactorToggle = (column) => {
    if (column === targetColumn) {
      return;
    }

    const isSelected = factorColumns.includes(column);
    setFactorColumns(function (previous) {
      if (previous.includes(column)) {
        return previous.filter(function (item) {
          return item !== column;
        });
      }
      return previous.concat(column);
    });
    if (!isSelected) {
      setObjectiveColumns(function (previous) {
        return previous.filter(function (item) {
          return item !== column;
        });
      });
    }
    setNeedsProjection(true);
    onProjectionChange(null);
    onResult(null);
  };

  const handleObjectiveToggle = (column) => {
    if (factorColumns.includes(column)) {
      return;
    }

    setObjectiveColumns(function (previous) {
      if (previous.includes(column)) {
        return previous.filter(function (item) {
          return item !== column;
        });
      }
      return previous.concat(column);
    });
    setNeedsProjection(true);
    setObjectiveFilters({});
    setObjectiveActiveSegments({});
    setObjectiveDirections(function (previous) {
      const next = {...previous};
      if (next[column] === undefined) {
        next[column] = "range";
      }
      return next;
    });
    onProjectionChange(null);
    onResult(null);
  };

  const handleTargetChange = (event) => {
    const nextTarget = event.target.value;
    setTargetColumn(nextTarget);
    setFactorColumns(function (previous) {
      if (!nextTarget) {
        return previous;
      }
      return previous.filter(function (column) {
        return column !== nextTarget;
      });
    });
    setNeedsProjection(true);
    setProjectionError("");
    onProjectionChange(null);
    onResult(null);
  };

  const handleProjectionConfigChange = () => {
    setNeedsProjection(true);
    setProjectionError("");
    setObjectiveFilters({});
    setObjectiveDirections({});
    setObjectiveActiveSegments({});
    onProjectionChange(null);
    onResult(null);
  };

  const canRunProjection =
    Boolean(dataset) &&
    factorColumns.length > 0 &&
    !(projectionMode === "supervised-umap" && !targetColumn);

  const projectionSummary = useMemo(
    function () {
      if (!dataset) {
        return "";
      }
      if (!factorColumns.length) {
        return "Pick at least one numeric factor column before generating the DimBridge view.";
      }
      if (projectionMode === "supervised-umap" && !targetColumn) {
        return "Supervised UMAP needs a target column.";
      }
      return [
        formatProjectionMode(projectionMode),
        "Factors: " + factorColumns.join(", "),
        objectiveColumns.length ? "Objectives: " + objectiveColumns.join(", ") : "Objectives: none",
        targetColumn ? "Target / Color: " + targetColumn : "Target / Color: none",
      ].join(" | ");
    },
    [dataset, factorColumns, objectiveColumns, projectionMode, targetColumn],
  );

  const objectiveExtents = useMemo(
    function () {
      return buildObjectiveExtents(analysis);
    },
    [analysis],
  );

  const activeObjectiveColumns = analysis ? analysis.objective_columns || [] : [];
  const objectiveSelection = useMemo(
    function () {
      if (!analysis || !activeObjectiveColumns.length) {
        return {isActive: false, mask: null, selectedCount: 0, totalCount: 0};
      }

      const records = analysis.records || [];
      const isActive = activeObjectiveColumns.some(function (column) {
        const extent = objectiveExtents[column];
        if (!extent) {
          return false;
        }
        const range = objectiveFilters[column] || {min: extent.min, max: extent.max};
        const epsilon = Math.max(Math.abs(extent.max - extent.min) * 1e-9, 1e-12);
        return range.min > extent.min + epsilon || range.max < extent.max - epsilon;
      });

      if (!isActive) {
        return {
          isActive: false,
          mask: null,
          selectedCount: records.length,
          totalCount: records.length,
        };
      }

      const mask = records.map(function (record) {
        return activeObjectiveColumns.every(function (column) {
          const extent = objectiveExtents[column];
          if (!extent) {
            return false;
          }
          const range = objectiveFilters[column] || {min: extent.min, max: extent.max};
          const value = Number(record[column]);
          if (!Number.isFinite(value) || value < range.min || value > range.max) {
            return false;
          }
          const activeSegment = objectiveActiveSegments[column];
          return !activeSegment ||
            activeSegment.index === activeSegment.count - 1 ||
            value < range.max;
        });
      });

      return {
        isActive: true,
        mask,
        selectedCount: mask.filter(Boolean).length,
        totalCount: records.length,
      };
    },
    [activeObjectiveColumns, analysis, objectiveActiveSegments, objectiveExtents, objectiveFilters],
  );

  const objectiveFilterCount = objectiveSelection.selectedCount;
  const objectiveTotalCount = analysis ? (analysis.records || []).length : 0;

  const setObjectiveFilterRange = (column, bound, value) => {
    const extent = objectiveExtents[column];
    if (!extent) {
      return;
    }

    const numericValue = Number(value);
    if (!Number.isFinite(numericValue)) {
      return;
    }

    setObjectiveFilters(function (previous) {
      const current = previous[column] || {min: extent.min, max: extent.max};
      const next = {
        min: bound === "min" ? Math.min(numericValue, current.max) : current.min,
        max: bound === "max" ? Math.max(numericValue, current.min) : current.max,
      };
      return {...previous, [column]: next};
    });
    setObjectiveSelectionSource("objective-slider");
    setObjectiveActiveSegments(function (previous) {
      const next = {...previous};
      delete next[column];
      return next;
    });
    onResult(null);
  };

  const setObjectiveDirection = (column, direction) => {
    setObjectiveDirections(function (previous) {
      return {...previous, [column]: direction};
    });
  };

  const applyObjectiveShortcut = (column, shortcut) => {
    const extent = objectiveExtents[column];
    if (!extent) {
      return;
    }

    let nextRange = {min: extent.min, max: extent.max};
    if (shortcut === "top20") {
      const threshold = quantile(extent.values || [], 0.8);
      if (threshold === null) {
        return;
      }
      nextRange = {min: threshold, max: extent.max};
    } else if (shortcut === "top10") {
      const threshold = quantile(extent.values || [], 0.9);
      if (threshold === null) {
        return;
      }
      nextRange = {min: threshold, max: extent.max};
    } else if (shortcut === "bottom20") {
      const threshold = quantile(extent.values || [], 0.2);
      if (threshold === null) {
        return;
      }
      nextRange = {min: extent.min, max: threshold};
    } else if (shortcut === "bottom10") {
      const threshold = quantile(extent.values || [], 0.1);
      if (threshold === null) {
        return;
      }
      nextRange = {min: extent.min, max: threshold};
    }

    setObjectiveFilters(function (previous) {
      return {...previous, [column]: nextRange};
    });
    setObjectiveSelectionSource("objective-shortcut");
    setObjectiveActiveSegments(function (previous) {
      const next = {...previous};
      delete next[column];
      return next;
    });
    onResult(null);
  };

  const applyObjectiveSegment = (column, count, index) => {
    const extent = objectiveExtents[column];
    if (!extent || extent.min === extent.max) {
      return;
    }
    const segmentWidth = (extent.max - extent.min) / count;
    const min = extent.min + segmentWidth * index;
    const max = index === count - 1 ? extent.max : extent.min + segmentWidth * (index + 1);
    setObjectiveDirections(function (previous) {
      return {...previous, [column]: "range"};
    });
    setObjectiveFilters(function (previous) {
      return {...previous, [column]: {min, max}};
    });
    setObjectiveSelectionSource(`equal-segment-${count}`);
    setObjectiveActiveSegments(function (previous) {
      return {...previous, [column]: {count, index}};
    });
    onResult(null);
  };

  const resetObjectiveFilters = () => {
    setObjectiveFilters({});
    setObjectiveDirections({});
    setObjectiveActiveSegments({});
    onResult(null);
  };

  const submitProjection = async () => {
    if (!dataset) {
      return;
    }
    if (!factorColumns.length) {
      setProjectionError("Select at least one numeric factor column.");
      return;
    }
    if (projectionMode === "supervised-umap" && !targetColumn) {
      setProjectionError("Supervised UMAP requires a target column.");
      return;
    }

    setProjectionBusy(true);
    setProjectionError("");
    setError("");
    setIsLoading(false);
    onResult(null);

    try {
      const requestConfig = projectionRequestConfig(projectionMode);
      const projection = await runProjection({
        dataset_id: dataset.dataset_id,
        factor_columns: factorColumns,
        objective_columns: objectiveColumns,
        target_column: targetColumn || null,
        method: requestConfig.method,
        supervision: requestConfig.supervision,
        standardize,
        n_neighbors: nNeighbors,
        min_dist: minDist,
        metric: "euclidean",
        target_weight: targetWeight,
        perplexity,
        learning_rate: learningRate,
        max_iterations: maxIterations,
      });
      onProjectionChange(projection);
      setObjectiveFilters({});
      setNeedsProjection(false);
      setResetToken(function (previous) {
        return previous + 1;
      });
    } catch (requestError) {
      setProjectionError(requestError.message);
      onProjectionChange(null);
    } finally {
      setProjectionBusy(false);
    }
  };

  return (
    <section className="panel workspace-panel">
      <div className="panel-header">
        <h2>Analysis Workspace</h2>
        <span className={isLoading ? "badge loading-badge" : "badge"}>
          {isLoading ? "Computing" : "Step 2"}
        </span>
      </div>
      <p className="muted">
        This page now follows the notebook workflow: choose factor and target columns, generate
        a dimensionality reduction projection, then mount the original DimBridge interaction views on top of that
        projected dataset.
      </p>

      <div className="projection-config">
        <div className="projection-config-panel">
          <label>
            <span>Projection Mode</span>
            <select
              onChange={function (event) {
                setProjectionMode(event.target.value);
                handleProjectionConfigChange();
              }}
              value={projectionMode}
            >
              <option value="pca">PCA</option>
              <option value="tsne">t-SNE</option>
              <option value="umap">UMAP</option>
              <option value="supervised-umap">Supervised UMAP</option>
            </select>
          </label>
          <label>
            <span>Target Column</span>
            <select disabled={!dataset} onChange={handleTargetChange} value={targetColumn}>
              <option value="">None</option>
              {targetOptions.map(function (column) {
                return (
                  <option key={column} value={column}>
                    {column}
                  </option>
                );
              })}
            </select>
          </label>

          <label>
            <span>Predicate Mode</span>
            <select
              onChange={function (event) {
                setPredicateMode(event.target.value);
              }}
              value={predicateMode}
            >
              <option value="data-extent">Data Extent</option>
              <option value="regression">Predicate Regression</option>
            </select>
          </label>

          <label>
            <span>Predicate View Factors</span>
            <input
              max="100"
              min="1"
              onChange={function (event) {
                const nextValue = Number(event.target.value);
                setPredicateFactorLimit(Math.max(1, Math.min(100, nextValue || 1)));
              }}
              step="1"
              type="number"
              value={predicateFactorLimit}
            />
          </label>

          <label>
            <span>Brush Mode</span>
            <select
              onChange={function (event) {
                setBrushMode(event.target.value);
              }}
              value={brushMode}
            >
              <option value="single">Single</option>
              <option value="contrastive">Contrastive</option>
              <option value="curve">Curve</option>
            </select>
          </label>

          {projectionMode === "tsne" ? (
            <>
              <label>
                <span>perplexity</span>
                <input
                  min="1"
                  onChange={function (event) {
                    setPerplexity(Number(event.target.value));
                    handleProjectionConfigChange();
                  }}
                  step="1"
                  type="number"
                  value={perplexity}
                />
              </label>

              <label>
                <span>learning_rate</span>
                <input
                  min="1"
                  onChange={function (event) {
                    setLearningRate(Number(event.target.value));
                    handleProjectionConfigChange();
                  }}
                  step="10"
                  type="number"
                  value={learningRate}
                />
              </label>

              <label>
                <span>max_iterations</span>
                <input
                  min="250"
                  onChange={function (event) {
                    setMaxIterations(Number(event.target.value));
                    handleProjectionConfigChange();
                  }}
                  step="50"
                  type="number"
                  value={maxIterations}
                />
              </label>
            </>
          ) : null}

          {projectionMode === "umap" || projectionMode === "supervised-umap" ? (
            <>
              <label>
                <span>n_neighbors</span>
                <input
                  min="2"
                  onChange={function (event) {
                    setNNeighbors(Number(event.target.value));
                    handleProjectionConfigChange();
                  }}
                  type="number"
                  value={nNeighbors}
                />
              </label>

              <label>
                <span>min_dist</span>
                <input
                  max="1"
                  min="0"
                  onChange={function (event) {
                    setMinDist(Number(event.target.value));
                    handleProjectionConfigChange();
                  }}
                  step="0.05"
                  type="number"
                  value={minDist}
                />
              </label>

              {projectionMode === "supervised-umap" ? (
                <label>
                  <span>target_weight</span>
                  <input
                    max="1"
                    min="0"
                    onChange={function (event) {
                      setTargetWeight(Number(event.target.value));
                      handleProjectionConfigChange();
                    }}
                    step="0.05"
                    type="number"
                    value={targetWeight}
                  />
                </label>
              ) : null}
            </>
          ) : null}

          <div className="projection-config-item">
            <span>Standardize</span>
            <button
              aria-pressed={standardize}
              className={standardize ? "toggle-button checked" : "toggle-button"}
              onClick={function () {
                setStandardize(function (previous) {
                  return !previous;
                });
                handleProjectionConfigChange();
              }}
              type="button"
            >
              {standardize ? "On" : "Off"}
            </button>
          </div>
        </div>

        <div className="button-row projection-actions-row">
          <button disabled={!canRunProjection || projectionBusy} onClick={submitProjection} type="button">
            {projectionBusy ? "Generating..." : "Generate Projection"}
          </button>
          <button
            className="secondary-button"
            onClick={function () {
              setResetToken(function (previous) {
                return previous + 1;
              });
              setError("");
              setIsLoading(false);
              onResult(null);
            }}
            type="button"
          >
            Reset Widget
          </button>
        </div>
      </div>

      <div className="factor-picker">
        <div className="factor-picker-header">
          <strong>Factor Columns</strong>
          <span className="muted">{factorColumns.length} selected</span>
        </div>
        <div className="checkbox-grid">
          {numericColumns.map(function (column) {
            const checked = factorColumns.includes(column);
            const disabled = column === targetColumn;
            return (
              <label
                className={checked ? "checkbox-pill checked" : "checkbox-pill"}
                key={column}
              >
                <input
                  checked={checked}
                  disabled={disabled}
                  onChange={function () {
                    handleFactorToggle(column);
                  }}
                  type="checkbox"
                />
                <span>{column}</span>
              </label>
            );
          })}
        </div>
      </div>

      <div className="factor-picker objective-picker">
        <div className="factor-picker-header">
          <strong>Performance Metrics</strong>
          <span className="muted">{objectiveColumns.length} selected</span>
        </div>
        <div className="checkbox-grid">
          {numericColumns.map(function (column) {
            const checked = objectiveColumns.includes(column);
            const disabled = factorColumns.includes(column);
            return (
              <label
                className={checked ? "checkbox-pill checked" : "checkbox-pill"}
                key={column}
              >
                <input
                  checked={checked}
                  disabled={disabled}
                  onChange={function () {
                    handleObjectiveToggle(column);
                  }}
                  type="checkbox"
                />
                <span>{column}</span>
              </label>
            );
          })}
        </div>
      </div>

      <p className="muted">{projectionSummary}</p>

      <div className="legacy-widget-panel">
        {isLoading ? (
          <div className="info-banner">
            Quick extent preview is shown immediately. Predicate regression is still running on
            the backend.
          </div>
        ) : null}
        <LegacyDimbridgeWidget
          analysis={analysis}
          brushMode={brushMode}
          focusedSplomAttributes={focusedCorrelationAttributes}
          objectiveSelectionMask={objectiveSelection.mask}
          objectiveSelectionSource={objectiveSelectionSource}
          onError={function (requestError) {
            setError(requestError.message);
            setIsLoading(false);
          }}
          onLoadingChange={setIsLoading}
          onMetaChange={setWidgetMeta}
          onPredicates={handlePredicateResult}
          predicateFactorLimit={predicateFactorLimit}
          onSelectionChange={setCurrentSelectionMask}
          predicateMode={predicateMode}
          resetToken={resetToken}
        />
      </div>

      <FactorOccurrenceStats
        history={selectionHistory}
        onExport={function () {
          downloadFactorReport({analysis, dataset, history: selectionHistory, sessionStartedAt});
        }}
        onReset={function () {
          setSelectionHistory([]);
          setSessionStartedAt(new Date().toISOString());
        }}
      />

      {analysis && activeObjectiveColumns.length ? (
        <div className="objective-filter-panel">
          <div className="factor-picker-header">
            <strong>Multi-objective Constraints</strong>
            <span className="muted">
              {objectiveSelection.isActive
                ? `${objectiveFilterCount} / ${objectiveTotalCount} selected`
                : "No active constraints"}
            </span>
          </div>
          <div className="objective-filter-grid">
            {activeObjectiveColumns.map(function (column) {
              const extent = objectiveExtents[column];
              if (!extent) {
                return (
                  <div className="objective-filter-item" key={column}>
                    <div className="objective-filter-heading">
                      <strong>{column}</strong>
                      <span>No numeric values</span>
                    </div>
                  </div>
                );
              }

              const range = objectiveFilters[column] || {min: extent.min, max: extent.max};
              const direction = objectiveDirections[column] || "range";
              const disabled = extent.min === extent.max;
              const span = extent.max - extent.min || 1;
              const activeStart = ((range.min - extent.min) / span) * 100;
              const activeEnd = ((range.max - extent.min) / span) * 100;
              const segmentCount = objectiveSegmentCounts[column] || 0;
              const activeSegment = objectiveActiveSegments[column];
              return (
                <div className="objective-filter-item" key={column}>
                  <div className="objective-filter-heading">
                    <strong>{column}</strong>
                    <span>
                      {formatMetricValue(range.min)} to {formatMetricValue(range.max)}
                    </span>
                  </div>
                  <div className="objective-direction-row">
                    <button
                      className={direction === "higher" ? "mini-toggle active" : "mini-toggle"}
                      onClick={function () {
                        setObjectiveDirection(column, "higher");
                      }}
                      type="button"
                    >
                      Higher
                    </button>
                    <button
                      className={direction === "lower" ? "mini-toggle active" : "mini-toggle"}
                      onClick={function () {
                        setObjectiveDirection(column, "lower");
                      }}
                      type="button"
                    >
                      Lower
                    </button>
                    <button
                      className={direction === "range" ? "mini-toggle active" : "mini-toggle"}
                      onClick={function () {
                        setObjectiveDirection(column, "range");
                        applyObjectiveShortcut(column, "range");
                      }}
                      type="button"
                    >
                      Range
                    </button>
                  </div>
                  {direction !== "range" ? (
                    <div className="objective-shortcut-row">
                      {direction === "higher" ? (
                        <>
                          <button
                            className="mini-button"
                            disabled={disabled}
                            onClick={function () {
                              applyObjectiveShortcut(column, "top20");
                            }}
                            type="button"
                          >
                            Top 20%
                          </button>
                          <button
                            className="mini-button"
                            disabled={disabled}
                            onClick={function () {
                              applyObjectiveShortcut(column, "top10");
                            }}
                            type="button"
                          >
                            Top 10%
                          </button>
                        </>
                      ) : (
                        <>
                          <button
                            className="mini-button"
                            disabled={disabled}
                            onClick={function () {
                              applyObjectiveShortcut(column, "bottom20");
                            }}
                            type="button"
                          >
                            Bottom 20%
                          </button>
                          <button
                            className="mini-button"
                            disabled={disabled}
                            onClick={function () {
                              applyObjectiveShortcut(column, "bottom10");
                            }}
                            type="button"
                          >
                            Bottom 10%
                          </button>
                        </>
                      )}
                    </div>
                  ) : null}
                  <div className="objective-segment-mode-row">
                    <span>Equal intervals</span>
                    {[5, 10].map(function (count) {
                      return (
                        <button
                          className={segmentCount === count ? "mini-toggle active" : "mini-toggle"}
                          disabled={disabled}
                          key={count}
                          onClick={function () {
                            setObjectiveSegmentCounts(function (previous) {
                              return {...previous, [column]: count};
                            });
                          }}
                          type="button"
                        >
                          {count} Segments
                        </button>
                      );
                    })}
                  </div>
                  {segmentCount > 0 ? (
                    <div
                      className="objective-segment-grid"
                      style={{gridTemplateColumns: `repeat(${segmentCount}, minmax(0, 1fr))`}}
                    >
                      {Array.from({length: segmentCount}, function (_, index) {
                        const segmentWidth = (extent.max - extent.min) / segmentCount;
                        const segmentMin = extent.min + segmentWidth * index;
                        const segmentMax =
                          index === segmentCount - 1
                            ? extent.max
                            : extent.min + segmentWidth * (index + 1);
                        const isActive =
                          activeSegment?.count === segmentCount && activeSegment?.index === index;
                        return (
                          <button
                            aria-label={`${column}: ${formatMetricValue(segmentMin)} to ${formatMetricValue(segmentMax)}`}
                            className={isActive ? "objective-segment active" : "objective-segment"}
                            disabled={disabled}
                            key={index}
                            onClick={function () {
                              applyObjectiveSegment(column, segmentCount, index);
                            }}
                            title={`${formatMetricValue(segmentMin)} to ${formatMetricValue(segmentMax)}`}
                            type="button"
                          >
                            <span>{index + 1}</span>
                            <small>{formatMetricValue(segmentMin)}</small>
                            <small>{formatMetricValue(segmentMax)}</small>
                          </button>
                        );
                      })}
                    </div>
                  ) : null}
                  <div className="objective-double-slider">
                    <div className="objective-slider-track" />
                    <div
                      className="objective-slider-active"
                      style={{
                        left: Math.max(0, Math.min(100, activeStart)) + "%",
                        right: 100 - Math.max(0, Math.min(100, activeEnd)) + "%",
                      }}
                    />
                    <input
                      aria-label={column + " minimum"}
                      disabled={disabled}
                      max={extent.max}
                      min={extent.min}
                      onChange={function (event) {
                        setObjectiveFilterRange(column, "min", event.target.value);
                      }}
                      step={extent.step}
                      type="range"
                      value={range.min}
                    />
                    <input
                      aria-label={column + " maximum"}
                      disabled={disabled}
                      max={extent.max}
                      min={extent.min}
                      onChange={function (event) {
                        setObjectiveFilterRange(column, "max", event.target.value);
                      }}
                      step={extent.step}
                      type="range"
                      value={range.max}
                    />
                  </div>
                </div>
              );
            })}
          </div>
          <div className="button-row objective-filter-actions">
            <button className="secondary-button" onClick={resetObjectiveFilters} type="button">
              Reset Constraints
            </button>
          </div>
        </div>
      ) : null}

      {analysis ? (
        <CorrelationHeatmap
          analysis={analysis}
          autoSelectedColumns={correlationPredicateSelection.columns}
          autoSelectionVersion={correlationPredicateSelection.version}
          onSelectedPairsChange={setCorrelationPairs}
          selectionMask={currentSelectionMask}
          selectedPairs={correlationPairs}
        />
      ) : null}

      <div className="demo-footer">
        <span className="muted">
          {projectionBusy
            ? "Projection is being generated on the backend."
            : isLoading
              ? "Predicate regression is running for the current brush selection. The quick preview above will be replaced when the backend returns."
              : "After the projection is ready, the predicate response panel updates automatically."}
        </span>
        {needsProjection ? <span className="warning-text">Projection settings changed. Regenerate to refresh the view.</span> : null}
      </div>

      {widgetMeta.hasRenderableData ? (
        <p className="muted">
          Rendered rows: {widgetMeta.validRows}. Dropped during projection preprocessing: {widgetMeta.droppedRows}.
        </p>
      ) : null}
      {dataset && !analysis && !projectionBusy && !projectionError ? (
        <p className="muted">Generate a projection to render the standalone DimBridge view.</p>
      ) : null}
      {projectionError ? <p className="error-text">{projectionError}</p> : null}
      {error ? <p className="error-text">{error}</p> : null}
      {!dataset ? (
        <p className="muted">Upload a dataset to enable projection configuration and DimBridge rendering.</p>
      ) : null}
    </section>
  );
}
