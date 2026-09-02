import {lazy, Suspense, useEffect, useMemo, useRef, useState} from "react";

import CorrelationNetwork from "./CorrelationNetwork";
import {buildCorrelationGraph, sameCorrelationPair} from "./correlationGraph";

const CorrelationSphere = lazy(function () {
  return import("./CorrelationSphere");
});

const METHOD_OPTIONS = [
  {value: "pearson", label: "Pearson (linear)"},
  {value: "spearman", label: "Spearman (rank)"},
  {value: "mutual-information", label: "Mutual Information (normalized)"},
];

function numericValue(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function pairedValues(records, firstColumn, secondColumn) {
  const first = [];
  const second = [];

  records.forEach(function (record) {
    const firstValue = numericValue(record[firstColumn]);
    const secondValue = numericValue(record[secondColumn]);
    if (firstValue !== null && secondValue !== null) {
      first.push(firstValue);
      second.push(secondValue);
    }
  });

  return {first, second};
}

function pearsonCorrelation(first, second) {
  if (first.length < 3 || first.length !== second.length) {
    return null;
  }

  const count = first.length;
  let firstMean = 0;
  let secondMean = 0;
  for (let index = 0; index < count; index += 1) {
    firstMean += first[index];
    secondMean += second[index];
  }
  firstMean /= count;
  secondMean /= count;

  let numerator = 0;
  let firstVariance = 0;
  let secondVariance = 0;
  for (let index = 0; index < count; index += 1) {
    const firstDelta = first[index] - firstMean;
    const secondDelta = second[index] - secondMean;
    numerator += firstDelta * secondDelta;
    firstVariance += firstDelta * firstDelta;
    secondVariance += secondDelta * secondDelta;
  }

  const denominator = Math.sqrt(firstVariance * secondVariance);
  if (denominator === 0) {
    return null;
  }
  return Math.max(-1, Math.min(1, numerator / denominator));
}

function averageRanks(values) {
  const sorted = values
    .map(function (value, index) {
      return {value, index};
    })
    .sort(function (first, second) {
      return first.value - second.value;
    });
  const ranks = new Array(values.length);

  let start = 0;
  while (start < sorted.length) {
    let end = start + 1;
    while (end < sorted.length && sorted[end].value === sorted[start].value) {
      end += 1;
    }
    const averageRank = (start + end - 1) / 2;
    for (let index = start; index < end; index += 1) {
      ranks[sorted[index].index] = averageRank;
    }
    start = end;
  }

  return ranks;
}

function rankBins(values, binCount) {
  const ranks = averageRanks(values);
  return ranks.map(function (rank) {
    return Math.min(binCount - 1, Math.floor((rank / values.length) * binCount));
  });
}

function normalizedMutualInformation(first, second) {
  if (first.length < 5 || first.length !== second.length) {
    return null;
  }

  const uniqueFirst = new Set(first).size;
  const uniqueSecond = new Set(second).size;
  if (uniqueFirst < 2 || uniqueSecond < 2) {
    return null;
  }

  const count = first.length;
  const binCount = Math.min(10, Math.max(2, Math.floor(Math.sqrt(count / 2))));
  const firstBins = rankBins(first, binCount);
  const secondBins = rankBins(second, binCount);
  const firstCounts = new Array(binCount).fill(0);
  const secondCounts = new Array(binCount).fill(0);
  const jointCounts = Array.from({length: binCount}, function () {
    return new Array(binCount).fill(0);
  });

  for (let index = 0; index < count; index += 1) {
    const firstBin = firstBins[index];
    const secondBin = secondBins[index];
    firstCounts[firstBin] += 1;
    secondCounts[secondBin] += 1;
    jointCounts[firstBin][secondBin] += 1;
  }

  let mutualInformation = 0;
  let firstEntropy = 0;
  let secondEntropy = 0;
  for (let firstBin = 0; firstBin < binCount; firstBin += 1) {
    const firstProbability = firstCounts[firstBin] / count;
    if (firstProbability > 0) {
      firstEntropy -= firstProbability * Math.log(firstProbability);
    }
    const secondProbability = secondCounts[firstBin] / count;
    if (secondProbability > 0) {
      secondEntropy -= secondProbability * Math.log(secondProbability);
    }

    for (let secondBin = 0; secondBin < binCount; secondBin += 1) {
      const jointProbability = jointCounts[firstBin][secondBin] / count;
      if (jointProbability === 0) {
        continue;
      }
      mutualInformation +=
        jointProbability *
        Math.log(
          jointProbability /
            ((firstCounts[firstBin] / count) * (secondCounts[secondBin] / count)),
        );
    }
  }

  const denominator = Math.sqrt(firstEntropy * secondEntropy);
  if (denominator === 0) {
    return null;
  }
  return Math.max(0, Math.min(1, mutualInformation / denominator));
}

function calculateCorrelation(records, firstColumn, secondColumn, method) {
  const values = pairedValues(records, firstColumn, secondColumn);
  const count = values.first.length;
  if (firstColumn === secondColumn) {
    return {value: count >= 3 ? 1 : null, count};
  }

  let value;
  if (method === "spearman") {
    value = pearsonCorrelation(averageRanks(values.first), averageRanks(values.second));
  } else if (method === "mutual-information") {
    value = normalizedMutualInformation(values.first, values.second);
  } else {
    value = pearsonCorrelation(values.first, values.second);
  }
  return {value, count};
}

function mixColor(start, end, amount) {
  const bounded = Math.max(0, Math.min(1, amount));
  const channels = start.map(function (channel, index) {
    return Math.round(channel + (end[index] - channel) * bounded);
  });
  return `rgb(${channels.join(", ")})`;
}

function cellColors(value, method) {
  if (!Number.isFinite(value)) {
    return {background: "#eef2f4", color: "#87939c"};
  }

  if (method === "mutual-information") {
    const amount = Math.pow(value, 0.72);
    return {
      background: mixColor([239, 246, 243], [14, 116, 101], amount),
      color: amount > 0.58 ? "#ffffff" : "#173b35",
    };
  }

  const amount = Math.pow(Math.abs(value), 0.72);
  const endpoint = value < 0 ? [184, 64, 63] : [30, 103, 151];
  return {
    background: mixColor([247, 247, 244], endpoint, amount),
    color: amount > 0.62 ? "#ffffff" : "#25313a",
  };
}

function formatCorrelation(value) {
  if (!Number.isFinite(value)) {
    return "-";
  }
  return value.toFixed(2);
}

export default function CorrelationHeatmap({
  analysis,
  autoSelectedColumns = [],
  autoSelectionVersion = 0,
  selectionMask,
  selectedPairs = [],
  onSelectedPairsChange,
}) {
  const factorColumns = analysis?.factor_columns || [];
  const objectiveColumns = analysis?.objective_columns || [];
  const availableColumns = useMemo(
    function () {
      return Array.from(new Set(factorColumns.concat(objectiveColumns)));
    },
    [factorColumns, objectiveColumns],
  );
  const columnSignature = availableColumns.join("\u0000");
  const autoSelectionSignature = autoSelectedColumns.join("\u0000");
  const [method, setMethod] = useState("pearson");
  const [enabledColumns, setEnabledColumns] = useState([]);
  const [viewMode, setViewMode] = useState("heatmap");
  const [edgeThreshold, setEdgeThreshold] = useState(0.3);
  const [networkGroups, setNetworkGroups] = useState([]);
  const [networkGroupHistory, setNetworkGroupHistory] = useState([]);
  const nextGroupIdRef = useRef(1);

  const selectedAttributeCount = useMemo(
    function () {
      return new Set(
        selectedPairs.flatMap(function (pair) {
          return pair;
        }),
      ).size;
    },
    [selectedPairs],
  );

  useEffect(
    function () {
      setEnabledColumns([]);
      setNetworkGroups([]);
      setNetworkGroupHistory([]);
      nextGroupIdRef.current = 1;
    },
    [analysis?.dataset_id, columnSignature],
  );

  useEffect(
    function () {
      const requestedColumns = new Set(autoSelectedColumns);
      setEnabledColumns(
        availableColumns.filter(function (column) {
          return requestedColumns.has(column);
        }),
      );
      setNetworkGroups([]);
      setNetworkGroupHistory([]);
      nextGroupIdRef.current = 1;
    },
    [autoSelectionSignature, autoSelectionVersion, columnSignature],
  );

  const visibleColumns = useMemo(
    function () {
      return availableColumns.filter(function (column) {
        return enabledColumns.includes(column);
      });
    },
    [availableColumns, enabledColumns],
  );
  const records = analysis?.records || [];
  const hasSelection = Array.isArray(selectionMask);
  const scopedRecords = useMemo(
    function () {
      if (!hasSelection) {
        return records;
      }
      return records.filter(function (_, index) {
        return Boolean(selectionMask[index]);
      });
    },
    [hasSelection, records, selectionMask],
  );

  const matrix = useMemo(
    function () {
      const result = visibleColumns.map(function () {
        return new Array(visibleColumns.length);
      });
      for (let rowIndex = 0; rowIndex < visibleColumns.length; rowIndex += 1) {
        for (let columnIndex = rowIndex; columnIndex < visibleColumns.length; columnIndex += 1) {
          const value = calculateCorrelation(
            scopedRecords,
            visibleColumns[rowIndex],
            visibleColumns[columnIndex],
            method,
          );
          result[rowIndex][columnIndex] = value;
          result[columnIndex][rowIndex] = value;
        }
      }
      return result;
    },
    [method, scopedRecords, visibleColumns],
  );
  const correlationGraph = useMemo(
    function () {
      return buildCorrelationGraph({
        columns: visibleColumns,
        objectiveColumns,
        matrix,
        threshold: edgeThreshold,
        groups: networkGroups,
      });
    },
    [edgeThreshold, matrix, networkGroups, objectiveColumns, visibleColumns],
  );

  const toggleColumn = (column) => {
    if (enabledColumns.includes(column) && typeof onSelectedPairsChange === "function") {
      onSelectedPairsChange(function (previous) {
        return previous.filter(function (pair) {
          return !pair.includes(column);
        });
      });
    }
    setEnabledColumns(function (previous) {
      if (previous.includes(column)) {
        return previous.filter(function (item) {
          return item !== column;
        });
      }
      return previous.concat(column);
    });
  };

  const toggleSelectedPair = (pair) => {
    if (!Array.isArray(pair) || pair.length < 2 || typeof onSelectedPairsChange !== "function") {
      return;
    }
    onSelectedPairsChange(function (previous) {
      const isSelected = previous.some(function (selectedPair) {
        return sameCorrelationPair(selectedPair, pair);
      });
      if (isSelected) {
        return previous.filter(function (selectedPair) {
          return !sameCorrelationPair(selectedPair, pair);
        });
      }
      return previous.concat([[pair[0], pair[1]]]);
    });
  };

  const createNetworkGroup = (members) => {
    const orderedMembers = availableColumns.filter(function (column) {
      return members.includes(column);
    });
    if (orderedMembers.length < 2) {
      return null;
    }
    const groupNumber = nextGroupIdRef.current;
    nextGroupIdRef.current += 1;
    const group = {
      id: "correlation-group-" + groupNumber,
      label: "Group " + groupNumber,
      members: orderedMembers,
    };
    setNetworkGroupHistory(function (previous) {
      return previous.concat([
        networkGroups.map(function (item) {
          return {...item, members: item.members.slice()};
        }),
      ]);
    });
    setNetworkGroups(networkGroups.concat(group));
    return group.id;
  };

  const mergeNetworkNodes = (sourceNode, targetNode) => {
    if (!sourceNode || !targetNode || sourceNode.id === targetNode.id) {
      return null;
    }
    if (sourceNode.kind === "group" && targetNode.kind === "group") {
      return null;
    }

    const sourceGroup =
      sourceNode.kind === "group"
        ? networkGroups.find(function (group) {
            return group.id === sourceNode.id;
          })
        : null;
    const targetGroup =
      targetNode.kind === "group"
        ? networkGroups.find(function (group) {
            return group.id === targetNode.id;
          })
        : null;
    const members = new Set(
      (sourceGroup?.members || sourceNode.members).concat(
        targetGroup?.members || targetNode.members,
      ),
    );
    const orderedMembers = availableColumns.filter(function (column) {
      return members.has(column);
    });
    if (orderedMembers.length < 2) {
      return null;
    }

    const existingGroup = sourceGroup || targetGroup;
    if (existingGroup) {
      setNetworkGroupHistory(function (previous) {
        return previous.concat([
          networkGroups.map(function (item) {
            return {...item, members: item.members.slice()};
          }),
        ]);
      });
      setNetworkGroups(
        networkGroups.map(function (group) {
          return group.id === existingGroup.id
            ? {...group, members: orderedMembers}
            : group;
        }),
      );
      return existingGroup.id;
    }
    return createNetworkGroup(orderedMembers);
  };

  const expandNetworkGroup = (groupId) => {
    setNetworkGroups(function (previous) {
      return previous.filter(function (group) {
        return group.id !== groupId;
      });
    });
    setNetworkGroupHistory([]);
  };

  const undoNetworkMerge = () => {
    if (networkGroupHistory.length === 0) {
      return;
    }
    const previousGroups = networkGroupHistory[networkGroupHistory.length - 1];
    setNetworkGroups(
      previousGroups.map(function (group) {
        return {...group, members: group.members.slice()};
      }),
    );
    setNetworkGroupHistory(networkGroupHistory.slice(0, -1));
  };

  return (
    <div className="correlation-panel">
      <div className="correlation-header">
        <div>
          <strong>Correlation Analysis</strong>
          <span className="correlation-scope">
            {hasSelection ? "Current selection" : "Global"}: {scopedRecords.length} / {records.length}
          </span>
        </div>
        <div className="correlation-header-controls">
          <div className="correlation-view-toggle" aria-label="Correlation view">
            <button
              aria-pressed={viewMode === "heatmap"}
              className={viewMode === "heatmap" ? "mini-toggle active" : "mini-toggle"}
              onClick={function () {
                setViewMode("heatmap");
              }}
              type="button"
            >
              Heatmap
            </button>
            <button
              aria-pressed={viewMode === "network"}
              className={viewMode === "network" ? "mini-toggle active" : "mini-toggle"}
              onClick={function () {
                setViewMode("network");
              }}
              type="button"
            >
              Network
            </button>
            <button
              aria-pressed={viewMode === "sphere"}
              className={viewMode === "sphere" ? "mini-toggle active" : "mini-toggle"}
              onClick={function () {
                setViewMode("sphere");
              }}
              type="button"
            >
              Sphere 3D
            </button>
          </div>
          <label className="correlation-method">
            <span>Calculation Method</span>
            <select
              onChange={function (event) {
                setMethod(event.target.value);
              }}
              value={method}
            >
              {METHOD_OPTIONS.map(function (option) {
                return (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                );
              })}
            </select>
          </label>
        </div>
      </div>

      <div className="correlation-variable-row">
        <div className="correlation-variable-heading">
          <span>Variables</span>
          <div className="correlation-variable-actions">
            <button
              className="mini-button"
              onClick={function () {
                setEnabledColumns(availableColumns);
              }}
              type="button"
            >
              Select All
            </button>
            <button
              className="mini-button"
              disabled={selectedPairs.length === 0}
              onClick={function () {
                onSelectedPairsChange?.([]);
              }}
              type="button"
            >
              Clear Selection
            </button>
          </div>
        </div>
        <div className="correlation-variable-list">
          {availableColumns.map(function (column) {
            const checked = enabledColumns.includes(column);
            const isObjective = objectiveColumns.includes(column);
            return (
              <label
                className={[
                  "correlation-variable",
                  checked ? "checked" : "",
                  isObjective ? "objective" : "factor",
                ]
                  .filter(Boolean)
                  .join(" ")}
                key={column}
              >
                <input
                  checked={checked}
                  onChange={function () {
                    toggleColumn(column);
                  }}
                  type="checkbox"
                />
                <span>{column}</span>
                <small>{isObjective ? "Metric" : "Factor"}</small>
              </label>
            );
          })}
        </div>
      </div>

      {visibleColumns.length < 2 ? (
        <p className="muted correlation-empty">Select at least two variables to build the correlation view.</p>
      ) : viewMode === "network" ? (
        <CorrelationNetwork
          allGraphEdges={correlationGraph.allEdges}
          canUndoGroup={networkGroupHistory.length > 0}
          graphEdges={correlationGraph.edges}
          graphNodes={correlationGraph.nodes}
          method={method}
          onExpandGroup={expandNetworkGroup}
          onMergeNodes={mergeNetworkNodes}
          onPairSelect={toggleSelectedPair}
          onThresholdChange={setEdgeThreshold}
          onUndoGroup={undoNetworkMerge}
          selectedPairs={selectedPairs}
          threshold={edgeThreshold}
          totalEdgeCount={correlationGraph.totalEdgeCount}
        />
      ) : viewMode === "sphere" ? (
        <Suspense fallback={<p className="muted correlation-empty">Loading 3D view...</p>}>
          <CorrelationSphere
            canUndoGroup={networkGroupHistory.length > 0}
            columns={visibleColumns}
            graphEdges={correlationGraph.edges}
            graphNodes={correlationGraph.nodes}
            method={method}
            onCreateGroup={createNetworkGroup}
            onExpandGroup={expandNetworkGroup}
            onPairSelect={toggleSelectedPair}
            onThresholdChange={setEdgeThreshold}
            onUndoGroup={undoNetworkMerge}
            selectedPairs={selectedPairs}
            threshold={edgeThreshold}
            totalEdgeCount={correlationGraph.totalEdgeCount}
          />
        </Suspense>
      ) : (
        <>
          <div className="correlation-matrix-scroll">
            <table className="correlation-matrix">
              <thead>
                <tr>
                  <th aria-hidden="true" className="correlation-corner" />
                  {visibleColumns.map(function (column) {
                    const isObjective = objectiveColumns.includes(column);
                    return (
                      <th
                        className={isObjective ? "correlation-axis objective" : "correlation-axis factor"}
                        key={column}
                        scope="col"
                        title={column}
                      >
                        <span>{column}</span>
                      </th>
                    );
                  })}
                </tr>
              </thead>
              <tbody>
                {visibleColumns.map(function (rowColumn, rowIndex) {
                  const rowIsObjective = objectiveColumns.includes(rowColumn);
                  return (
                    <tr key={rowColumn}>
                      <th
                        className={
                          rowIsObjective ? "correlation-row-label objective" : "correlation-row-label factor"
                        }
                        scope="row"
                        title={rowColumn}
                      >
                        {rowColumn}
                      </th>
                      {visibleColumns.map(function (column, columnIndex) {
                        const result = matrix[rowIndex][columnIndex];
                        const colors = cellColors(result.value, method);
                        const isDiagonal = rowIndex === columnIndex;
                        const pair = [column, rowColumn];
                        const isActive = selectedPairs.some(function (selectedPair) {
                          return sameCorrelationPair(selectedPair, pair);
                        });
                        const methodLabel = METHOD_OPTIONS.find(function (option) {
                          return option.value === method;
                        })?.label;
                        return (
                          <td key={column}>
                            <button
                              aria-label={`${column} and ${rowColumn}: ${formatCorrelation(result.value)}`}
                              aria-pressed={isActive}
                              className={isActive ? "correlation-cell active" : "correlation-cell"}
                              disabled={isDiagonal || !Number.isFinite(result.value)}
                              onClick={function () {
                                toggleSelectedPair(pair);
                              }}
                              style={colors}
                              title={`${methodLabel}: ${formatCorrelation(result.value)} (n=${result.count})`}
                              type="button"
                            >
                              {formatCorrelation(result.value)}
                            </button>
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div className="correlation-footer">
            <span>
              {method === "mutual-information"
                ? "0 weaker dependence | 1 stronger dependence"
                : "-1 negative | 0 weak linear/monotonic | +1 positive"}
            </span>
            <span>
              {selectedPairs.length
                ? `Third view: ${selectedPairs.length} selected pairs across ${selectedAttributeCount} variables`
                : "Click a non-diagonal cell to update the third view."}
            </span>
          </div>
        </>
      )}
    </div>
  );
}
