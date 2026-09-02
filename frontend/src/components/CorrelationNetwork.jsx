import * as d3 from "d3";
import {useCallback, useEffect, useMemo, useRef, useState} from "react";

import {correlationEdgeColor, sameCorrelationPair} from "./correlationGraph";

const VIEW_WIDTH = 920;
const VIEW_HEIGHT = 560;
const VIEW_PADDING = 42;
const CENTER_X = VIEW_WIDTH / 2;
const CENTER_Y = VIEW_HEIGHT / 2;
const INNER_RADIUS = 76;
const OUTER_RADIUS = 228;
const RING_STRENGTHS = [0.8, 0.6, 0.4, 0.2, 0];

function formatValue(value) {
  return Number.isFinite(value) ? value.toFixed(2) : "-";
}

function shortLabel(label, limit = 16) {
  if (label.length <= limit) {
    return label;
  }
  return label.slice(0, limit - 1) + "...";
}

function nodeRadius(node) {
  if (node.kind === "group") {
    return 23 + Math.min(8, node.members.length * 1.5);
  }
  return node.kind === "objective" ? 20 : 18;
}

function radiusForStrength(strength) {
  const boundedStrength = Math.max(0, Math.min(1, strength || 0));
  return INNER_RADIUS + (1 - boundedStrength) * (OUTER_RADIUS - INNER_RADIUS);
}

function fallbackPosition(index, count) {
  const angle = (index / Math.max(1, count)) * Math.PI * 2;
  return {
    x: CENTER_X + Math.cos(angle) * 180,
    y: CENTER_Y + Math.sin(angle) * 180,
  };
}

export default function CorrelationNetwork({
  allGraphEdges,
  graphNodes,
  graphEdges,
  totalEdgeCount,
  canUndoGroup,
  method,
  threshold,
  onThresholdChange,
  selectedPairs = [],
  onPairSelect,
  onMergeNodes,
  onExpandGroup,
  onUndoGroup,
}) {
  const [positions, setPositions] = useState({});
  const [hoveredNodeId, setHoveredNodeId] = useState(null);
  const [focusedNodeId, setFocusedNodeId] = useState(null);
  const [draggedNodeId, setDraggedNodeId] = useState(null);
  const [dropTargetId, setDropTargetId] = useState(null);
  const positionsRef = useRef({});
  const freePositionsRef = useRef({});
  const simulationRef = useRef(null);
  const simulationFrameRef = useRef(null);
  const layoutFrameRef = useRef(null);
  const dragStateRef = useRef(null);
  const dropTargetIdRef = useRef(null);

  const nodeKey = graphNodes
    .map(function (node) {
      return node.id;
    })
    .sort()
    .join("\u0000");
  const latestEdgesRef = useRef(graphEdges);
  latestEdgesRef.current = graphEdges;

  const animateToPositions = useCallback(function (targets, duration = 420) {
    if (!targets) {
      return;
    }
    if (layoutFrameRef.current !== null) {
      window.cancelAnimationFrame(layoutFrameRef.current);
    }
    const starts = {};
    Object.keys(targets).forEach(function (id) {
      starts[id] = positionsRef.current[id] || targets[id];
    });
    const startedAt = window.performance.now();

    const tick = (now) => {
      const elapsed = Math.min(1, (now - startedAt) / duration);
      const eased = d3.easeCubicInOut(elapsed);
      const next = {...positionsRef.current};
      Object.keys(targets).forEach(function (id) {
        next[id] = {
          x: starts[id].x + (targets[id].x - starts[id].x) * eased,
          y: starts[id].y + (targets[id].y - starts[id].y) * eased,
        };
      });
      positionsRef.current = next;
      setPositions({...next});
      if (elapsed < 1) {
        layoutFrameRef.current = window.requestAnimationFrame(tick);
      } else {
        layoutFrameRef.current = null;
      }
    };
    layoutFrameRef.current = window.requestAnimationFrame(tick);
  }, []);

  useEffect(function () {
    return function () {
      if (layoutFrameRef.current !== null) {
        window.cancelAnimationFrame(layoutFrameRef.current);
      }
    };
  }, []);

  useEffect(
    function () {
      const availableIds = new Set(
        graphNodes.map(function (node) {
          return node.id;
        }),
      );
      setFocusedNodeId(function (previous) {
        return previous && availableIds.has(previous) ? previous : null;
      });
      setHoveredNodeId(function (previous) {
        return previous && availableIds.has(previous) ? previous : null;
      });
    },
    [nodeKey],
  );

  useEffect(
    function () {
      simulationRef.current?.stop();
      if (focusedNodeId) {
        simulationRef.current = null;
        return undefined;
      }

      const simulationNodes = graphNodes.map(function (node, index) {
        const existing =
          freePositionsRef.current[node.id] ||
          positionsRef.current[node.id] ||
          fallbackPosition(index, graphNodes.length);
        return {...node, x: existing.x, y: existing.y};
      });
      const simulationLinks = latestEdgesRef.current.map(function (edge) {
        return {...edge};
      });
      const simulation = d3
        .forceSimulation(simulationNodes)
        .alpha(0.82)
        .alphaDecay(0.035)
        .velocityDecay(0.32)
        .force(
          "link",
          d3
            .forceLink(simulationLinks)
            .id(function (node) {
              return node.id;
            })
            .distance(function (edge) {
              return 150 - edge.strength * 55;
            })
            .strength(0.28),
        )
        .force("charge", d3.forceManyBody().strength(-360))
        .force("center", d3.forceCenter(CENTER_X, CENTER_Y))
        .force(
          "collision",
          d3.forceCollide().radius(function (node) {
            return nodeRadius(node) + 22;
          }),
        )
        .force("x", d3.forceX(CENTER_X).strength(0.035))
        .force("y", d3.forceY(CENTER_Y).strength(0.035));

      const publishPositions = () => {
        simulationFrameRef.current = null;
        const next = {...positionsRef.current};
        simulationNodes.forEach(function (node) {
          const radius = nodeRadius(node);
          node.x = Math.max(
            VIEW_PADDING + radius,
            Math.min(VIEW_WIDTH - VIEW_PADDING - radius, node.x),
          );
          node.y = Math.max(
            VIEW_PADDING + radius,
            Math.min(VIEW_HEIGHT - VIEW_PADDING - radius, node.y),
          );
          next[node.id] = {x: node.x, y: node.y};
        });
        positionsRef.current = next;
        freePositionsRef.current = {...freePositionsRef.current, ...next};
        setPositions({...next});
      };

      simulation.on("tick", function () {
        if (simulationFrameRef.current === null) {
          simulationFrameRef.current = window.requestAnimationFrame(publishPositions);
        }
      });
      simulation.on("end", publishPositions);
      simulationRef.current = simulation;

      return function () {
        simulation.stop();
        if (simulationFrameRef.current !== null) {
          window.cancelAnimationFrame(simulationFrameRef.current);
          simulationFrameRef.current = null;
        }
      };
    },
    [focusedNodeId, nodeKey],
  );

  const focusedNode = graphNodes.find(function (node) {
    return node.id === focusedNodeId;
  });
  const radialLayout = useMemo(
    function () {
      if (!focusedNodeId) {
        return null;
      }
      const targets = {
        [focusedNodeId]: {x: CENTER_X, y: CENTER_Y},
      };
      const edgeByNodeId = new Map();
      allGraphEdges.forEach(function (edge) {
        if (edge.source === focusedNodeId) {
          edgeByNodeId.set(edge.target, edge);
        } else if (edge.target === focusedNodeId) {
          edgeByNodeId.set(edge.source, edge);
        }
      });
      const surroundingNodes = graphNodes.filter(function (node) {
        return node.id !== focusedNodeId;
      });
      surroundingNodes.forEach(function (node, index) {
        const edge = edgeByNodeId.get(node.id);
        const strength = edge?.strength || 0;
        const angle = -Math.PI / 2 + (index / Math.max(1, surroundingNodes.length)) * Math.PI * 2;
        const radius = radiusForStrength(strength);
        targets[node.id] = {
          x: CENTER_X + Math.cos(angle) * radius,
          y: CENTER_Y + Math.sin(angle) * radius,
        };
      });
      return {positions: targets};
    },
    [allGraphEdges, focusedNodeId, graphNodes],
  );

  useEffect(
    function () {
      if (radialLayout) {
        animateToPositions(radialLayout.positions);
      }
    },
    [animateToPositions, radialLayout],
  );

  const positionForNode = (node, index) => {
    return positions[node.id] || fallbackPosition(index, graphNodes.length);
  };
  const positionById = new Map(
    graphNodes.map(function (node, index) {
      return [node.id, positionForNode(node, index)];
    }),
  );

  const connectedNodeIds = useMemo(
    function () {
      if (!hoveredNodeId || focusedNodeId) {
        return new Set();
      }
      const connected = new Set([hoveredNodeId]);
      graphEdges.forEach(function (edge) {
        if (edge.source === hoveredNodeId) {
          connected.add(edge.target);
        }
        if (edge.target === hoveredNodeId) {
          connected.add(edge.source);
        }
      });
      return connected;
    },
    [focusedNodeId, graphEdges, hoveredNodeId],
  );

  const pointFromEvent = (event) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    return {
      x: ((event.clientX - bounds.left) / bounds.width) * VIEW_WIDTH,
      y: ((event.clientY - bounds.top) / bounds.height) * VIEW_HEIGHT,
    };
  };

  const findDropTarget = (sourceNode, point) => {
    let closest = null;
    let closestDistance = Number.POSITIVE_INFINITY;
    graphNodes.forEach(function (node) {
      if (node.id === sourceNode.id) {
        return;
      }
      if (node.kind === "group" && sourceNode.kind === "group") {
        return;
      }
      const targetPosition = positionsRef.current[node.id] || positionById.get(node.id);
      if (!targetPosition) {
        return;
      }
      const distance = Math.hypot(point.x - targetPosition.x, point.y - targetPosition.y);
      const mergeDistance = nodeRadius(node) + nodeRadius(sourceNode) * 0.55 + 14;
      if (distance <= mergeDistance && distance < closestDistance) {
        closest = node;
        closestDistance = distance;
      }
    });
    return closest;
  };

  const beginDrag = (event, node) => {
    if (event.pointerType === "mouse" && event.button !== 0) {
      return;
    }
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    const point = pointFromEvent({
      currentTarget: event.currentTarget.ownerSVGElement,
      clientX: event.clientX,
      clientY: event.clientY,
    });
    const origin = positionsRef.current[node.id] || positionById.get(node.id) || point;
    dragStateRef.current = {
      id: node.id,
      startX: point.x,
      startY: point.y,
      origin: {...origin},
      moved: false,
      node,
    };
    simulationRef.current?.stop();
    if (layoutFrameRef.current !== null) {
      window.cancelAnimationFrame(layoutFrameRef.current);
      layoutFrameRef.current = null;
    }
    setDraggedNodeId(node.id);
  };

  const moveDrag = (event) => {
    const drag = dragStateRef.current;
    if (!drag) {
      return;
    }
    const point = pointFromEvent(event);
    if (Math.hypot(point.x - drag.startX, point.y - drag.startY) > 4) {
      drag.moved = true;
    }
    if (!drag.moved) {
      return;
    }
    const radius = nodeRadius(drag.node);
    const nextPosition = {
      x: Math.max(VIEW_PADDING + radius, Math.min(VIEW_WIDTH - VIEW_PADDING - radius, point.x)),
      y: Math.max(VIEW_PADDING + radius, Math.min(VIEW_HEIGHT - VIEW_PADDING - radius, point.y)),
    };
    positionsRef.current = {...positionsRef.current, [drag.id]: nextPosition};
    setPositions({...positionsRef.current});

    const target = findDropTarget(drag.node, nextPosition);
    const nextTargetId = target?.id || null;
    if (dropTargetIdRef.current !== nextTargetId) {
      dropTargetIdRef.current = nextTargetId;
      setDropTargetId(nextTargetId);
    }
  };

  const resetDragState = () => {
    dragStateRef.current = null;
    dropTargetIdRef.current = null;
    setDraggedNodeId(null);
    setDropTargetId(null);
  };

  const endDrag = () => {
    const drag = dragStateRef.current;
    if (!drag) {
      return;
    }
    const targetId = dropTargetIdRef.current;
    resetDragState();

    if (!drag.moved) {
      setFocusedNodeId(drag.node.id);
      return;
    }

    const targetNode = graphNodes.find(function (node) {
      return node.id === targetId;
    });
    if (targetNode) {
      const mergedGroupId = onMergeNodes?.(drag.node, targetNode);
      if (mergedGroupId) {
        setFocusedNodeId(mergedGroupId);
        return;
      }
    }

    if (radialLayout) {
      animateToPositions(radialLayout.positions);
    } else {
      freePositionsRef.current = {
        ...freePositionsRef.current,
        [drag.id]: positionsRef.current[drag.id],
      };
    }
  };

  const cancelDrag = () => {
    const drag = dragStateRef.current;
    if (!drag) {
      return;
    }
    positionsRef.current = {...positionsRef.current, [drag.id]: drag.origin};
    setPositions({...positionsRef.current});
    resetDragState();
    if (radialLayout) {
      animateToPositions(radialLayout.positions);
    }
  };

  const expandFocusedGroup = () => {
    if (focusedNode?.kind !== "group") {
      return;
    }
    onExpandGroup?.(focusedNode.id);
    setFocusedNodeId(null);
  };

  return (
    <div className="correlation-network">
      <div className="correlation-network-controls">
        <label className="correlation-threshold">
          <span>
            Edge Threshold <strong>{threshold.toFixed(2)}</strong>
          </span>
          <input
            aria-label="Correlation edge threshold"
            max="1"
            min="0"
            onChange={function (event) {
              onThresholdChange(Number(event.target.value));
            }}
            step="0.05"
            type="range"
            value={threshold}
          />
        </label>
        <div className="correlation-network-actions">
          <button
            className="mini-button"
            disabled={threshold === 0}
            onClick={function () {
              onThresholdChange(0);
            }}
            type="button"
          >
            Show All
          </button>
          <span>{focusedNode ? `Focused: ${focusedNode.label}` : "Free layout"}</span>
          <button
            className="mini-button"
            disabled={!focusedNode}
            onClick={function () {
              setFocusedNodeId(null);
            }}
            type="button"
          >
            Free Layout
          </button>
          <button
            className="mini-button"
            disabled={focusedNode?.kind !== "group"}
            onClick={expandFocusedGroup}
            type="button"
          >
            Expand Group
          </button>
          <button
            className="mini-button"
            disabled={!canUndoGroup}
            onClick={function () {
              onUndoGroup?.();
            }}
            type="button"
          >
            Undo Last Merge
          </button>
        </div>
      </div>

      <div className="correlation-network-stage">
        <svg
          aria-label="Interactive correlation network"
          onPointerCancel={cancelDrag}
          onPointerMove={moveDrag}
          onPointerUp={endDrag}
          role="img"
          viewBox={`0 0 ${VIEW_WIDTH} ${VIEW_HEIGHT}`}
        >
          <rect
            className="correlation-network-background"
            height={VIEW_HEIGHT}
            onPointerDown={function () {
              setFocusedNodeId(null);
              setHoveredNodeId(null);
            }}
            width={VIEW_WIDTH}
            x="0"
            y="0"
          />

          {radialLayout ? (
            <g className="correlation-focus-rings" aria-hidden="true">
              {RING_STRENGTHS.map(function (strength) {
                const radius = radiusForStrength(strength);
                return (
                  <g key={strength}>
                    <circle cx={CENTER_X} cy={CENTER_Y} r={radius} />
                    <text textAnchor="start" x={CENTER_X + 7} y={CENTER_Y - radius + 12}>
                      {method === "mutual-information" ? "MI " : "|r| "}
                      {strength.toFixed(1)}
                    </text>
                  </g>
                );
              })}
            </g>
          ) : null}

          {graphEdges.map(function (edge) {
            const source = positionById.get(edge.source);
            const target = positionById.get(edge.target);
            if (!source || !target) {
              return null;
            }
            const isFocusEdge =
              focusedNodeId &&
              (edge.source === focusedNodeId || edge.target === focusedNodeId);
            const isDimmed = focusedNodeId
              ? !isFocusEdge
              : hoveredNodeId &&
                edge.source !== hoveredNodeId &&
                edge.target !== hoveredNodeId;
            const isActive = selectedPairs.some(function (pair) {
              return sameCorrelationPair(pair, edge.pair);
            });
            const title = `${edge.pair[0]} x ${edge.pair[1]}: ${formatValue(edge.value)} (n=${edge.count})`;
            const selectPair = () => {
              onPairSelect?.(edge.pair);
            };
            return (
              <g
                className={isActive ? "correlation-edge active" : "correlation-edge"}
                key={edge.id}
                onClick={function (event) {
                  event.stopPropagation();
                  selectPair();
                }}
                onKeyDown={function (event) {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    selectPair();
                  }
                }}
                role="button"
                tabIndex="0"
              >
                <title>{title}</title>
                <line
                  className="correlation-edge-hit"
                  x1={source.x}
                  x2={target.x}
                  y1={source.y}
                  y2={target.y}
                />
                <line
                  className="correlation-edge-line"
                  opacity={
                    isDimmed
                      ? 0.05
                      : Math.min(1, 0.2 + edge.strength * 0.72 + (isFocusEdge ? 0.12 : 0))
                  }
                  stroke={isActive ? "#172a3a" : correlationEdgeColor(edge, method)}
                  strokeWidth={(isActive ? 2 : 0) + (isFocusEdge ? 1.5 : 0) + 1 + edge.strength * 7}
                  x1={source.x}
                  x2={target.x}
                  y1={source.y}
                  y2={target.y}
                />
              </g>
            );
          })}

          {graphNodes.map(function (node, index) {
            const position = positionById.get(node.id);
            const radius = nodeRadius(node);
            const isFocused = focusedNodeId === node.id;
            const isDropTarget = dropTargetId === node.id;
            const isDragging = draggedNodeId === node.id;
            const isDimmed =
              !focusedNodeId && hoveredNodeId && !connectedNodeIds.has(node.id);
            const title =
              node.kind === "group"
                ? `${node.label}: ${node.members.join(", ")}`
                : `${node.kind === "objective" ? "Performance metric" : "Factor"}: ${node.label}`;
            return (
              <g
                aria-label={title}
                className={[
                  "correlation-node",
                  node.kind,
                  isFocused ? "focused" : "",
                  isDropTarget ? "drop-target" : "",
                  isDragging ? "dragging" : "",
                  isDimmed ? "dimmed" : "",
                ]
                  .filter(Boolean)
                  .join(" ")}
                key={node.id}
                onDoubleClick={function (event) {
                  event.stopPropagation();
                  if (node.kind === "group") {
                    onExpandGroup?.(node.id);
                    setFocusedNodeId(null);
                  }
                }}
                onKeyDown={function (event) {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    setFocusedNodeId(node.id);
                  }
                }}
                onPointerDown={function (event) {
                  beginDrag(event, node);
                }}
                onPointerEnter={function () {
                  setHoveredNodeId(node.id);
                }}
                onPointerLeave={function () {
                  setHoveredNodeId(null);
                }}
                role="button"
                tabIndex="0"
                transform={`translate(${position.x} ${position.y})`}
              >
                <title>{title}</title>
                {isFocused ? <circle className="correlation-node-focus-halo" r={radius + 9} /> : null}
                {isDropTarget ? <circle className="correlation-node-drop-halo" r={radius + 13} /> : null}
                {node.kind === "group" ? (
                  <>
                    <circle className="correlation-node-stack back" cx="-7" cy="5" r={radius - 2} />
                    <circle className="correlation-node-stack middle" cx="5" cy="-5" r={radius - 2} />
                  </>
                ) : null}
                <circle className="correlation-node-body" r={radius} />
                <text className="correlation-node-label" textAnchor="middle" y="4">
                  {shortLabel(node.label)}
                </text>
                {node.kind === "group" ? (
                  <text className="correlation-node-count" textAnchor="middle" y={radius + 16}>
                    {node.members.length} variables
                  </text>
                ) : null}
              </g>
            );
          })}

          {graphEdges.length === 0 ? (
            <text className="correlation-network-empty" textAnchor="middle" x={CENTER_X} y={VIEW_HEIGHT - 24}>
              No edges meet the current threshold.
            </text>
          ) : null}
        </svg>
      </div>

      <div className="correlation-network-footer">
        <div className="correlation-network-legend">
          <span><i className="legend-node factor" />Factor</span>
          <span><i className="legend-node objective" />Metric</span>
          <span><i className="legend-node group" />Group</span>
          {method === "mutual-information" ? (
            <span><i className="legend-line information" />Dependence</span>
          ) : (
            <>
              <span><i className="legend-line positive" />Positive</span>
              <span><i className="legend-line negative" />Negative</span>
            </>
          )}
        </div>
        <span>
          {graphNodes.length} nodes | {graphEdges.length} / {totalEdgeCount} edges shown
        </span>
      </div>
    </div>
  );
}
