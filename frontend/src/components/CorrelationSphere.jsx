import {useEffect, useRef, useState} from "react";
import * as THREE from "three";
import {OrbitControls} from "three/examples/jsm/controls/OrbitControls.js";

import {correlationEdgeColor, sameCorrelationPair} from "./correlationGraph";

const SPHERE_RADIUS = 2.35;
const NODE_COLORS = {
  factor: "#4f8db4",
  objective: "#4b9873",
  group: "#7a6a9f",
};

function formatValue(value) {
  return Number.isFinite(value) ? value.toFixed(2) : "-";
}

function fibonacciPosition(index, count) {
  if (count <= 1) {
    return new THREE.Vector3(0, SPHERE_RADIUS, 0);
  }
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  const y = 1 - (index / (count - 1)) * 2;
  const horizontalRadius = Math.sqrt(Math.max(0, 1 - y * y));
  const angle = goldenAngle * index;
  return new THREE.Vector3(
    Math.cos(angle) * horizontalRadius,
    y,
    Math.sin(angle) * horizontalRadius,
  ).multiplyScalar(SPHERE_RADIUS);
}

function positionsForNodes(columns, nodes) {
  const basePositions = new Map(
    columns.map(function (column, index) {
      return [column, fibonacciPosition(index, columns.length)];
    }),
  );
  const positions = new Map();

  nodes.forEach(function (node, index) {
    if (node.kind !== "group") {
      positions.set(
        node.id,
        basePositions.get(node.id)?.clone() || fibonacciPosition(index, nodes.length),
      );
      return;
    }

    const average = new THREE.Vector3();
    node.members.forEach(function (member) {
      const position = basePositions.get(member);
      if (position) {
        average.add(position.clone().normalize());
      }
    });
    if (average.lengthSq() < 0.0001) {
      positions.set(node.id, fibonacciPosition(index, nodes.length));
    } else {
      positions.set(node.id, average.normalize().multiplyScalar(SPHERE_RADIUS));
    }
  });

  return positions;
}

function edgeCurve(source, target) {
  const sourceDirection = source.clone().normalize();
  const targetDirection = target.clone().normalize();
  const middleDirection = sourceDirection.clone().add(targetDirection);
  if (middleDirection.lengthSq() < 0.0001) {
    middleDirection.crossVectors(sourceDirection, new THREE.Vector3(0, 1, 0));
    if (middleDirection.lengthSq() < 0.0001) {
      middleDirection.crossVectors(sourceDirection, new THREE.Vector3(1, 0, 0));
    }
  }
  middleDirection.normalize();
  const angle = sourceDirection.angleTo(targetDirection);
  const controlRadius = SPHERE_RADIUS + 0.28 + Math.min(0.78, angle * 0.25);
  return new THREE.QuadraticBezierCurve3(
    source.clone().multiplyScalar(1.015),
    middleDirection.multiplyScalar(controlRadius),
    target.clone().multiplyScalar(1.015),
  );
}

function roundedRect(context, x, y, width, height, radius) {
  context.beginPath();
  context.moveTo(x + radius, y);
  context.lineTo(x + width - radius, y);
  context.quadraticCurveTo(x + width, y, x + width, y + radius);
  context.lineTo(x + width, y + height - radius);
  context.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  context.lineTo(x + radius, y + height);
  context.quadraticCurveTo(x, y + height, x, y + height - radius);
  context.lineTo(x, y + radius);
  context.quadraticCurveTo(x, y, x + radius, y);
  context.closePath();
}

function labelSprite(label, kind) {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 112;
  const context = canvas.getContext("2d");
  const displayLabel = label.length > 24 ? label.slice(0, 23) + "..." : label;

  context.clearRect(0, 0, canvas.width, canvas.height);
  roundedRect(context, 5, 5, canvas.width - 10, canvas.height - 10, 24);
  context.fillStyle = "rgba(251, 253, 254, 0.94)";
  context.fill();
  context.lineWidth = 5;
  context.strokeStyle = NODE_COLORS[kind] || NODE_COLORS.factor;
  context.stroke();
  context.fillStyle = "#23323c";
  context.font = "700 34px system-ui, sans-serif";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(displayLabel, canvas.width / 2, canvas.height / 2 + 1, canvas.width - 42);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.minFilter = THREE.LinearFilter;
  const material = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthTest: false,
    depthWrite: false,
  });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(1.42, 0.31, 1);
  sprite.renderOrder = 8;
  return sprite;
}

function disposeObject(object) {
  object.traverse(function (child) {
    child.geometry?.dispose();
    const materials = Array.isArray(child.material) ? child.material : [child.material];
    materials.filter(Boolean).forEach(function (material) {
      material.map?.dispose();
      material.dispose();
    });
  });
}

function clearGroup(group) {
  while (group.children.length > 0) {
    const child = group.children[0];
    group.remove(child);
    disposeObject(child);
  }
}

function nodeDescription(node) {
  if (node.kind === "group") {
    return `${node.members.length} variables: ${node.members.join(", ")}`;
  }
  return node.kind === "objective" ? "Performance metric" : "Factor";
}

export default function CorrelationSphere({
  columns,
  graphNodes,
  graphEdges,
  totalEdgeCount,
  canUndoGroup,
  method,
  threshold,
  onThresholdChange,
  selectedPairs = [],
  onPairSelect,
  onCreateGroup,
  onExpandGroup,
  onUndoGroup,
}) {
  const stageRef = useRef(null);
  const sceneStateRef = useRef(null);
  const graphObjectsRef = useRef({nodes: new Map(), edges: new Map(), labels: []});
  const callbacksRef = useRef({onPairSelect, onExpandGroup});
  const visualStateRef = useRef({});
  const [hoveredTarget, setHoveredTarget] = useState(null);
  const [pinnedNodeId, setPinnedNodeId] = useState(null);
  const [selectedVariableIds, setSelectedVariableIds] = useState([]);
  const [selectedGroupId, setSelectedGroupId] = useState(null);
  const [autoRotate, setAutoRotate] = useState(true);
  const [renderError, setRenderError] = useState(null);

  callbacksRef.current = {onPairSelect, onExpandGroup};
  visualStateRef.current = {
    autoRotate,
    hoveredTarget,
    pinnedNodeId,
    selectedVariableIds,
    selectedGroupId,
  };

  useEffect(function () {
    const host = stageRef.current;
    if (!host) {
      return undefined;
    }

    setRenderError(null);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
    camera.position.set(0.15, 0.18, 7.4);

    let renderer;
    try {
      renderer = new THREE.WebGLRenderer({
        alpha: true,
        antialias: true,
        powerPreference: "high-performance",
        preserveDrawingBuffer: true,
      });
    } catch (error) {
      setRenderError("3D rendering is unavailable in this browser.");
      return undefined;
    }

    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0xf8fbfc, 0);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.domElement.dataset.correlationSphereCanvas = "true";
    renderer.domElement.setAttribute("aria-label", "Interactive 3D correlation sphere");
    renderer.domElement.setAttribute("role", "img");
    host.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.075;
    controls.rotateSpeed = 0.72;
    controls.zoomSpeed = 0.8;
    controls.minDistance = 4.7;
    controls.maxDistance = 11;
    controls.autoRotateSpeed = 0.7;
    controls.target.set(0, 0, 0);
    controls.update();
    controls.saveState();

    scene.add(new THREE.HemisphereLight(0xffffff, 0xc7d5dd, 2.1));
    const keyLight = new THREE.DirectionalLight(0xffffff, 2.4);
    keyLight.position.set(4, 5, 6);
    scene.add(keyLight);
    const fillLight = new THREE.DirectionalLight(0x9ec9df, 1.2);
    fillLight.position.set(-5, -2, 3);
    scene.add(fillLight);

    const sphereSurface = new THREE.Mesh(
      new THREE.SphereGeometry(SPHERE_RADIUS * 0.992, 40, 28),
      new THREE.MeshBasicMaterial({
        color: 0xd7e5eb,
        opacity: 0.055,
        transparent: true,
        depthWrite: false,
        side: THREE.DoubleSide,
      }),
    );
    sphereSurface.renderOrder = 0;
    scene.add(sphereSurface);

    const sphereWireframe = new THREE.LineSegments(
      new THREE.WireframeGeometry(new THREE.SphereGeometry(SPHERE_RADIUS, 22, 14)),
      new THREE.LineBasicMaterial({
        color: 0x8aa7b7,
        opacity: 0.12,
        transparent: true,
        depthWrite: false,
      }),
    );
    sphereWireframe.renderOrder = 1;
    scene.add(sphereWireframe);

    const graphGroup = new THREE.Group();
    scene.add(graphGroup);
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    let pointerStart = null;

    const pickTarget = (event) => {
      const bounds = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - bounds.left) / bounds.width) * 2 - 1;
      pointer.y = -((event.clientY - bounds.top) / bounds.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);

      const nodeHits = raycaster.intersectObjects(
        Array.from(graphObjectsRef.current.nodes.values()),
        false,
      );
      if (nodeHits.length > 0) {
        return {
          kind: "node",
          data: nodeHits[0].object.userData.correlationNode,
        };
      }
      const edgeHits = raycaster.intersectObjects(
        Array.from(graphObjectsRef.current.edges.values()),
        false,
      );
      if (edgeHits.length > 0) {
        return {
          kind: "edge",
          data: edgeHits[0].object.userData.correlationEdge,
        };
      }
      return null;
    };

    const activateNode = (node) => {
      setPinnedNodeId(function (previous) {
        return previous === node.id ? null : node.id;
      });
      if (node.kind !== "group") {
        setSelectedGroupId(null);
        setSelectedVariableIds(function (previous) {
          return previous.includes(node.id)
            ? previous.filter(function (id) {
                return id !== node.id;
              })
            : previous.concat(node.id);
        });
      } else if (node.kind === "group") {
        setSelectedVariableIds([]);
        setSelectedGroupId(function (previous) {
          return previous === node.id ? null : node.id;
        });
      }
    };

    const handlePointerDown = (event) => {
      pointerStart = {x: event.clientX, y: event.clientY, moved: false};
      renderer.domElement.style.cursor = "grabbing";
    };
    const handlePointerMove = (event) => {
      if (
        pointerStart &&
        Math.hypot(event.clientX - pointerStart.x, event.clientY - pointerStart.y) > 4
      ) {
        pointerStart.moved = true;
      }
      if (pointerStart?.moved) {
        setHoveredTarget(null);
        return;
      }
      const target = pickTarget(event);
      renderer.domElement.style.cursor = target ? "pointer" : "grab";
      setHoveredTarget(target);
    };
    const handlePointerUp = (event) => {
      const wasClick = pointerStart && !pointerStart.moved;
      pointerStart = null;
      const target = pickTarget(event);
      renderer.domElement.style.cursor = target ? "pointer" : "grab";
      if (!wasClick) {
        return;
      }
      if (target?.kind === "node") {
        activateNode(target.data);
      } else if (target?.kind === "edge") {
        callbacksRef.current.onPairSelect?.(target.data.pair);
      } else {
        setPinnedNodeId(null);
        setSelectedVariableIds([]);
        setSelectedGroupId(null);
      }
    };
    const handlePointerCancel = () => {
      pointerStart = null;
      renderer.domElement.style.cursor = "grab";
    };
    const handlePointerLeave = () => {
      setHoveredTarget(null);
      if (!pointerStart) {
        renderer.domElement.style.cursor = "grab";
      }
    };
    const handleDoubleClick = (event) => {
      const target = pickTarget(event);
      if (target?.kind === "node" && target.data.kind === "group") {
        callbacksRef.current.onExpandGroup?.(target.data.id);
        setSelectedGroupId(null);
        setPinnedNodeId(null);
      }
    };

    const canvas = renderer.domElement;
    canvas.addEventListener("pointerdown", handlePointerDown);
    canvas.addEventListener("pointermove", handlePointerMove);
    canvas.addEventListener("pointerup", handlePointerUp);
    canvas.addEventListener("pointercancel", handlePointerCancel);
    canvas.addEventListener("pointerleave", handlePointerLeave);
    canvas.addEventListener("dblclick", handleDoubleClick);

    const resize = () => {
      const width = Math.max(1, host.clientWidth);
      const height = Math.max(1, host.clientHeight);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
    };
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(host);
    resize();

    sceneStateRef.current = {camera, controls, graphGroup, renderer, scene};
    let animationFrame;
    const animate = () => {
      controls.autoRotate = Boolean(visualStateRef.current.autoRotate);
      controls.update();

      const forcedLabels = new Set(
        [
          visualStateRef.current.pinnedNodeId,
          visualStateRef.current.hoveredTarget?.kind === "node"
            ? visualStateRef.current.hoveredTarget.data.id
            : null,
          ...(visualStateRef.current.selectedVariableIds || []),
          visualStateRef.current.selectedGroupId,
        ].filter(Boolean),
      );
      const denseGraph = graphObjectsRef.current.labels.length > 18;
      graphObjectsRef.current.labels.forEach(function (entry, index) {
        const outward = entry.position.clone().normalize();
        const toCamera = camera.position.clone().sub(entry.position).normalize();
        const facesCamera = outward.dot(toCamera) > (denseGraph ? 0.42 : 0.1);
        entry.sprite.visible = forcedLabels.has(entry.nodeId) || (facesCamera && (!denseGraph || index % 2 === 0));
        entry.sprite.material.opacity = forcedLabels.has(entry.nodeId) ? 1 : 0.88;
      });

      renderer.render(scene, camera);
      animationFrame = window.requestAnimationFrame(animate);
    };
    animate();

    return function () {
      window.cancelAnimationFrame(animationFrame);
      resizeObserver.disconnect();
      canvas.removeEventListener("pointerdown", handlePointerDown);
      canvas.removeEventListener("pointermove", handlePointerMove);
      canvas.removeEventListener("pointerup", handlePointerUp);
      canvas.removeEventListener("pointercancel", handlePointerCancel);
      canvas.removeEventListener("pointerleave", handlePointerLeave);
      canvas.removeEventListener("dblclick", handleDoubleClick);
      controls.dispose();
      disposeObject(scene);
      renderer.dispose();
      renderer.forceContextLoss();
      if (host.contains(canvas)) {
        host.removeChild(canvas);
      }
      sceneStateRef.current = null;
      graphObjectsRef.current = {nodes: new Map(), edges: new Map(), labels: []};
    };
  }, []);

  useEffect(
    function () {
      const sceneState = sceneStateRef.current;
      if (!sceneState) {
        return;
      }
      clearGroup(sceneState.graphGroup);

      const nodePositions = positionsForNodes(columns, graphNodes);
      const edgeObjects = new Map();
      graphEdges.forEach(function (edge) {
        const source = nodePositions.get(edge.source);
        const target = nodePositions.get(edge.target);
        if (!source || !target) {
          return;
        }
        const geometry = new THREE.TubeGeometry(
          edgeCurve(source, target),
          28,
          0.008 + edge.strength * 0.023,
          6,
          false,
        );
        const material = new THREE.MeshBasicMaterial({
          color: correlationEdgeColor(edge, method),
          opacity: 0.22 + edge.strength * 0.58,
          transparent: true,
          depthWrite: false,
        });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.renderOrder = 2;
        mesh.userData.correlationEdge = edge;
        sceneState.graphGroup.add(mesh);
        edgeObjects.set(edge.id, mesh);
      });

      const nodeObjects = new Map();
      const labels = [];
      graphNodes.forEach(function (node) {
        const position = nodePositions.get(node.id);
        let geometry;
        if (node.kind === "objective") {
          geometry = new THREE.OctahedronGeometry(0.19, 1);
        } else if (node.kind === "group") {
          geometry = new THREE.IcosahedronGeometry(0.24, 1);
        } else {
          geometry = new THREE.SphereGeometry(0.16, 24, 18);
        }
        const material = new THREE.MeshStandardMaterial({
          color: NODE_COLORS[node.kind] || NODE_COLORS.factor,
          emissive: 0x000000,
          metalness: node.kind === "objective" ? 0.12 : 0.04,
          opacity: 1,
          roughness: 0.55,
          transparent: true,
        });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.copy(position);
        mesh.renderOrder = 4;
        mesh.userData.correlationNode = node;
        sceneState.graphGroup.add(mesh);
        nodeObjects.set(node.id, mesh);

        const sprite = labelSprite(node.label, node.kind);
        const labelPosition = position.clone().normalize().multiplyScalar(SPHERE_RADIUS + 0.34);
        sprite.position.copy(labelPosition);
        sceneState.graphGroup.add(sprite);
        labels.push({nodeId: node.id, position: labelPosition, sprite});
      });

      graphObjectsRef.current = {nodes: nodeObjects, edges: edgeObjects, labels};
      setHoveredTarget(null);
    },
    [columns, graphEdges, graphNodes, method],
  );

  useEffect(
    function () {
      const availableNodeIds = new Set(
        graphNodes.map(function (node) {
          return node.id;
        }),
      );
      setSelectedVariableIds(function (previous) {
        return previous.filter(function (id) {
          return availableNodeIds.has(id);
        });
      });
      setSelectedGroupId(function (previous) {
        return previous && availableNodeIds.has(previous) ? previous : null;
      });
      setPinnedNodeId(function (previous) {
        return previous && availableNodeIds.has(previous) ? previous : null;
      });
    },
    [graphNodes],
  );

  useEffect(
    function () {
      const highlightedNodeId =
        hoveredTarget?.kind === "node" ? hoveredTarget.data.id : pinnedNodeId;
      const connectedNodeIds = new Set(highlightedNodeId ? [highlightedNodeId] : []);
      if (highlightedNodeId) {
        graphEdges.forEach(function (edge) {
          if (edge.source === highlightedNodeId) {
            connectedNodeIds.add(edge.target);
          }
          if (edge.target === highlightedNodeId) {
            connectedNodeIds.add(edge.source);
          }
        });
      }

      graphObjectsRef.current.nodes.forEach(function (mesh, nodeId) {
        const selected =
          selectedVariableIds.includes(nodeId) || selectedGroupId === nodeId;
        const highlighted = highlightedNodeId === nodeId;
        const dimmed = highlightedNodeId && !connectedNodeIds.has(nodeId);
        mesh.material.opacity = dimmed ? 0.2 : 1;
        mesh.material.emissive.set(selected ? 0x172a3a : highlighted ? 0x214a63 : 0x000000);
        mesh.material.emissiveIntensity = selected ? 0.34 : highlighted ? 0.2 : 0;
        mesh.scale.setScalar(selected ? 1.34 : highlighted ? 1.18 : 1);
      });

      graphObjectsRef.current.edges.forEach(function (mesh) {
        const edge = mesh.userData.correlationEdge;
        const isConnected =
          highlightedNodeId &&
          (edge.source === highlightedNodeId || edge.target === highlightedNodeId);
        const isHovered = hoveredTarget?.kind === "edge" && hoveredTarget.data.id === edge.id;
        const isActive = selectedPairs.some(function (pair) {
          return sameCorrelationPair(pair, edge.pair);
        });
        const isDimmed = highlightedNodeId && !isConnected;
        mesh.material.color.set(
          isHovered || isActive ? "#172a3a" : correlationEdgeColor(edge, method),
        );
        mesh.material.opacity = isDimmed
          ? 0.04
          : isHovered || isActive
            ? 1
            : 0.22 + edge.strength * 0.58;
        mesh.renderOrder = isHovered || isActive ? 5 : 2;
      });
    },
    [graphEdges, graphNodes, hoveredTarget, method, pinnedNodeId, selectedGroupId, selectedPairs, selectedVariableIds],
  );

  const mergeSelectedVariables = () => {
    if (selectedVariableIds.length < 2) {
      return;
    }
    onCreateGroup?.(selectedVariableIds);
    setSelectedVariableIds([]);
  };
  const expandSelectedGroup = () => {
    if (!selectedGroupId) {
      return;
    }
    onExpandGroup?.(selectedGroupId);
    setSelectedGroupId(null);
  };
  const pinnedNode = graphNodes.find(function (node) {
    return node.id === pinnedNodeId;
  });
  const detailTarget = hoveredTarget || (pinnedNode ? {kind: "node", data: pinnedNode} : null);

  return (
    <div className="correlation-sphere">
      <div className="correlation-network-controls correlation-sphere-controls">
        <label className="correlation-threshold">
          <span>
            Edge Threshold <strong>{threshold.toFixed(2)}</strong>
          </span>
          <input
            aria-label="3D correlation edge threshold"
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
          <button
            aria-pressed={autoRotate}
            className={autoRotate ? "mini-toggle active" : "mini-toggle"}
            onClick={function () {
              setAutoRotate(function (previous) {
                return !previous;
              });
            }}
            type="button"
          >
            Auto Rotate
          </button>
          <button
            className="mini-button"
            onClick={function () {
              sceneStateRef.current?.controls.reset();
            }}
            type="button"
          >
            Reset View
          </button>
          <span>{selectedVariableIds.length} variables selected</span>
          <button
            className="mini-button"
            disabled={selectedVariableIds.length < 2}
            onClick={mergeSelectedVariables}
            type="button"
          >
            Merge Selected
          </button>
          <button
            className="mini-button"
            disabled={!selectedGroupId}
            onClick={expandSelectedGroup}
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

      <div className="correlation-sphere-stage" ref={stageRef}>
        {renderError ? <p className="error-text correlation-sphere-error">{renderError}</p> : null}
        {detailTarget ? (
          <div className="correlation-sphere-tooltip">
            {detailTarget.kind === "node" ? (
              <>
                <strong>{detailTarget.data.label}</strong>
                <span>{nodeDescription(detailTarget.data)}</span>
              </>
            ) : (
              <>
                <strong>{detailTarget.data.pair.join(" x ")}</strong>
                <span>
                  Correlation {formatValue(detailTarget.data.value)} | n={detailTarget.data.count}
                </span>
              </>
            )}
          </div>
        ) : null}
        {graphEdges.length === 0 && !renderError ? (
          <span className="correlation-sphere-empty">No edges meet the current threshold.</span>
        ) : null}
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
