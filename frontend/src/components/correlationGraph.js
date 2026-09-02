export function buildCorrelationGraph({
  columns,
  objectiveColumns,
  matrix,
  threshold,
  groups,
}) {
  const columnIndex = new Map(
    columns.map(function (column, index) {
      return [column, index];
    }),
  );
  const visibleGroups = groups
    .map(function (group) {
      return {
        ...group,
        members: group.members.filter(function (member) {
          return columnIndex.has(member);
        }),
      };
    })
    .filter(function (group) {
      return group.members.length >= 2;
    });
  const groupedMembers = new Set(
    visibleGroups.flatMap(function (group) {
      return group.members;
    }),
  );
  const individualNodes = columns
    .filter(function (column) {
      return !groupedMembers.has(column);
    })
    .map(function (column) {
      return {
        id: column,
        label: column,
        kind: objectiveColumns.includes(column) ? "objective" : "factor",
        members: [column],
      };
    });
  const groupNodes = visibleGroups.map(function (group) {
    return {
      id: group.id,
      label: group.label,
      kind: "group",
      members: group.members,
    };
  });
  const nodes = individualNodes.concat(groupNodes);
  const allEdges = [];
  const edges = [];
  let totalEdgeCount = 0;

  for (let firstIndex = 0; firstIndex < nodes.length; firstIndex += 1) {
    for (let secondIndex = firstIndex + 1; secondIndex < nodes.length; secondIndex += 1) {
      const firstNode = nodes[firstIndex];
      const secondNode = nodes[secondIndex];
      let representative = null;

      firstNode.members.forEach(function (firstMember) {
        secondNode.members.forEach(function (secondMember) {
          const rowIndex = columnIndex.get(firstMember);
          const columnPosition = columnIndex.get(secondMember);
          const result = matrix[rowIndex]?.[columnPosition];
          if (!Number.isFinite(result?.value)) {
            return;
          }
          if (
            representative === null ||
            Math.abs(result.value) > Math.abs(representative.value)
          ) {
            representative = {
              value: result.value,
              count: result.count,
              pair: [firstMember, secondMember],
            };
          }
        });
      });

      if (representative === null) {
        continue;
      }
      const edge = {
        id: firstNode.id + "::" + secondNode.id,
        source: firstNode.id,
        target: secondNode.id,
        strength: Math.abs(representative.value),
        ...representative,
      };
      totalEdgeCount += 1;
      allEdges.push(edge);
      if (edge.strength >= threshold) {
        edges.push(edge);
      }
    }
  }

  return {allEdges, nodes, edges, totalEdgeCount};
}

export function sameCorrelationPair(first, second) {
  return (
    Array.isArray(first) &&
    Array.isArray(second) &&
    ((first[0] === second[0] && first[1] === second[1]) ||
      (first[0] === second[1] && first[1] === second[0]))
  );
}

export function correlationEdgeColor(edge, method) {
  if (method === "mutual-information") {
    return "#2b8a72";
  }
  return edge.value < 0 ? "#b84f4a" : "#2c77a5";
}
