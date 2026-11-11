import {useEffect, useMemo, useRef} from "react";

import legacyWidget from "../legacy/widget";
import {createLegacyWidgetModel} from "../legacy/model-adapter";

const MODEL_PREDICATE_MODES = {
  regression: "predicate regression",
  "paper-regression": "paper predicate regression",
  rpi: "recursive predicate induction",
};

export default function LegacyDimbridgeWidget({
  analysis,
  predicateMode,
  predicateOptions,
  brushMode,
  focusedSplomAttributes,
  objectiveSelectionMask,
  objectiveSelectionSource,
  predicateFactorLimit,
  resetToken,
  onPredicates,
  onError,
  onLoadingChange,
  onMetaChange,
  onSelectionChange,
}) {
  const hostRef = useRef(null);
  const renderedRef = useRef(null);
  const callbacksRef = useRef({
    onPredicates,
    onError,
    onLoadingChange,
    onSelectionChange,
  });

  useEffect(
    function () {
      callbacksRef.current = {
        onPredicates,
        onError,
        onLoadingChange,
        onSelectionChange,
      };
    },
    [onPredicates, onError, onLoadingChange, onSelectionChange],
  );

  const preparedAnalysis = useMemo(
    function () {
      if (!analysis) {
        return null;
      }

      const factorColumns = analysis.factor_columns || [];
      const objectiveColumns = analysis.objective_columns || [];
      const sourceRecords = analysis.records || [];
      const x = analysis.x || [];
      const y = analysis.y || [];
      const rowCount = Math.min(sourceRecords.length, x.length, y.length);

      if (!factorColumns.length || rowCount === 0) {
        return null;
      }

      const records = [];
      const xValues = [];
      const yValues = [];
      const colors = [];
      const sourceIndices = [];
      const rawColorValues = Array.isArray(analysis.color_values) ? analysis.color_values : [];

      for (let index = 0; index < rowCount; index += 1) {
        const record = sourceRecords[index];
        const numericRecord = {};
        let isValid = Number.isFinite(Number(x[index])) && Number.isFinite(Number(y[index]));

        factorColumns.forEach(function (column) {
          const rawValue = record[column];
          numericRecord[column] =
            rawValue === null || rawValue === undefined || rawValue === "" ? Number.NaN : Number(rawValue);
          if (!Number.isFinite(numericRecord[column])) {
            isValid = false;
          }
        });

        objectiveColumns.forEach(function (column) {
          if (Object.prototype.hasOwnProperty.call(numericRecord, column)) {
            return;
          }
          const rawValue = record[column];
          const numericValue =
            rawValue === null || rawValue === undefined || rawValue === "" ? Number.NaN : Number(rawValue);
          numericRecord[column] = Number.isFinite(numericValue) ? numericValue : Number.NaN;
        });

        if (isValid) {
          records.push(numericRecord);
          sourceIndices.push(index);
          xValues.push(Number(x[index]));
          yValues.push(Number(y[index]));
          if (analysis.color_mode === "continuous") {
            colors.push(Number(rawColorValues[index]));
          } else if (analysis.color_mode === "categorical") {
            const value = rawColorValues[index];
            if (Array.isArray(value) && value.length === 3) {
              colors.push(
                value.map(function (channel) {
                  return Number(channel);
                }),
              );
            } else {
              colors.push([160, 160, 160]);
            }
          } else {
            colors.push([31, 119, 180]);
          }
        }
      }

      const droppedRows = (analysis.dropped_row_count || 0) + (rowCount - records.length);
      if (analysis.color_mode === "continuous" && colors.length !== records.length) {
        return null;
      }
      if (analysis.color_mode === "categorical" && colors.length !== records.length) {
        return null;
      }

      return {
        datasetId: analysis.dataset_id,
        factorColumns,
        objectiveColumns,
        records,
        sourceIndices,
        sourceRowCount: sourceRecords.length,
        x: xValues,
        y: yValues,
        colors,
        totalRows: analysis.row_count || rowCount,
        droppedRows,
      };
    },
    [analysis],
  );

  const spec = useMemo(
    function () {
      if (!preparedAnalysis || preparedAnalysis.records.length === 0) {
        return null;
      }

      return {
        dataset_id: preparedAnalysis.datasetId,
        records: preparedAnalysis.records,
        attribute_names: preparedAnalysis.factorColumns,
        data: preparedAnalysis.records,
        x: preparedAnalysis.x,
        y: preparedAnalysis.y,
        c: preparedAnalysis.colors,
        s: 4,
        splom_s: 2,
        cmap: "viridis",
        predicate_mode: MODEL_PREDICATE_MODES[predicateMode] || "data extent",
        predicate_options: predicateOptions || {},
        brush_mode: brushMode,
        xticks: 5,
        yticks: 5,
        image_urls: [],
        splom_attributes: preparedAnalysis.factorColumns.slice(
          0,
          Math.min(6, preparedAnalysis.factorColumns.length),
        ),
        selected: [],
        predicates: {},
      };
    },
    [brushMode, predicateMode, predicateOptions, preparedAnalysis],
  );

  useEffect(
    function () {
      if (typeof onMetaChange !== "function") {
        return;
      }

      onMetaChange({
        totalRows: preparedAnalysis ? preparedAnalysis.totalRows : 0,
        validRows: preparedAnalysis ? preparedAnalysis.records.length : 0,
        droppedRows: preparedAnalysis ? preparedAnalysis.droppedRows : 0,
        hasRenderableData: Boolean(spec),
      });
    },
    [onMetaChange, preparedAnalysis, spec],
  );

  useEffect(
    function () {
      if (!hostRef.current) {
        return undefined;
      }

      hostRef.current.innerHTML = "";

      if (!spec) {
        return undefined;
      }

      const model = createLegacyWidgetModel(spec, {
        onPredicates: function (payload) {
          if (typeof callbacksRef.current.onPredicates === "function") {
            callbacksRef.current.onPredicates(payload);
          }
        },
        onError: function (error) {
          if (typeof callbacksRef.current.onError === "function") {
            callbacksRef.current.onError(error);
          }
        },
        onLoadingChange: function (loading) {
          if (typeof callbacksRef.current.onLoadingChange === "function") {
            callbacksRef.current.onLoadingChange(loading);
          }
        },
      });
      legacyWidget.initialize({model: model});
      renderedRef.current = legacyWidget.render({model: model, el: hostRef.current});
      const projectionView = renderedRef.current?.projection_view;
      if (projectionView && typeof projectionView.set_selection_callback === "function") {
        projectionView.set_selection_callback(function (preparedMask) {
          const callback = callbacksRef.current.onSelectionChange;
          if (typeof callback !== "function") {
            return;
          }
          if (!Array.isArray(preparedMask)) {
            callback(null);
            return;
          }

          const sourceMask = new Array(preparedAnalysis.sourceRowCount).fill(false);
          preparedMask.forEach(function (selected, preparedIndex) {
            const sourceIndex = preparedAnalysis.sourceIndices[preparedIndex];
            if (sourceIndex !== undefined) {
              sourceMask[sourceIndex] = Boolean(selected);
            }
          });
          callback(sourceMask);
        });
      }

      return function () {
        renderedRef.current = null;
        model.destroy();
        hostRef.current.innerHTML = "";
      };
    },
    [resetToken, spec],
  );

  useEffect(
    function () {
      const projectionView = renderedRef.current?.projection_view;
      if (!projectionView || typeof projectionView.apply_external_selection !== "function") {
        return;
      }
      const preparedMask = Array.isArray(objectiveSelectionMask)
        ? preparedAnalysis?.sourceIndices.map(function (sourceIndex) {
            return Boolean(objectiveSelectionMask[sourceIndex]);
          })
        : null;
      projectionView.apply_external_selection(preparedMask, objectiveSelectionSource);
    },
    [objectiveSelectionMask, objectiveSelectionSource, preparedAnalysis, resetToken, spec],
  );

  useEffect(
    function () {
      const predicateView = renderedRef.current?.predicate_view;
      if (predicateView && typeof predicateView.set_max_attributes === "function") {
        predicateView.set_max_attributes(predicateFactorLimit);
      }
    },
    [predicateFactorLimit, resetToken, spec],
  );

  useEffect(
    function () {
      const splomView = renderedRef.current?.splom_view;
      if (!splomView) {
        return;
      }
      if (
        Array.isArray(focusedSplomAttributes) &&
        focusedSplomAttributes.length >= 2 &&
        typeof splomView.focus_attributes === "function"
      ) {
        splomView.focus_attributes(focusedSplomAttributes);
      } else if (typeof splomView.clear_attribute_focus === "function") {
        splomView.clear_attribute_focus();
      }
    },
    [focusedSplomAttributes, resetToken, spec],
  );

  return <div className="dimbridge-standalone-host" ref={hostRef} />;
}
