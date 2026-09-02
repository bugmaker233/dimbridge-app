import {runPaperRegression, runRegression, runRPI} from "../services/api";

const PREDICATE_RUNNERS = {
  "predicate regression": runRegression,
  "paper predicate regression": runPaperRegression,
  "recursive predicate induction": runRPI,
};

function buildChangeEvent(name, model) {
  return {
    name,
    model,
    type: name,
  };
}

export class LegacyWidgetModel {
  constructor(initialValues, {onPredicates, onError, onLoadingChange} = {}) {
    this.values = new Map(Object.entries(initialValues));
    this.listeners = new Map();
    this.onPredicates = onPredicates;
    this.onError = onError;
    this.onLoadingChange = onLoadingChange;
    this.requestVersion = 0;
    this.destroyed = false;
  }

  get(key) {
    return this.values.get(key);
  }

  set(key, value, {silent = false} = {}) {
    this.values.set(key, value);
    if (!silent) {
      this.emit(`change:${key}`, value);
    }
  }

  on(eventName, handler) {
    const handlers = this.listeners.get(eventName) ?? new Set();
    handlers.add(handler);
    this.listeners.set(eventName, handlers);
    return () => {
      handlers.delete(handler);
    };
  }

  emit(eventName, payload) {
    const handlers = this.listeners.get(eventName);
    if (!handlers) {
      return;
    }
    for (const handler of handlers) {
      handler(buildChangeEvent(eventName, this), payload);
    }
  }

  async save_changes() {
    if (this.destroyed) {
      return;
    }

    const predicateMode = this.get("predicate_mode");
    const runPredicate = PREDICATE_RUNNERS[predicateMode];
    if (typeof runPredicate !== "function") {
      return;
    }

    const selectedMasks = this.get("selected");
    if (!Array.isArray(selectedMasks) || selectedMasks.length === 0) {
      return;
    }

    const currentVersion = ++this.requestVersion;
    const selectionContext = this.get("selection_context") || {};
    const requestMeta = {
      request_id: currentVersion,
      source: selectionContext.source || "brush",
      requested_at: selectionContext.requested_at || new Date().toISOString(),
      selected_counts: selectedMasks.map(function (mask) {
        return Array.isArray(mask) ? mask.filter(Boolean).length : 0;
      }),
      total_count: Array.isArray(selectedMasks[0]) ? selectedMasks[0].length : 0,
    };
    if (typeof this.onLoadingChange === "function") {
      this.onLoadingChange(true);
    }

    try {
      const records = this.get("records");
      const datasetId = this.get("dataset_id");
      const hasDatasetId = typeof datasetId === "string" && datasetId.length > 0;
      const predicateOptions = this.get("predicate_options") || {};
      const response = await runPredicate({
        ...predicateOptions,
        dataset_id: hasDatasetId ? datasetId : null,
        records: hasDatasetId ? null : Array.isArray(records) ? records : null,
        attribute_names: this.get("attribute_names"),
        selected_masks: selectedMasks,
      });

      if (this.destroyed || currentVersion !== this.requestVersion) {
        return;
      }

      const payload = {
        columns: response.columns ?? [],
        predicates: response.predicates,
        qualities: response.qualities ?? [],
        algorithm: response.algorithm ?? predicateMode,
        candidate_solutions: response.candidate_solutions ?? null,
        diagnostics: response.diagnostics ?? null,
        event_meta: {...requestMeta, completed_at: new Date().toISOString()},
      };

      this.values.set("predicates", payload);
      if (typeof this.onPredicates === "function") {
        this.onPredicates(payload);
      }
      this.emit("change:predicates", payload);
    } catch (error) {
      if (this.destroyed) {
        return;
      }
      if (typeof this.onError === "function") {
        this.onError(error);
      } else {
        console.error(error);
      }
    } finally {
      if (
        !this.destroyed &&
        currentVersion === this.requestVersion &&
        typeof this.onLoadingChange === "function"
      ) {
        this.onLoadingChange(false);
      }
    }
  }

  destroy() {
    this.destroyed = true;
    if (typeof this.onLoadingChange === "function") {
      this.onLoadingChange(false);
    }
    this.listeners.clear();
  }
}

export function createLegacyWidgetModel(initialValues, options) {
  return new LegacyWidgetModel(initialValues, options);
}
