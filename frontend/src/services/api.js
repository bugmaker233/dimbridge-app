async function parseResponse(response) {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = payload?.detail || payload?.message || "Request failed.";
    throw new Error(message);
  }
  return payload;
}

export async function fetchHealth() {
  const response = await fetch("/api/health");
  return parseResponse(response);
}

export async function uploadDataset(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("/api/dataset/upload", {
    method: "POST",
    body: formData,
  });
  return parseResponse(response);
}

export async function runProjection(body) {
  const response = await fetch("/api/projection/run", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body),
  });
  return parseResponse(response);
}

export async function runDataExtent(body) {
  const response = await fetch("/api/predicate/data-extent", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body),
  });
  return parseResponse(response);
}

export async function runRegression(body) {
  const response = await fetch("/api/predicate/regression", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body),
  });
  return parseResponse(response);
}

export async function runPaperRegression(body) {
  const response = await fetch("/api/predicate/paper-regression", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body),
  });
  return parseResponse(response);
}

export async function runRPI(body) {
  const response = await fetch("/api/predicate/rpi", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body),
  });
  return parseResponse(response);
}
