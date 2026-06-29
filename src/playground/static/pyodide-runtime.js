export class PyodideRuntime {
  constructor(onStatus) {
    this.onStatus = onStatus;
    this.worker = null;
    this.nextId = 1;
    this.pending = new Map();
  }

  execute(request) {
    this.#ensureWorker();
    const id = this.nextId;
    this.nextId += 1;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ id, request });
    });
  }

  dispose() {
    this.worker?.terminate();
    this.worker = null;
    for (const pending of this.pending.values()) {
      pending.reject(new Error("Pyodide worker was terminated"));
    }
    this.pending.clear();
  }

  #ensureWorker() {
    if (this.worker) {
      return;
    }
    this.onStatus("Loading Python sandbox");
    this.worker = new Worker("/pyodide-worker.js", { type: "module" });
    this.worker.addEventListener("message", (event) => {
      const pending = this.pending.get(event.data?.id);
      if (!pending) {
        return;
      }
      this.pending.delete(event.data.id);
      pending.resolve(event.data.response);
    });
    this.worker.addEventListener("error", (event) => {
      const error = new Error(event.message || "Pyodide worker failed");
      for (const pending of this.pending.values()) {
        pending.reject(error);
      }
      this.pending.clear();
      this.worker?.terminate();
      this.worker = null;
    });
  }
}
