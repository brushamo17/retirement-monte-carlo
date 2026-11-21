// Runs the Python simulator in a background thread, so the page stays
// responsive while thousands of retirements are simulated.
//
// How it works:
//   1. Load Pyodide (Python compiled to WebAssembly) and NumPy from the CDN.
//   2. Download retirement_mc.zip (the repository's own Python package, bundled
//      by scripts/build_web.py) and unpack it inside Pyodide.
//   3. Wait for messages {id, fn, args} from the page, call
//      retirement_mc.web.call(fn, args) and send back {id, result} or {id, error}.
//
// Pyodide 314+ is an ES module, so the page starts this as a module worker:
//   new Worker("pyodide-worker.js", { type: "module" })

import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v314.0.8/full/pyodide.mjs";

const status = (text) => postMessage({ status: text });

async function start() {
  status("Downloading Python (about 10 MB the first time, then cached)…");
  const pyodide = await loadPyodide();
  status("Loading NumPy…");
  await pyodide.loadPackage("numpy", { messageCallback: () => {} });
  status("Loading the simulator…");
  const response = await fetch("retirement_mc.zip", { cache: "no-cache" });
  if (!response.ok) throw new Error(`could not download retirement_mc.zip (HTTP ${response.status})`);
  pyodide.unpackArchive(await response.arrayBuffer(), "zip", { extractDir: "/home/pyodide" });
  pyodide.runPython("import sys; sys.path.insert(0, '/home/pyodide'); from retirement_mc import web");
  return pyodide.globals.get("web").call;   // Python function: call(name: str, json: str) -> str
}

const ready = start();
ready.then(() => postMessage({ ready: true }), (err) => postMessage({ fatal: String(err && err.message || err) }));

onmessage = async ({ data }) => {
  const { id, fn, args } = data;
  try {
    const call = await ready;
    const json = call(fn, JSON.stringify(args || {}));
    postMessage({ id, result: JSON.parse(json) });
  } catch (err) {
    postMessage({ id, error: String(err && err.message || err) });
  }
};
