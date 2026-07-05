import { createReadStream } from "node:fs";
import { createServer } from "node:http";
import { basename, resolve } from "node:path";
import process from "node:process";

import { loadPyodide } from "pyodide";

const wheelPath = resolve(process.argv[2] ?? "");
const wheelName = basename(wheelPath);

if (!wheelName.endsWith(".whl")) {
  throw new Error("Pass the built pydantic-monty wheel path");
}

const server = createServer((request, response) => {
  if (request.url !== `/${wheelName}`) {
    response.writeHead(404).end();
    return;
  }
  response.writeHead(200, {
    "Content-Type": "application/zip",
    "Access-Control-Allow-Origin": "*",
  });
  createReadStream(wheelPath).pipe(response);
});

await new Promise((resolveReady, reject) => {
  server.once("error", reject);
  server.listen(0, "127.0.0.1", resolveReady);
});

try {
  const address = server.address();
  if (!address || typeof address === "string") {
    throw new Error("Monty wheel smoke-test server did not bind");
  }

  const pyodide = await loadPyodide();
  await pyodide.loadPackage(["micropip", "pydantic", "pygments"]);
  pyodide.globals.set(
    "__monty_wheel_url",
    `http://127.0.0.1:${address.port}/${wheelName}`,
  );
  await pyodide.runPythonAsync(`
import micropip

await micropip.install([
    "protobuf==6.33.5",
    "opentelemetry-api==1.41.1",
    "opentelemetry-sdk==1.41.1",
    "opentelemetry-exporter-otlp-proto-http==1.41.1",
    "opentelemetry-instrumentation==0.62b1",
    "opentelemetry-semantic-conventions==0.62b1",
    "logfire==4.33.0",
    __monty_wheel_url,
])

import logfire
import pydantic_monty

sandbox = pydantic_monty.Monty("left + right", inputs=["left", "right"])
assert sandbox.run(inputs={"left": 20, "right": 22}) == 42
assert pydantic_monty.__version__ == "0.0.18"
assert logfire.__version__ == "4.33.0"
`);
} finally {
  server.closeAllConnections();
  await new Promise((resolveClosed) => server.close(resolveClosed));
}
