import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  CircuitBoard,
  Clipboard,
  Download,
  FileImage,
  Loader2,
  Play,
  RotateCcw,
  Upload,
  Zap,
} from "lucide-react";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

const DOMAINS = [
  {
    id: "hand-drawn",
    label: "Hand-Drawn Circuits",
    description: "Noisy scans and sketches",
  },
  {
    id: "digital",
    label: "Digital Circuits",
    description: "CAD exports and screenshots",
  },
];

function App() {
  const [domain, setDomain] = useState(DOMAINS[0].id);
  const [imageFile, setImageFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const fileInputRef = useRef(null);

  const selectedDomain = useMemo(
    () => DOMAINS.find((item) => item.id === domain) ?? DOMAINS[0],
    [domain],
  );

  const setImage = useCallback((file) => {
    if (!file || !file.type.startsWith("image/")) {
      setError("Select an image file.");
      return;
    }
    setImageFile(file);
    setResult(null);
    setError("");
    setPreviewUrl((oldUrl) => {
      if (oldUrl) URL.revokeObjectURL(oldUrl);
      return URL.createObjectURL(file);
    });
  }, []);

  useEffect(() => {
    const onPaste = (event) => {
      const items = Array.from(event.clipboardData?.items ?? []);
      const imageItem = items.find((item) => item.type.startsWith("image/"));
      if (!imageItem) return;
      const file = imageItem.getAsFile();
      if (file) {
        setImage(
          new File([file], `pasted-circuit-${Date.now()}.png`, {
            type: file.type || "image/png",
          }),
        );
      }
    };

    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [setImage]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);
    const file = event.dataTransfer.files?.[0];
    if (file) setImage(file);
  };

  const handleProcess = async () => {
    if (!imageFile) {
      setError("Add a circuit image first.");
      return;
    }

    const formData = new FormData();
    formData.append("image", imageFile);
    formData.append("domain", domain);

    setIsProcessing(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_BASE}/api/process`, {
        method: "POST",
        body: formData,
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.detail || `Request failed with ${response.status}`);
      }
      setResult(payload);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const resetImage = () => {
    setImageFile(null);
    setResult(null);
    setError("");
    setPreviewUrl((oldUrl) => {
      if (oldUrl) URL.revokeObjectURL(oldUrl);
      return "";
    });
  };

  const downloadNetlist = () => {
    if (!result?.netlist) return;
    const blob = new Blob([result.netlist], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "circuit.cir";
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <main className="min-h-screen bg-[#eef1ec] px-4 py-5 text-ink sm:px-6 lg:px-8">
      <div className="mx-auto flex max-w-7xl flex-col gap-5">
        <header className="flex flex-col gap-4 border-b border-line pb-4 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex items-center gap-3">
            <div className="grid h-11 w-11 place-items-center rounded-md bg-ink text-white">
              <CircuitBoard className="h-6 w-6" aria-hidden="true" />
            </div>
            <div>
              <h1 className="text-2xl font-semibold tracking-normal">Circuits OCR</h1>
              <p className="text-sm text-ink/65">Image to SPICE inference console</p>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
            {DOMAINS.map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => setDomain(item.id)}
                className={`rounded-md border px-4 py-3 text-left transition ${
                  domain === item.id
                    ? "border-accent bg-white shadow-sm"
                    : "border-line bg-panel hover:border-ink/30"
                }`}
                aria-pressed={domain === item.id}
              >
                <span className="block text-sm font-semibold">{item.label}</span>
                <span className="block text-xs text-ink/60">{item.description}</span>
              </button>
            ))}
          </div>
        </header>

        <section className="grid gap-5 lg:grid-cols-[minmax(0,1.15fr)_minmax(360px,0.85fr)]">
          <div className="rounded-md border border-line bg-white p-4 shadow-tool">
            <div
              role="button"
              tabIndex={0}
              onClick={() => fileInputRef.current?.click()}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  fileInputRef.current?.click();
                }
              }}
              onDragOver={(event) => {
                event.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              className={`flex min-h-[520px] flex-col overflow-hidden rounded-md border border-dashed transition ${
                isDragging ? "border-accent bg-accent/5" : "border-line bg-panel"
              }`}
            >
              {previewUrl ? (
                <div className="flex h-full min-h-[520px] flex-col">
                  <div className="flex items-center justify-between border-b border-line bg-white px-4 py-3">
                    <div className="flex min-w-0 items-center gap-2">
                      <FileImage className="h-5 w-5 shrink-0 text-accent" aria-hidden="true" />
                      <span className="truncate text-sm font-medium">
                        {imageFile?.name ?? "circuit image"}
                      </span>
                    </div>
                    <button
                      type="button"
                      onClick={(event) => {
                        event.stopPropagation();
                        resetImage();
                      }}
                      className="grid h-9 w-9 place-items-center rounded-md border border-line bg-white text-ink/70 hover:text-ink"
                      title="Reset image"
                    >
                      <RotateCcw className="h-4 w-4" aria-hidden="true" />
                    </button>
                  </div>
                  <div className="grid flex-1 place-items-center p-3">
                    <img
                      src={previewUrl}
                      alt="Circuit preview"
                      className="max-h-[680px] max-w-full rounded-sm object-contain"
                    />
                  </div>
                </div>
              ) : (
                <div className="grid flex-1 place-items-center p-8 text-center">
                  <div className="flex max-w-sm flex-col items-center gap-4">
                    <div className="grid h-16 w-16 place-items-center rounded-md border border-line bg-white">
                      <Clipboard className="h-8 w-8 text-accent" aria-hidden="true" />
                    </div>
                    <div>
                      <p className="text-lg font-semibold">Paste or drop a circuit image</p>
                      <p className="mt-1 text-sm text-ink/60">
                        PNG, JPG, WebP, BMP, or TIFF
                      </p>
                    </div>
                    <button
                      type="button"
                      className="inline-flex items-center gap-2 rounded-md bg-ink px-4 py-2.5 text-sm font-semibold text-white hover:bg-ink/90"
                    >
                      <Upload className="h-4 w-4" aria-hidden="true" />
                      Select Image
                    </button>
                  </div>
                </div>
              )}
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) setImage(file);
                  event.target.value = "";
                }}
              />
            </div>
          </div>

          <aside className="flex flex-col gap-4">
            <div className="rounded-md border border-line bg-white p-4 shadow-tool">
              <div className="mb-4 flex items-center justify-between gap-3">
                <div>
                  <h2 className="text-base font-semibold">Inference</h2>
                  <p className="text-sm text-ink/60">{selectedDomain.label}</p>
                </div>
                <div className="grid h-10 w-10 place-items-center rounded-md bg-accent/10 text-accent">
                  <Zap className="h-5 w-5" aria-hidden="true" />
                </div>
              </div>

              <button
                type="button"
                onClick={handleProcess}
                disabled={!imageFile || isProcessing}
                className="inline-flex h-11 w-full items-center justify-center gap-2 rounded-md bg-accent px-4 text-sm font-semibold text-white transition hover:bg-accent/90 disabled:cursor-not-allowed disabled:bg-ink/25"
              >
                {isProcessing ? (
                  <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
                ) : (
                  <Play className="h-4 w-4" aria-hidden="true" />
                )}
                {isProcessing ? "Processing" : "Process Circuit"}
              </button>

              {error ? (
                <div className="mt-4 rounded-md border border-signal/30 bg-signal/10 p-3 text-sm text-signal">
                  <div className="flex gap-2">
                    <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
                    <span>{error}</span>
                  </div>
                </div>
              ) : null}
            </div>

            <ResultsPanel result={result} onDownloadNetlist={downloadNetlist} />
          </aside>
        </section>
      </div>
    </main>
  );
}

function ResultsPanel({ result, onDownloadNetlist }) {
  if (!result) {
    return (
      <div className="rounded-md border border-line bg-white p-4 shadow-tool">
        <h2 className="text-base font-semibold">Results</h2>
        <p className="mt-2 text-sm text-ink/60">Component boxes, OCR, nets, and SPICE output appear here.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="rounded-md border border-line bg-white p-4 shadow-tool">
        <div className="mb-3 flex items-center justify-between gap-3">
          <h2 className="text-base font-semibold">Results</h2>
          <span className="rounded-sm bg-panel px-2 py-1 text-xs text-ink/70">
            {result.timing?.elapsedMs ?? 0} ms
          </span>
        </div>

        <div className="grid grid-cols-3 gap-2">
          <Metric label="Components" value={result.components?.length ?? 0} />
          <Metric label="OCR" value={result.ocr?.length ?? 0} />
          <Metric label="Nets" value={result.nets?.length ?? 0} />
        </div>

        {result.warnings?.length ? (
          <div className="mt-4 rounded-md border border-signal/30 bg-signal/10 p-3">
            <p className="text-sm font-semibold text-signal">Warnings</p>
            <ul className="mt-2 space-y-1 text-sm text-signal">
              {result.warnings.slice(0, 5).map((warning, index) => (
                <li key={`${warning}-${index}`}>{warning}</li>
              ))}
            </ul>
          </div>
        ) : null}
      </div>

      <div className="rounded-md border border-line bg-white p-4 shadow-tool">
        <h3 className="mb-3 text-sm font-semibold">Components</h3>
        <div className="result-scroll max-h-64 overflow-auto rounded-md border border-line">
          <table className="w-full border-collapse text-left text-sm">
            <thead className="sticky top-0 bg-panel text-xs uppercase text-ink/60">
              <tr>
                <th className="px-3 py-2 font-semibold">Name</th>
                <th className="px-3 py-2 font-semibold">Class</th>
                <th className="px-3 py-2 font-semibold">Value</th>
                <th className="px-3 py-2 font-semibold">Box</th>
                <th className="px-3 py-2 font-semibold">Nets</th>
              </tr>
            </thead>
            <tbody>
              {(result.components ?? []).map((component) => (
                <tr key={component.uid} className="border-t border-line">
                  <td className="px-3 py-2 font-medium">{component.name}</td>
                  <td className="px-3 py-2 text-ink/70">{component.className}</td>
                  <td className="px-3 py-2 text-ink/70">{component.value ?? "-"}</td>
                  <td className="px-3 py-2 text-ink/70">
                    {formatBox(component.bbox)}
                  </td>
                  <td className="px-3 py-2 text-ink/70">
                    {(component.terminalNets ?? []).join(", ") || "-"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="rounded-md border border-line bg-white p-4 shadow-tool">
        <div className="mb-3 flex items-center justify-between gap-3">
          <h3 className="text-sm font-semibold">SPICE Netlist</h3>
          <button
            type="button"
            onClick={onDownloadNetlist}
            className="grid h-8 w-8 place-items-center rounded-md border border-line hover:bg-panel"
            title="Download netlist"
          >
            <Download className="h-4 w-4" aria-hidden="true" />
          </button>
        </div>
        <pre className="result-scroll max-h-80 overflow-auto rounded-md bg-ink p-3 text-xs leading-relaxed text-white">
          {result.netlist}
        </pre>
      </div>
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div className="rounded-md border border-line bg-panel px-3 py-2">
      <p className="text-xs text-ink/60">{label}</p>
      <p className="text-xl font-semibold">{value}</p>
    </div>
  );
}

function formatBox(bbox) {
  if (!bbox) return "-";
  const x1 = Math.round(bbox.x1);
  const y1 = Math.round(bbox.y1);
  const x2 = Math.round(bbox.x2);
  const y2 = Math.round(bbox.y2);
  return `${x1},${y1} ${x2},${y2}`;
}

export default App;
