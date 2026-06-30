"use client";

/* eslint-disable @next/next/no-img-element */

import {
  AlertTriangle,
  CheckCircle2,
  Clipboard,
  Copy,
  Download,
  FileImage,
  Loader2,
  Play,
  RotateCcw,
  Upload,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { domains, sampleNetlist, type CircuitDomain } from "@/lib/site";

type BoundingBox = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  width?: number;
  height?: number;
};

type ComponentResult = {
  uid?: string;
  name?: string;
  className?: string;
  spicePrefix?: string;
  confidence?: number;
  value?: string | null;
  bbox?: BoundingBox;
  terminalNets?: string[];
};

type ProcessResult = {
  domain?: string;
  filename?: string;
  timing?: {
    elapsedMs?: number;
    modelsLoadedAt?: number;
  };
  weights?: {
    unet?: string;
    yolo?: string;
  };
  components?: ComponentResult[];
  ocr?: Array<{ text: string; confidence?: number; bbox?: BoundingBox }>;
  nets?: Array<{ label: number; name: string }>;
  warnings?: string[];
  netlist?: string;
};

type HealthPayload = {
  ok: boolean;
  backendConfigured: boolean;
  backendReachable: boolean;
  backendUrl?: string;
  detail?: string;
};

const acceptedTypes = new Set([
  "image/png",
  "image/jpeg",
  "image/webp",
  "image/bmp",
  "image/tiff",
]);

const maxUploadBytes = 8 * 1024 * 1024;

export function DemoConsole({ compact = false }: { compact?: boolean }) {
  const [domain, setDomain] = useState<CircuitDomain>("hand-drawn");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<ProcessResult | null>(null);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);
  const [health, setHealth] = useState<HealthPayload | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const selectedDomain = useMemo(
    () => domains.find((item) => item.id === domain) ?? domains[0],
    [domain],
  );

  const setImage = useCallback((file: File) => {
    if (!acceptedTypes.has(file.type)) {
      setError("Upload a PNG, JPG, WebP, BMP, or TIFF circuit image.");
      return;
    }
    if (file.size > maxUploadBytes) {
      setError("Keep uploads under 8 MB for the web demo.");
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
    fetch("/api/health", { cache: "no-store" })
      .then((response) => response.json())
      .then((payload: HealthPayload) => setHealth(payload))
      .catch(() =>
        setHealth({
          ok: false,
          backendConfigured: false,
          backendReachable: false,
          detail: "Could not read the local API status.",
        }),
      );
  }, []);

  useEffect(() => {
    const onPaste = (event: ClipboardEvent) => {
      const imageItem = Array.from(event.clipboardData?.items ?? []).find((item) =>
        item.type.startsWith("image/"),
      );
      const file = imageItem?.getAsFile();
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

  const resetImage = () => {
    setImageFile(null);
    setResult(null);
    setError("");
    setPreviewUrl((oldUrl) => {
      if (oldUrl) URL.revokeObjectURL(oldUrl);
      return "";
    });
  };

  const handleProcess = async () => {
    if (!imageFile) {
      setError("Paste, drop, or select a circuit image first.");
      return;
    }

    const formData = new FormData();
    formData.append("image", imageFile);
    formData.append("domain", domain);

    setIsProcessing(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch("/api/process", {
        method: "POST",
        body: formData,
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        const message =
          typeof payload.error === "string"
            ? payload.error
            : `Request failed with status ${response.status}.`;
        throw new Error(message);
      }
      setResult(payload as ProcessResult);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Inference failed.");
    } finally {
      setIsProcessing(false);
    }
  };

  const downloadNetlist = () => {
    const text = result?.netlist ?? sampleNetlist;
    const blob = new Blob([text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "circuit-spice-list.cir";
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const copyNetlist = async () => {
    await navigator.clipboard.writeText(result?.netlist ?? sampleNetlist);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  };

  return (
    <section
      aria-label="Image to SPICE demo console"
      className={`panel overflow-hidden rounded-lg ${compact ? "" : "min-h-[680px]"}`}
    >
      <div className="grid gap-0 lg:grid-cols-[minmax(0,1.05fr)_minmax(320px,0.95fr)]">
        <div className="border-b border-line/70 p-4 sm:p-5 lg:border-b-0 lg:border-r">
          <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
            <div>
              <h2 className="text-lg font-semibold text-ink">Circuit to SPICE List</h2>
              <p className="mt-1 max-w-xl text-sm leading-6 text-muted">
                Drop a schematic, paste from the clipboard, or choose a local image.
              </p>
            </div>
            <BackendBadge health={health} />
          </div>

          <div className="mb-4 grid grid-cols-1 gap-2 sm:grid-cols-2">
            {domains.map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => setDomain(item.id)}
                className={`motion-control focus-ring min-h-14 rounded-md border px-3 py-2 text-left ${
                  domain === item.id
                    ? "border-accent bg-accent/12 text-ink"
                    : "border-line/75 bg-background/35 text-muted hover:border-muted/70 hover:text-ink"
                }`}
                aria-pressed={domain === item.id}
              >
                <span className="block text-sm font-semibold">{item.label}</span>
                <span className="block text-xs leading-5">{item.description}</span>
              </button>
            ))}
          </div>

          <div
            role="button"
            tabIndex={0}
            onClick={() => fileInputRef.current?.click()}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                fileInputRef.current?.click();
              }
            }}
            onDragOver={(event) => {
              event.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={(event) => {
              event.preventDefault();
              setIsDragging(false);
              const file = event.dataTransfer.files?.[0];
              if (file) setImage(file);
            }}
            className={`focus-ring flex min-h-[310px] cursor-pointer flex-col overflow-hidden rounded-md border border-dashed transition ${
              isDragging
                ? "border-accent bg-accent/12"
                : "border-line/80 bg-background/40 hover:border-muted"
            }`}
            aria-label="Upload or paste a circuit image"
          >
            {previewUrl ? (
              <div className="flex min-h-[310px] flex-col">
                <div className="flex items-center justify-between gap-3 border-b border-line/70 bg-raised/60 px-3 py-2">
                  <div className="flex min-w-0 items-center gap-2">
                    <FileImage className="h-4 w-4 shrink-0 text-accent" aria-hidden="true" />
                    <span className="truncate text-sm text-ink">
                      {imageFile?.name ?? "circuit image"}
                    </span>
                  </div>
                  <button
                    type="button"
                    onClick={(event) => {
                      event.stopPropagation();
                      resetImage();
                    }}
                    className="motion-control focus-ring grid h-9 w-9 shrink-0 place-items-center rounded-md border border-line bg-background/60 text-muted hover:text-ink"
                    title="Reset image"
                    aria-label="Reset image"
                  >
                    <RotateCcw className="h-4 w-4" aria-hidden="true" />
                  </button>
                </div>
                <div className="grid flex-1 place-items-center p-3">
                  <img
                    src={previewUrl}
                    alt="Selected circuit preview"
                    className="max-h-[420px] max-w-full rounded-sm object-contain"
                  />
                </div>
              </div>
            ) : (
              <div className="grid flex-1 place-items-center p-6 text-center">
                <div className="max-w-sm">
                  <div className="mx-auto grid h-14 w-14 place-items-center rounded-md border border-line bg-raised text-accent">
                    <Clipboard className="h-7 w-7" aria-hidden="true" />
                  </div>
                  <p className="mt-4 text-base font-semibold text-ink">Paste or drop a circuit image</p>
                  <p className="mt-2 text-sm leading-6 text-muted">
                    The demo accepts PNG, JPG, WebP, BMP, or TIFF files up to 8 MB.
                  </p>
                  <span className="motion-control mt-4 inline-flex min-h-11 items-center gap-2 rounded-md bg-accent px-4 text-sm font-semibold text-background">
                    <Upload className="h-4 w-4" aria-hidden="true" />
                    <span>Select image</span>
                  </span>
                </div>
              </div>
            )}
            <input
              ref={fileInputRef}
              type="file"
              accept="image/png,image/jpeg,image/webp,image/bmp,image/tiff"
              className="hidden"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) setImage(file);
                event.target.value = "";
              }}
            />
          </div>

          <div className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-center">
            <button
              type="button"
              onClick={handleProcess}
              disabled={!imageFile || isProcessing}
              className="motion-control focus-ring inline-flex min-h-11 items-center justify-center gap-2 rounded-md bg-accent px-4 text-sm font-semibold text-background hover:bg-accent/90 disabled:cursor-not-allowed disabled:bg-line disabled:text-muted"
            >
              {isProcessing ? (
                <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
              ) : (
                <Play className="h-4 w-4" aria-hidden="true" />
              )}
              <span>{isProcessing ? "Processing" : `Process ${selectedDomain.label}`}</span>
            </button>
            <p className="text-sm text-muted" aria-live="polite">
              {imageFile ? "Ready to send through /api/process." : "Clipboard paste works anywhere on this page."}
            </p>
          </div>

          {error ? (
            <div
              className="mt-4 rounded-md border border-danger/60 bg-danger/12 p-3 text-sm text-ink"
              role="alert"
            >
              <div className="flex gap-2">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-danger" aria-hidden="true" />
                <span>{error}</span>
              </div>
            </div>
          ) : null}
        </div>

        <div className="flex min-h-[460px] flex-col p-4 sm:p-5">
          <div className="mb-4 flex items-start justify-between gap-3">
            <div>
              <h3 className="text-base font-semibold text-ink">SPICE output</h3>
              <p className="mt-1 text-sm text-muted">
                {result ? "Generated by the backend pipeline." : "A sample output is shown until inference runs."}
              </p>
            </div>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={copyNetlist}
                className="motion-control focus-ring grid h-10 w-10 place-items-center rounded-md border border-line bg-background/45 text-muted hover:text-ink"
                aria-label="Copy netlist"
                title="Copy netlist"
              >
                {copied ? <CheckCircle2 className="h-4 w-4 text-accent" aria-hidden="true" /> : <Copy className="h-4 w-4" aria-hidden="true" />}
              </button>
              <button
                type="button"
                onClick={downloadNetlist}
                className="motion-control focus-ring grid h-10 w-10 place-items-center rounded-md border border-line bg-background/45 text-muted hover:text-ink"
                aria-label="Download netlist"
                title="Download netlist"
              >
                <Download className="h-4 w-4" aria-hidden="true" />
              </button>
            </div>
          </div>

          <pre className="code-scroll min-h-[220px] flex-1 overflow-auto rounded-md border border-line/70 bg-background/72 p-4 font-mono text-xs leading-6 text-ink">
            {result?.netlist ?? sampleNetlist}
          </pre>

          <ResultDetails result={result} />
        </div>
      </div>
    </section>
  );
}

function BackendBadge({ health }: { health: HealthPayload | null }) {
  if (!health) {
    return (
      <span className="inline-flex min-h-9 items-center rounded-md border border-line bg-background/45 px-3 text-xs font-medium text-muted">
        Checking API
      </span>
    );
  }

  const ok = health.backendReachable;
  return (
    <span
      className={`inline-flex min-h-9 items-center rounded-md border px-3 text-xs font-medium ${
        ok
          ? "border-accent/60 bg-accent/12 text-ink"
          : "border-warning/60 bg-warning/12 text-ink"
      }`}
      title={health.detail}
    >
      {ok ? "Backend online" : health.backendConfigured ? "Backend unreachable" : "Local API only"}
    </span>
  );
}

function ResultDetails({ result }: { result: ProcessResult | null }) {
  if (!result) {
    return (
      <div className="mt-4 rounded-md border border-line/70 bg-background/36 p-3">
        <p className="text-sm font-medium text-ink">Empty state</p>
        <p className="mt-1 text-sm leading-6 text-muted">
          Once the backend is configured, detected components, OCR spans, nets, warnings, and timing appear here.
        </p>
      </div>
    );
  }

  const components = result.components ?? [];
  const warnings = result.warnings ?? [];

  return (
    <div className="mt-4 space-y-4">
      <div className="grid grid-cols-3 gap-2">
        <Metric label="Components" value={components.length} />
        <Metric label="OCR spans" value={result.ocr?.length ?? 0} />
        <Metric label="Nets" value={result.nets?.length ?? 0} />
      </div>

      {warnings.length ? (
        <div className="rounded-md border border-warning/60 bg-warning/12 p-3">
          <p className="text-sm font-semibold text-ink">Warnings</p>
          <ul className="mt-2 space-y-1 text-sm leading-6 text-muted">
            {warnings.slice(0, 5).map((warning, index) => (
              <li key={`${warning}-${index}`}>{warning}</li>
            ))}
          </ul>
        </div>
      ) : null}

      <div className="code-scroll max-h-56 overflow-auto rounded-md border border-line/70">
        <table className="w-full min-w-[520px] border-collapse text-left text-sm">
          <thead className="sticky top-0 bg-raised text-xs text-muted">
            <tr>
              <th className="px-3 py-2 font-semibold">Name</th>
              <th className="px-3 py-2 font-semibold">Class</th>
              <th className="px-3 py-2 font-semibold">Value</th>
              <th className="px-3 py-2 font-semibold">Nets</th>
            </tr>
          </thead>
          <tbody>
            {components.length ? (
              components.map((component, index) => (
                <tr key={component.uid ?? `${component.name}-${index}`} className="border-t border-line/70">
                  <td className="px-3 py-2 font-medium text-ink">{component.name ?? "-"}</td>
                  <td className="px-3 py-2 text-muted">{component.className ?? "-"}</td>
                  <td className="px-3 py-2 text-muted">{component.value ?? "-"}</td>
                  <td className="px-3 py-2 text-muted">
                    {(component.terminalNets ?? []).join(", ") || "-"}
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td className="px-3 py-4 text-muted" colSpan={4}>
                  No components were returned by the backend.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-md border border-line/70 bg-background/36 px-3 py-2">
      <p className="text-xs text-muted">{label}</p>
      <p className="mt-1 text-xl font-semibold text-ink">{value}</p>
    </div>
  );
}
