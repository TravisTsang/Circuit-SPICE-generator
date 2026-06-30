import { NextResponse } from "next/server";
import { domains } from "@/lib/site";

export const runtime = "nodejs";
export const maxDuration = 60;

const maxUploadBytes = 8 * 1024 * 1024;
const acceptedTypes = new Set([
  "image/png",
  "image/jpeg",
  "image/webp",
  "image/bmp",
  "image/tiff",
]);

export async function POST(request: Request) {
  const backendUrl = process.env.BOT_OR_NOT_BACKEND_URL;
  if (!backendUrl) {
    return NextResponse.json(
      {
        error:
          "The inference backend is not configured. Set BOT_OR_NOT_BACKEND_URL to a running FastAPI service.",
        code: "BACKEND_NOT_CONFIGURED",
        backendConfigured: false,
      },
      { status: 503 },
    );
  }

  let formData: FormData;
  try {
    formData = await request.formData();
  } catch {
    return NextResponse.json(
      { error: "Expected multipart/form-data with image and domain fields." },
      { status: 400 },
    );
  }

  const image = formData.get("image");
  const domain = String(formData.get("domain") ?? "");
  const allowedDomains = new Set<string>(domains.map((item) => item.id));

  if (!(image instanceof File)) {
    return NextResponse.json({ error: "Missing image file." }, { status: 400 });
  }

  if (!acceptedTypes.has(image.type)) {
    return NextResponse.json(
      { error: "Upload must be PNG, JPG, WebP, BMP, or TIFF." },
      { status: 400 },
    );
  }

  if (image.size > maxUploadBytes) {
    return NextResponse.json(
      { error: "Upload must be 8 MB or smaller." },
      { status: 413 },
    );
  }

  if (!allowedDomains.has(domain)) {
    return NextResponse.json(
      { error: "domain must be one of: hand-drawn, digital." },
      { status: 400 },
    );
  }

  const outbound = new FormData();
  outbound.append("image", image, image.name || "circuit.png");
  outbound.append("domain", domain);

  try {
    const response = await fetch(new URL("/api/process", backendUrl), {
      method: "POST",
      body: outbound,
      signal: AbortSignal.timeout(55_000),
    });
    const payload = await response.json().catch(() => ({
      error: `Backend returned status ${response.status}.`,
    }));

    return NextResponse.json(payload, { status: response.status });
  } catch (error) {
    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? `Could not reach the inference backend: ${error.message}`
            : "Could not reach the inference backend.",
        code: "BACKEND_UNREACHABLE",
        backendConfigured: true,
      },
      { status: 502 },
    );
  }
}
