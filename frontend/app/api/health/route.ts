import { NextResponse } from "next/server";

export const runtime = "nodejs";

export async function GET() {
  const backendUrl = process.env.BOT_OR_NOT_BACKEND_URL;

  if (!backendUrl) {
    return NextResponse.json({
      ok: true,
      backendConfigured: false,
      backendReachable: false,
      detail: "BOT_OR_NOT_BACKEND_URL is not set.",
    });
  }

  try {
    const response = await fetch(new URL("/api/health", backendUrl), {
      cache: "no-store",
      signal: AbortSignal.timeout(5_000),
    });
    const backend = await response.json().catch(() => null);

    return NextResponse.json({
      ok: response.ok,
      backendConfigured: true,
      backendReachable: response.ok,
      backendUrl,
      backend,
    });
  } catch (error) {
    return NextResponse.json(
      {
        ok: false,
        backendConfigured: true,
        backendReachable: false,
        backendUrl,
        detail: error instanceof Error ? error.message : "Backend health check failed.",
      },
      { status: 502 },
    );
  }
}
