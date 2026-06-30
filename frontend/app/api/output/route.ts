import { NextResponse } from "next/server";
import { apiFacts, navItems } from "@/lib/site";

export const runtime = "nodejs";

export function GET() {
  return NextResponse.json({
    app: "Circuit SPICE List Generator",
    pages: navItems,
    api: {
      process: {
        method: "POST",
        path: "/api/process",
        contentType: "multipart/form-data",
        fields: {
          image: "PNG, JPG, WebP, BMP, or TIFF file up to 8 MB",
          domain: "hand-drawn | digital",
        },
      },
      health: {
        method: "GET",
        path: "/api/health",
      },
    },
    outputShape: {
      domain: "string",
      timing: { elapsedMs: "number" },
      weights: { unet: "string", yolo: "string" },
      components: "array",
      ocr: "array",
      nets: "array",
      warnings: "array",
      netlist: "string",
    },
    integration: apiFacts,
  });
}
