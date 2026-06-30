import { ImageResponse } from "next/og";

export const runtime = "edge";
export const alt = "Circuit SPICE List Generator technical circuit interface";
export const size = {
  width: 1200,
  height: 630,
};
export const contentType = "image/png";

export default function OpenGraphImage() {
  return new ImageResponse(
    (
      <div
        style={{
          alignItems: "stretch",
          background: "#06111b",
          color: "#eff8fb",
          display: "flex",
          flexDirection: "column",
          fontFamily: "Arial, sans-serif",
          height: "100%",
          justifyContent: "space-between",
          padding: 64,
          position: "relative",
          width: "100%",
        }}
      >
        <div
          style={{
            background:
              "linear-gradient(90deg, rgba(77, 225, 232, 0.18) 1px, transparent 1px), linear-gradient(0deg, rgba(77, 225, 232, 0.14) 1px, transparent 1px)",
            backgroundSize: "48px 48px",
            bottom: 0,
            left: 0,
            opacity: 0.55,
            position: "absolute",
            right: 0,
            top: 0,
          }}
        />
        <div style={{ display: "flex", justifyContent: "space-between", position: "relative" }}>
          <div
            style={{
              border: "1px solid rgba(77, 225, 232, 0.55)",
              borderRadius: 12,
              color: "#75f3e8",
              display: "flex",
              fontSize: 28,
              padding: "12px 18px",
            }}
          >
            image to SPICE
          </div>
          <div style={{ color: "#8fb1c6", display: "flex", fontSize: 24 }}>
            FastAPI / Next.js
          </div>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 26, position: "relative" }}>
          <div style={{ display: "flex", fontSize: 76, fontWeight: 700, lineHeight: 1.02 }}>
            Circuit SPICE
            <br />
            List Generator
          </div>
          <div style={{ color: "#a7c4d5", display: "flex", fontSize: 30, maxWidth: 830 }}>
            Paste a schematic image and inspect the generated circuit netlist.
          </div>
        </div>
        <div
          style={{
            alignItems: "center",
            display: "flex",
            gap: 22,
            position: "relative",
          }}
        >
          {["trace mask", "components", "nets", ".cir export"].map((item) => (
            <div
              key={item}
              style={{
                background: "rgba(13, 31, 47, 0.92)",
                border: "1px solid rgba(77, 225, 232, 0.35)",
                borderRadius: 10,
                color: "#d7eef6",
                display: "flex",
                fontSize: 24,
                padding: "16px 20px",
              }}
            >
              {item}
            </div>
          ))}
        </div>
      </div>
    ),
    size,
  );
}
