import { Activity, Binary, Cpu, FileCode2 } from "lucide-react";
import { sampleNetlist } from "@/lib/site";

const readouts = [
  { label: "Trace mask", value: "online", icon: Activity },
  { label: "Symbol pass", value: "YOLOv8", icon: Cpu },
  { label: "Net solver", value: "NetworkX", icon: Binary },
  { label: "Export", value: ".cir", icon: FileCode2 },
];

export function TechSignalPanel() {
  return (
    <div className="relative min-h-[340px] overflow-hidden rounded-lg border border-line/70 bg-background/58 p-4">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_20%,oklch(var(--accent)/0.16),transparent_32%),radial-gradient(circle_at_80%_70%,oklch(var(--signal)/0.12),transparent_34%)]" />
      <div className="relative flex h-full min-h-[308px] flex-col justify-between gap-5">
        <div className="flex items-center justify-between gap-3">
          <div>
            <p className="text-xs font-medium text-accent">signal map</p>
            <h2 className="mt-1 text-xl font-semibold text-ink">Circuit extraction view</h2>
          </div>
          <span className="rounded-md border border-accent/50 bg-accent/10 px-3 py-1.5 font-mono text-xs text-accent">
            SPICE_READY
          </span>
        </div>

        <svg
          className="h-44 w-full overflow-visible"
          viewBox="0 0 680 220"
          role="img"
          aria-label="Animated circuit signal paths showing schematic analysis"
        >
          <defs>
            <filter id="signalGlow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>
          <path
            className="signal-path"
            d="M28 112 H126 V58 H232 V96 H326 V38 H466 V82 H634"
            fill="none"
            filter="url(#signalGlow)"
            stroke="oklch(var(--accent))"
            strokeLinecap="round"
            strokeWidth="3"
          />
          <path
            className="signal-path"
            d="M52 168 H180 V132 H292 V174 H418 V124 H520 V164 H648"
            fill="none"
            filter="url(#signalGlow)"
            stroke="oklch(var(--signal))"
            strokeLinecap="round"
            strokeWidth="3"
          />
          <path
            d="M102 112 V168 M232 58 V132 M326 96 V174 M466 38 V124 M520 82 V164"
            fill="none"
            stroke="oklch(var(--line))"
            strokeLinecap="round"
            strokeWidth="2"
          />
          {[28, 126, 232, 326, 466, 634].map((x, index) => (
            <circle
              key={`top-${x}`}
              className="signal-node origin-center"
              cx={x}
              cy={[112, 112, 58, 96, 38, 82][index]}
              fill="oklch(var(--accent))"
              r="6"
            />
          ))}
          {[52, 180, 292, 418, 520, 648].map((x, index) => (
            <circle
              key={`bottom-${x}`}
              className="signal-node origin-center"
              cx={x}
              cy={[168, 168, 132, 174, 124, 164][index]}
              fill="oklch(var(--signal))"
              r="5"
            />
          ))}
        </svg>

        <div className="grid gap-2 sm:grid-cols-2">
          {readouts.map((item) => {
            const Icon = item.icon;
            return (
              <div key={item.label} className="tech-readout rounded-md border border-line/70 p-3">
                <div className="flex items-center gap-2">
                  <Icon className="h-4 w-4 text-accent" aria-hidden="true" />
                  <span className="text-xs text-muted">{item.label}</span>
                </div>
                <p className="mt-2 font-mono text-sm text-ink">{item.value}</p>
              </div>
            );
          })}
        </div>

        <pre className="code-scroll max-h-28 overflow-auto rounded-md border border-line/70 bg-background/80 p-3 font-mono text-xs leading-5 text-ink">
          {sampleNetlist}
        </pre>
      </div>
    </div>
  );
}
