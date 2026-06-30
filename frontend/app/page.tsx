import type { Metadata } from "next";
import Link from "next/link";
import {
  ArrowRight,
  BrainCircuit,
  CircuitBoard,
  FileCode2,
  Mail,
  ScanSearch,
  ShieldCheck,
  Waypoints,
} from "lucide-react";
import { TechSignalPanel } from "@/components/tech-signal-panel";
import { contactLinks, pipelineStages } from "@/lib/site";

export const metadata: Metadata = {
  title: "Circuit SPICE List Generator",
  description:
    "A fast project site and demo for converting schematic images into SPICE netlists.",
};

const homeFeatures = [
  {
    title: "Image-first input",
    detail:
      "Start from a pasted schematic, scan, or exported circuit image without manually redrawing the circuit.",
    icon: ScanSearch,
  },
  {
    title: "Model-assisted extraction",
    detail:
      "Wire segmentation, component detection, and OCR work together before topology is recovered.",
    icon: BrainCircuit,
  },
  {
    title: "Inspectable output",
    detail:
      "The result is a SPICE-style list with components, values, named nets, and warnings kept visible.",
    icon: FileCode2,
  },
];

const outputDetails = [
  "Detected components and symbol classes",
  "OCR labels and nearby value matches",
  "Skeletonized wires grouped into nets",
  "SPICE lines ready for review",
];

export default function HomePage() {
  return (
    <main id="main-content">
      <section className="mx-auto grid w-full max-w-7xl items-center gap-10 px-4 py-14 sm:px-6 sm:py-16 lg:grid-cols-[0.92fr_1.08fr] lg:px-8 lg:py-20">
        <div className="max-w-2xl">
          <p className="mb-4 inline-flex rounded-md border border-accent/45 bg-accent/10 px-3 py-1 text-sm font-medium text-accent">
            Circuit schematic to SPICE list
          </p>
          <h1 className="text-balance text-5xl font-semibold leading-tight text-ink sm:text-6xl">
            Generate a SPICE list from a circuit image.
          </h1>
          <p className="text-pretty mt-5 max-w-xl text-lg leading-8 text-muted">
            Circuit SPICE List Generator is a project site and demo for turning
            schematic images into connected components, named nets, and SPICE
            output that can be inspected before use.
          </p>
          <div className="mt-7 flex flex-col gap-3 sm:flex-row">
            <Link
              href="/demo"
              className="motion-control focus-ring inline-flex min-h-11 items-center justify-center gap-2 rounded-md bg-accent px-4 text-sm font-semibold text-background hover:bg-accent/90"
            >
              <span>Try the demo</span>
              <ArrowRight className="h-4 w-4" aria-hidden="true" />
            </Link>
            <Link
              href="/information"
              className="motion-control focus-ring inline-flex min-h-11 items-center justify-center rounded-md border border-line bg-surface/75 px-4 text-sm font-semibold text-ink hover:bg-raised"
            >
              <span>How it works</span>
            </Link>
          </div>

          <dl className="mt-10 grid gap-4 border-y border-line/70 py-5 sm:grid-cols-3">
            <div>
              <dt className="text-xs font-medium text-muted">Input</dt>
              <dd className="mt-1 text-sm font-semibold text-ink">Schematic image</dd>
            </div>
            <div>
              <dt className="text-xs font-medium text-muted">Analysis</dt>
              <dd className="mt-1 text-sm font-semibold text-ink">U-Net, YOLOv8, OCR</dd>
            </div>
            <div>
              <dt className="text-xs font-medium text-muted">Output</dt>
              <dd className="mt-1 text-sm font-semibold text-ink">SPICE netlist</dd>
            </div>
          </dl>
        </div>

        <TechSignalPanel />
      </section>

      <section className="border-y border-line/70 bg-background/45">
        <div className="mx-auto grid max-w-7xl gap-8 px-4 py-12 sm:px-6 lg:grid-cols-[0.7fr_1.3fr] lg:px-8">
          <div>
            <div className="grid h-11 w-11 place-items-center rounded-md border border-signal/45 bg-signal/10 text-signal">
              <CircuitBoard className="h-5 w-5" aria-hidden="true" />
            </div>
            <h2 className="mt-5 text-2xl font-semibold text-ink">
              Built for circuit extraction, not generic image upload.
            </h2>
            <p className="text-pretty mt-3 text-sm leading-6 text-muted">
              The homepage gives the project shape; the demo page stays focused
              on running an image through the API and reviewing the returned SPICE
              output.
            </p>
          </div>

          <div className="grid gap-3 md:grid-cols-3">
            {homeFeatures.map((feature) => {
              const Icon = feature.icon;
              return (
                <article
                  key={feature.title}
                  className="rounded-md border border-line/70 bg-surface/72 p-5"
                >
                  <Icon className="h-5 w-5 text-accent" aria-hidden="true" />
                  <h3 className="mt-4 text-base font-semibold text-ink">
                    {feature.title}
                  </h3>
                  <p className="mt-2 text-sm leading-6 text-muted">
                    {feature.detail}
                  </p>
                </article>
              );
            })}
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8">
        <div className="grid gap-8 lg:grid-cols-[1.05fr_0.95fr]">
          <div className="rounded-lg border border-line/70 bg-surface/75 p-5">
            <div className="flex items-center gap-3">
              <Waypoints className="h-5 w-5 text-signal" aria-hidden="true" />
              <h2 className="text-2xl font-semibold text-ink">Project flow</h2>
            </div>
            <ol className="mt-6 grid gap-3">
              {pipelineStages.slice(0, 5).map((stage, index) => {
                const Icon = stage.icon;
                const step = String(index + 1).padStart(2, "0");
                return (
                  <li
                    key={stage.title}
                    className="grid gap-3 rounded-md border border-line/70 bg-background/42 p-4 sm:grid-cols-[44px_1fr]"
                  >
                    <span className="grid h-10 w-10 place-items-center rounded-md border border-line/70 text-accent">
                      <Icon className="h-4 w-4" aria-hidden="true" />
                    </span>
                    <div>
                      <p className="font-mono text-xs text-accent">{step}</p>
                      <h3 className="mt-1 text-sm font-semibold text-ink">
                        {stage.title}
                      </h3>
                      <p className="mt-1 text-sm leading-6 text-muted">
                        {stage.detail}
                      </p>
                    </div>
                  </li>
                );
              })}
            </ol>
          </div>

          <div className="rounded-lg border border-line/70 bg-surface/75 p-5">
            <div className="flex items-center gap-3">
              <ShieldCheck className="h-5 w-5 text-accent" aria-hidden="true" />
              <h2 className="text-2xl font-semibold text-ink">
                Output made for review
              </h2>
            </div>
            <p className="text-pretty mt-3 text-sm leading-6 text-muted">
              The generated result is meant to be checked by a human before it is
              trusted in a simulator or downstream workflow.
            </p>
            <ul className="mt-6 grid gap-3">
              {outputDetails.map((detail) => (
                <li key={detail} className="flex gap-3 text-sm leading-6 text-ink">
                  <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-signal" />
                  <span>{detail}</span>
                </li>
              ))}
            </ul>
            <Link
              href="/information"
              className="motion-control focus-ring mt-7 inline-flex min-h-11 items-center justify-center gap-2 rounded-md border border-line bg-background/45 px-4 text-sm font-semibold text-ink hover:bg-raised"
            >
              <span>Read technical details</span>
              <ArrowRight className="h-4 w-4" aria-hidden="true" />
            </Link>
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-7xl px-4 pb-14 sm:px-6 lg:px-8">
        <div className="flex flex-col justify-between gap-5 border-t border-line/70 pt-8 sm:flex-row sm:items-center">
          <div>
            <h2 className="text-xl font-semibold text-ink">Contact</h2>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-muted">
              Questions, deployment help, or project feedback can go directly
              through email.
            </p>
          </div>
          <div className="flex flex-col gap-2 sm:flex-row">
            {contactLinks.map((link) => (
              <a
                key={link.href}
                href={link.href}
                className="motion-control focus-ring inline-flex min-h-11 items-center justify-center gap-2 rounded-md border border-line bg-background/45 px-4 text-sm font-semibold text-ink hover:bg-raised"
              >
                <Mail className="h-4 w-4" aria-hidden="true" />
                <span>{link.label}</span>
              </a>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}
