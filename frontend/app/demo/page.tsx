import type { Metadata } from "next";
import { DemoConsole } from "@/components/demo-console";

export const metadata: Metadata = {
  title: "Demo",
  description:
    "Paste or upload a schematic image and request SPICE netlist generation through the Circuit SPICE List Generator API.",
};

export default function DemoPage() {
  return (
    <main id="main-content" className="mx-auto w-full max-w-7xl px-4 py-10 sm:px-6 lg:px-8">
      <section className="mb-8 max-w-3xl">
        <p className="mb-4 inline-flex rounded-md border border-accent/45 bg-accent/10 px-3 py-1 text-sm font-medium text-accent">
          Demo
        </p>
        <h1 className="text-balance text-4xl font-semibold leading-tight text-ink sm:text-5xl">
          Get Started
        </h1>
      </section>
      <DemoConsole />
    </main>
  );
}
