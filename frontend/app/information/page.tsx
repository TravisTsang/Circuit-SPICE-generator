import type { Metadata } from "next";
import {
  Boxes,
  BrainCircuit,
  Cpu,
  FileCode2,
  GitBranch,
  ListChecks,
  ScanSearch,
  SlidersHorizontal,
  SquareDashedMousePointer,
  Type,
  Waypoints,
  Wrench,
} from "lucide-react";

export const metadata: Metadata = {
  title: "Information",
  description:
    "How Circuit SPICE List Generator uses U-Net, YOLOv8, EasyOCR, OpenCV, and net labeling to convert schematic images into SPICE netlists.",
};

const modelOverview = [
  {
    title: "U-Net PyTorch model",
    detail:
      "Outputs a probability for which pixels are wires. It is trained on circuit schematic images where every pixel is labeled wire or not-wire.",
    icon: BrainCircuit,
  },
  {
    title: "YOLOv8 model",
    detail:
      "Detects and classifies components. It is trained on schematic images with bounding boxes around every resistor, capacitor, inductor, and other symbol classes.",
    icon: SquareDashedMousePointer,
  },
  {
    title: "EasyOCR model",
    detail:
      "Uses a pre-trained text recognition model to read component labels and values from the schematic image.",
    icon: Type,
  },
];

const pipeline = [
  {
    title: "Schematic image uploaded",
    detail:
      "The user uploads or pastes a schematic image, then the backend normalizes it for model inference.",
    icon: ScanSearch,
  },
  {
    title: "U-Net scans the image",
    detail:
      "The PyTorch U-Net evaluates the image and returns a probability map for wire pixels.",
    icon: BrainCircuit,
  },
  {
    title: "Sigmoid and thresholding",
    detail:
      "Sigmoid converts model output into probabilities. Pixels above 50% become wire mask pixels; the rest become background.",
    icon: SlidersHorizontal,
  },
  {
    title: "OpenCV morphology cleans the mask",
    detail:
      "Morphology operations close tiny gaps in traces, fill short breaks, and remove isolated noise pixels.",
    icon: Wrench,
  },
  {
    title: "YOLOv8 detects component symbols",
    detail:
      "YOLOv8 locates component symbols, classifies each one, and returns bounding boxes around them.",
    icon: Boxes,
  },
  {
    title: "EasyOCR reads labels and values",
    detail:
      "EasyOCR reads text, then proximity scoring and regex patterns match labels and values to nearby components.",
    icon: Type,
  },
  {
    title: "Wire mask becomes labeled nets",
    detail:
      "The cleaned wire mask is skeletonized into thin traces, then each connected segment receives a net ID.",
    icon: Waypoints,
  },
  {
    title: "Component terminals match to nets",
    detail:
      "Each component terminal is checked against nearby wire pixels so pins can be assigned to nets such as N001 and N002.",
    icon: GitBranch,
  },
  {
    title: "SPICE netlist gets written",
    detail:
      "Recovered components, labels, values, and terminal nets are formatted as SPICE lines and written to disk.",
    icon: FileCode2,
  },
];

const librariesUsed = [
  {
    label: "Wire segmentation",
    value: "PyTorch U-Net",
  },
  {
    label: "Component detection",
    value: "Ultralytics YOLOv8",
  },
  {
    label: "Text recognition",
    value: "EasyOCR",
  },
  {
    label: "Image processing",
    value: "OpenCV, scikit-image, NumPy, Pillow",
  },
  {
    label: "Circuit topology",
    value: "NetworkX",
  },
  {
    label: "SPICE output",
    value: "Python file I/O and formatting utilities",
  },
];

export default function InformationPage() {
  return (
    <main
      id="main-content"
      className="mx-auto w-full max-w-7xl px-4 py-10 sm:px-6 lg:px-8"
    >
      <section className="max-w-3xl">
        <p className="mb-4 inline-flex rounded-md border border-signal/45 bg-signal/10 px-3 py-1 text-sm font-medium text-signal">
          Technical Information
        </p>
        <h1 className="text-balance text-4xl font-semibold leading-tight text-ink sm:text-5xl">
          How a schematic image becomes a SPICE list.
        </h1>
        <p className="text-pretty mt-5 text-lg leading-8 text-muted">
          The backend combines trained image models, OCR, mask cleanup, and graph
          recovery so a circuit drawing can be translated into connected
          components and named nets.
        </p>
      </section>

      <section className="mt-10">
        <div className="flex items-center gap-3">
          <Cpu className="h-5 w-5 text-accent" aria-hidden="true" />
          <div>
            <h2 className="text-2xl font-semibold text-ink">High-level overview</h2>
            <p className="mt-1 text-sm text-muted">
              The program uses three models: two project-specific trained models
              and one pre-trained OCR model.
            </p>
          </div>
        </div>
        <div className="mt-5 grid gap-3 md:grid-cols-3">
          {modelOverview.map((model) => {
            const Icon = model.icon;
            return (
              <article
                key={model.title}
                className="rounded-md border border-line/70 bg-surface/75 p-5"
              >
                <Icon className="h-5 w-5 text-accent" aria-hidden="true" />
                <h3 className="mt-4 text-base font-semibold text-ink">{model.title}</h3>
                <p className="mt-2 text-sm leading-6 text-muted">{model.detail}</p>
              </article>
            );
          })}
        </div>
      </section>

      <section className="mt-12">
        <div className="rounded-lg border border-line/70 bg-surface/75 p-5">
          <div className="flex items-center gap-3">
            <Cpu className="h-5 w-5 text-accent" aria-hidden="true" />
            <div>
              <h2 className="text-2xl font-semibold text-ink">Libraries Used</h2>
              <p className="mt-1 text-sm text-muted">
                The Python backend combines model inference, image processing,
                OCR, graph recovery, and SPICE export utilities.
              </p>
            </div>
          </div>
          <dl className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {librariesUsed.map((item) => (
              <InfoRow key={item.label} label={item.label} value={item.value} />
            ))}
          </dl>
        </div>
      </section>

      <section className="mt-12">
        <div className="flex items-center gap-3">
          <ListChecks className="h-5 w-5 text-signal" aria-hidden="true" />
          <div>
            <h2 className="text-2xl font-semibold text-ink">Pipeline</h2>
            <p className="mt-1 text-sm text-muted">
              How the uploaded schematic becomes a SPICE netlist.
            </p>
          </div>
        </div>
        <div className="mt-5 grid gap-3">
          {pipeline.map((stage, index) => {
            const Icon = stage.icon;
            const step = String(index + 1).padStart(2, "0");
            return (
              <article
                key={stage.title}
                className="grid gap-4 rounded-md border border-line/70 bg-surface/75 p-4 sm:grid-cols-[72px_44px_1fr] sm:items-start"
              >
                <span className="font-mono text-sm font-semibold text-accent">{step}</span>
                <span className="grid h-11 w-11 place-items-center rounded-md border border-line/70 bg-background/55 text-accent">
                  <Icon className="h-5 w-5" aria-hidden="true" />
                </span>
                <div>
                  <h3 className="text-base font-semibold text-ink">{stage.title}</h3>
                  <p className="mt-1 text-sm leading-6 text-muted">{stage.detail}</p>
                </div>
              </article>
            );
          })}
        </div>
      </section>
    </main>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-xs font-medium text-muted">{label}</dt>
      <dd className="mt-1 text-sm leading-6 text-ink">{value}</dd>
    </div>
  );
}
