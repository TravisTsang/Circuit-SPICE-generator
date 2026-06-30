import {
  Binary,
  BrainCircuit,
  FileCode2,
  GitBranch,
  ScanSearch,
  ShieldCheck,
  Waypoints,
  Zap,
} from "lucide-react";

export const navItems = [
  { href: "/", label: "Home" },
  { href: "/information", label: "Information" },
  { href: "/demo", label: "Demo" },
];

export const domains = [
  {
    id: "hand-drawn",
    label: "Hand-drawn",
    description: "Noisy scans, sketches, and phone captures",
  },
  {
    id: "digital",
    label: "Digital",
    description: "CAD exports, screenshots, and clean schematics",
  },
] as const;

export type CircuitDomain = (typeof domains)[number]["id"];

export const pipelineStages = [
  {
    title: "Trace segmentation",
    detail: "A PyTorch U-Net predicts which pixels belong to wires and produces a wire probability map.",
    icon: ScanSearch,
  },
  {
    title: "Component detection",
    detail: "YOLOv8 finds schematic symbols, classifies each component, and returns bounding boxes.",
    icon: BrainCircuit,
  },
  {
    title: "Text association",
    detail: "EasyOCR reads nearby labels and values, then regex and distance scoring attach text to components.",
    icon: Binary,
  },
  {
    title: "Topology recovery",
    detail: "Cleaned wire masks are skeletonized, split into connected segments, and labeled as circuit nets.",
    icon: Waypoints,
  },
  {
    title: "Netlist export",
    detail: "Component terminals are matched to net IDs and serialized into SPICE-compatible lines.",
    icon: FileCode2,
  },
  {
    title: "Warnings",
    detail: "Ambiguous labels, missing terminals, and disconnected nets stay visible for review.",
    icon: ShieldCheck,
  },
];

export const apiFacts = [
  {
    label: "Vercel frontend",
    value: "Next.js App Router",
  },
  {
    label: "Inference backend",
    value: "External FastAPI URL via BOT_OR_NOT_BACKEND_URL",
  },
  {
    label: "Upload contract",
    value: "multipart/form-data with image and domain",
  },
  {
    label: "Model files",
    value: "hand/digital U-Net and YOLO .pt weights",
  },
];

export const sampleNetlist = `* Circuit SPICE List Generator preview
V1 N001 0 DC 5
R1 N001 N002 10k
C1 N002 0 0.1u
.end`;

export const contactLinks = [
  {
    label: "t4tsang@uwaterloo.ca",
    href: "mailto:t4tsang@uwaterloo.ca?subject=Circuit%20SPICE%20List%20Generator",
  },
  {
    label: "enkai.liu.1@gmail.com",
    href: "mailto:enkai.liu.1@gmail.com?subject=Circuit%20SPICE%20List%20Generator",
  },
];

export const statusCopy = {
  ready: {
    icon: Zap,
    title: "Ready to process",
    detail: "Upload or paste a circuit image, then run the proxy API.",
  },
  missing: {
    icon: GitBranch,
    title: "Backend URL not configured",
    detail: "Set BOT_OR_NOT_BACKEND_URL on Vercel or run FastAPI locally.",
  },
};
