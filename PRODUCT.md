# Product

## Register

product

## Users
Researchers, students, electronics hobbyists, and project reviewers who need to turn schematic images into a usable SPICE starting point. They may be testing the pipeline locally, evaluating the project online, or checking whether a pasted circuit image can produce a credible netlist.

## Product Purpose
Circuit SPICE List Generator presents and demonstrates an image-to-SPICE OCR pipeline. The site should quickly explain the system, show the current technical state honestly, and let users paste or upload a circuit image for netlist generation through the Python inference backend when model weights and a backend URL are available.

## Brand Personality
Precise, calm, technical. The interface should feel like a capable research instrument: focused, transparent about limitations, and polished enough to trust without drifting into generic startup marketing.

## Anti-references
Avoid generic SaaS hero sections, vague AI promises, oversized decorative cards, nested card layouts, lorem ipsum, hype-heavy copy, and visual effects that make the demo harder to read. Do not imply model-backed inference is available on Vercel unless the backend environment is configured.

## Design Principles
- Put the conversion task in the first screen so the project is immediately legible.
- Prefer technical clarity over persuasion: inputs, pipeline stages, limitations, and outputs should be easy to scan.
- Make error and empty states useful, especially when the ML backend or model weights are missing.
- Keep motion responsive and state-driven, with reduced-motion support.
- Use restrained contrast and spacing so long technical content remains readable in a dark environment.

## Accessibility & Inclusion
Target WCAG AA contrast, keyboard-accessible navigation and upload controls, visible focus states, text labels for all controls, reduced-motion alternatives for dynamic background effects, and responsive layouts from small mobile screens through desktop.
