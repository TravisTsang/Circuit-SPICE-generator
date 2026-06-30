# Design

## Theme
Circuit SPICE List Generator uses a dark, technical product interface that feels like a lab console rather than a startup landing page. The first viewport combines navigation, project identity, upload/paste affordance, status, and SPICE output preview. Visuals should be code-native circuit and signal motifs, not generated photography.

## Color
Use OKLCH tokens with a near-black neutral background, blue-green signal accents, amber warning states, and high-contrast foreground text.

- Background: `oklch(0.145 0.018 243)`
- Surface: `oklch(0.19 0.024 247)`
- Surface raised: `oklch(0.235 0.028 247)`
- Ink: `oklch(0.96 0.012 240)`
- Muted ink: `oklch(0.74 0.035 235)`
- Accent: `oklch(0.78 0.13 178)`
- Accent secondary: `oklch(0.72 0.12 232)`
- Warning: `oklch(0.78 0.13 74)`
- Error: `oklch(0.68 0.18 25)`
- Border: `oklch(0.34 0.035 245)`

## Typography
Use a single sans family, Inter or system UI fallback. Product text should stay readable and compact: no oversized display typography inside tool panels, no negative letter spacing, and prose capped around 70 characters.

## Layout
The site has four primary pages: Home, Information, Statistics, and Demo. Navigation remains accessible on mobile, with the demo surface prioritized over marketing copy. Sections use full-width bands and constrained inner content, with cards only for repeated items or tool panels.

## Components
Buttons, tabs, upload zones, segmented controls, result panes, API status panels, and tables must have complete default, hover, focus, disabled, loading, empty, and error states. Use Lucide icons consistently for controls.

## Motion
The background may respond to pointer movement and clicks, but the content must remain readable. Buttons should have clear hover and press feedback using transform, color, and subtle signal glows around 150-250ms. Disable nonessential motion under `prefers-reduced-motion`.
