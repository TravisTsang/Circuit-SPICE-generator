import type { Metadata, Viewport } from "next";
import { InteractiveBackdrop } from "@/components/interactive-backdrop";
import { SiteHeader } from "@/components/site-header";
import "./globals.css";

const siteUrl = process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000";

export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
  title: {
    default: "Circuit SPICE List Generator",
    template: "%s | Circuit SPICE List Generator",
  },
  description:
    "Circuit SPICE List Generator turns schematic images into SPICE netlists through a circuit OCR pipeline for traces, components, OCR, topology, and export.",
  keywords: [
    "circuit OCR",
    "SPICE netlist",
    "schematic recognition",
    "electrical engineering",
    "image to SPICE",
  ],
  authors: [{ name: "Circuit SPICE List Generator" }],
  openGraph: {
    title: "Circuit SPICE List Generator",
    description: "Paste a circuit image and inspect the generated SPICE netlist.",
    url: siteUrl,
    siteName: "Circuit SPICE List Generator",
    images: [
      {
        url: "/opengraph-image",
        width: 1200,
        height: 630,
        alt: "Circuit SPICE List Generator technical circuit interface",
      },
    ],
    locale: "en_US",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Circuit SPICE List Generator",
    description: "Image-to-SPICE circuit OCR demo and project site.",
    images: ["/opengraph-image"],
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  colorScheme: "dark",
  themeColor: "#071018",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="font-sans">
        <a className="skip-link" href="#main-content">
          Skip to content
        </a>
        <InteractiveBackdrop />
        <SiteHeader />
        {children}
      </body>
    </html>
  );
}
