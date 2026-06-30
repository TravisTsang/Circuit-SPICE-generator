"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { CircuitBoard, Menu, X } from "lucide-react";
import { useState } from "react";
import { navItems } from "@/lib/site";

export function SiteHeader() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  return (
    <header className="sticky top-0 z-40 border-b border-line/70 bg-background/82 backdrop-blur-md">
      <div className="mx-auto flex min-h-16 w-full max-w-7xl items-center justify-between gap-4 px-4 sm:px-6 lg:px-8">
        <Link
          href="/"
          className="motion-control focus-ring inline-flex min-h-11 items-center gap-3 rounded-md px-1"
          onClick={() => setOpen(false)}
        >
          <span className="grid h-10 w-10 place-items-center rounded-md border border-accent/45 bg-accent/12 text-accent">
            <CircuitBoard className="h-5 w-5" aria-hidden="true" />
          </span>
          <span className="leading-tight">
            <span className="block text-sm font-semibold text-ink">Circuit SPICE</span>
            <span className="block text-xs text-muted">List Generator</span>
          </span>
        </Link>

        <nav aria-label="Primary navigation" className="hidden items-center gap-1 md:flex">
          {navItems.map((item) => {
            const active = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`motion-control focus-ring min-h-11 rounded-md px-3 py-2 text-sm font-medium ${
                  active
                    ? "bg-raised text-ink"
                    : "text-muted hover:bg-raised/80 hover:text-ink"
                }`}
                aria-current={active ? "page" : undefined}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>

        <button
          type="button"
          className="motion-control focus-ring grid h-11 w-11 place-items-center rounded-md border border-line/75 bg-surface text-ink md:hidden"
          aria-label={open ? "Close navigation" : "Open navigation"}
          aria-expanded={open}
          onClick={() => setOpen((value) => !value)}
        >
          {open ? <X className="h-5 w-5" aria-hidden="true" /> : <Menu className="h-5 w-5" aria-hidden="true" />}
        </button>
      </div>

      {open ? (
        <nav
          aria-label="Mobile navigation"
          className="border-t border-line/70 bg-background px-4 py-3 md:hidden"
        >
          <div className="mx-auto grid max-w-7xl gap-1">
            {navItems.map((item) => {
              const active = pathname === item.href;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`motion-control focus-ring min-h-11 rounded-md px-3 py-3 text-sm font-medium ${
                    active ? "bg-raised text-ink" : "text-muted"
                  }`}
                  aria-current={active ? "page" : undefined}
                  onClick={() => setOpen(false)}
                >
                  {item.label}
                </Link>
              );
            })}
          </div>
        </nav>
      ) : null}
    </header>
  );
}
