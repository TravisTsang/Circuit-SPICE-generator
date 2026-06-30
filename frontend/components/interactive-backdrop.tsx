"use client";

import { useEffect } from "react";

export function InteractiveBackdrop() {
  useEffect(() => {
    const root = document.documentElement;
    let pulseTimer: number | undefined;

    const setPointer = (clientX: number, clientY: number) => {
      const x = `${Math.round((clientX / window.innerWidth) * 100)}%`;
      const y = `${Math.round((clientY / window.innerHeight) * 100)}%`;
      root.style.setProperty("--pointer-x", x);
      root.style.setProperty("--pointer-y", y);
    };

    const handlePointerMove = (event: PointerEvent) => {
      setPointer(event.clientX, event.clientY);
    };

    const handlePointerDown = (event: PointerEvent) => {
      setPointer(event.clientX, event.clientY);
      root.style.setProperty("--pulse-alpha", "0.18");
      if (pulseTimer) window.clearTimeout(pulseTimer);
      pulseTimer = window.setTimeout(() => {
        root.style.setProperty("--pulse-alpha", "0");
      }, 260);
    };

    window.addEventListener("pointermove", handlePointerMove, { passive: true });
    window.addEventListener("pointerdown", handlePointerDown, { passive: true });

    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerdown", handlePointerDown);
      if (pulseTimer) window.clearTimeout(pulseTimer);
    };
  }, []);

  return <div className="circuit-backdrop" aria-hidden="true" />;
}
