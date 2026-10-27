"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { cn } from "@/lib/utils";

const NAV = [
  { href: "/", label: "Meetings" },
  { href: "/series", label: "Series" },
  { href: "/search", label: "Search" },
];

export function SiteHeader() {
  const path = usePathname();
  return (
    <header className="sticky top-0 z-40 border-b bg-background/85 backdrop-blur supports-[backdrop-filter]:bg-background/70">
      <div className="mx-auto flex h-14 max-w-[1600px] items-center gap-6 px-4 sm:px-6">
        <Link href="/" className="flex items-baseline gap-1.5">
          <span className="font-serif text-2xl leading-none tracking-tight">Minute</span>
          <span className="rounded bg-primary px-1 py-px font-mono text-[10px] font-semibold tracking-wider text-primary-foreground">
            AI
          </span>
        </Link>
        <nav className="flex items-center gap-1 text-sm">
          {NAV.map((n) => {
            const active = n.href === "/" ? path === "/" || path.startsWith("/meetings") : path.startsWith(n.href);
            return (
              <Link
                key={n.href}
                href={n.href}
                className={cn(
                  "rounded-md px-2.5 py-1.5 text-muted-foreground transition-colors hover:text-foreground",
                  active && "bg-secondary text-foreground",
                )}
              >
                {n.label}
              </Link>
            );
          })}
        </nav>
        <p className="ml-auto hidden text-xs text-muted-foreground md:block">
          Every line cites what was said.
        </p>
      </div>
    </header>
  );
}
