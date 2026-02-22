"use client";

import { ReactNode } from "react";
import { LiveKitRoom } from "@livekit/components-react";
import { getAppConfig } from "@/lib/app-config";

interface ClientLayoutProps {
  children: ReactNode;
}

export default function ClientLayout({ children }: ClientLayoutProps) {
  const config = getAppConfig();

  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta name="description" content="Real-time vision and avatar agent" />
        <title>{config.brandName}</title>
      </head>
      <body className="bg-slate-50 dark:bg-slate-950">
        <main className="w-full h-screen flex flex-col">
          {children}
        </main>
      </body>
    </html>
  );
}
