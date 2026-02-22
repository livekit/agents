"use client";

import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";
import { getAppConfig } from "@/lib/app-config";
import { AgentSessionProvider } from "@/components/app/agent-session-provider";
import { AppView } from "@/components/app/app-view";

export default function HomePage() {
  const config = getAppConfig();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-slate-50 dark:bg-slate-950">
        <div className="flex flex-col items-center gap-2">
          <Loader2 className="w-8 h-8 animate-spin text-lk-accent" />
          <p className="text-sm text-slate-600 dark:text-slate-400">
            Initializing...
          </p>
        </div>
      </div>
    );
  }

  return (
    <AgentSessionProvider>
      <AppView />
    </AgentSessionProvider>
  );
}
