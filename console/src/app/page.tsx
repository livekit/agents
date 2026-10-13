"use client";

import dynamic from "next/dynamic";

/* The room reads localStorage for the dock layout and the session; rendering
   it only on the client keeps the server's default layout from ever
   disagreeing with the saved one. */
const Room = dynamic(() => import("@/components/room").then((m) => m.Room), { ssr: false });

export default function Page() {
  return <Room />;
}
