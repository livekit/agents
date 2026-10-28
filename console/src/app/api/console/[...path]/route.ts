import { backendHeaders, backendURL, localDeveloper } from "@/lib/backend";

export const dynamic = "force-dynamic";
async function proxy(request: Request, context: { params: Promise<{ path: string[] }> }) {
  if (!localDeveloper(request)) return Response.json({ error: "Developer access required" }, { status: 403 });
  const { path } = await context.params;
  // Encode each segment so a browser cannot redirect the privileged request.
  const target = `${backendURL}/api/${path.map(encodeURIComponent).join("/")}${new URL(request.url).search}`;
  const upstream = await fetch(target, {
    method: request.method, headers: backendHeaders(), cache: "no-store", signal: request.signal,
    ...(request.method === "POST" ? { body: await request.text() } : {}),
  });
  return new Response(upstream.body, { status: upstream.status, headers: { "Content-Type": "application/json", "Cache-Control": "no-store" } });
}
export { proxy as GET, proxy as POST };
