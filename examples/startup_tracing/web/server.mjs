import { createRequire } from 'node:module';
import { homedir } from 'node:os';
import { resolve, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
const sdk = process.env.LIVEKIT_JS_SDK_DIR || resolve(homedir(), 'dev/client-sdk-js');
const require = createRequire(resolve(sdk, 'package.json'));
const { createServer } = await import(pathToFileURL(require.resolve('vite')).href);
const server = await createServer({
  root: dirname(fileURLToPath(import.meta.url)),
  server: { host: '127.0.0.1', port: 18791, strictPort: true, fs: { allow: [sdk, dirname(fileURLToPath(import.meta.url))] } },
  resolve: { alias: { 'livekit-client': resolve(sdk, 'src/index.ts') } },
});
await server.listen();
const stop = async () => { await server.close(); process.exit(0); };
process.on('SIGINT', stop);
process.on('SIGTERM', stop);
