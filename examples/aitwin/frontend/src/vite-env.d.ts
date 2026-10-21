/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_AITWIN_TWIN_ID?: string
  readonly VITE_AITWIN_API_BASE?: string
  readonly VITE_TOPIC_SPEECH: string
  readonly VITE_TOPIC_SAY: string
  readonly VITE_TOPIC_CHAT: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
