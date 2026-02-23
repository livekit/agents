export interface AppConfig {
  pageTitle: string;
  pageDescription: string;
  companyName: string;

  supportsChatInput: boolean;
  supportsVideoInput: boolean;
  supportsScreenShare: boolean;
  isPreConnectBufferEnabled: boolean;

  logo: string;
  startButtonText: string;
  accent?: string;
  logoDark?: string;
  accentDark?: string;

  // agent dispatch configuration
  agentName?: string;

  // LiveKit Cloud Sandbox configuration
  sandboxId?: string;
}

export const APP_CONFIG_DEFAULTS: AppConfig = {
  companyName: 'LiveCare',
  pageTitle: 'LiveCare',
  pageDescription: 'AI-powered healthcare intake assistant',

  supportsChatInput: true,
  supportsVideoInput: true,
  supportsScreenShare: true,
  isPreConnectBufferEnabled: true,

  logo: '/livecare-logo-light.svg',
  accent: '#00753B',
  logoDark: '/livecare-logo-dark.svg',
  accentDark: '#1EB66A',
  startButtonText: 'Start intake',

  // agent dispatch configuration
  agentName: process.env.AGENT_NAME ?? 'Anam-Demo',

  // LiveKit Cloud Sandbox configuration
  sandboxId: undefined,
};
