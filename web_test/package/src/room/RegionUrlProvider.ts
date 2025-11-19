import type { RegionInfo, RegionSettings } from '@livekit/protocol';
import log from '../logger';
import { ConnectionError, ConnectionErrorReason } from './errors';
import { isCloud } from './utils';

export class RegionUrlProvider {
  private serverUrl: URL;

  private token: string;

  private regionSettings: RegionSettings | undefined;

  private lastUpdateAt: number = 0;

  private settingsCacheTime = 3_000;

  private attemptedRegions: RegionInfo[] = [];

  constructor(url: string, token: string) {
    this.serverUrl = new URL(url);
    this.token = token;
  }

  updateToken(token: string) {
    this.token = token;
  }

  isCloud() {
    return isCloud(this.serverUrl);
  }

  getServerUrl() {
    return this.serverUrl;
  }

  async getNextBestRegionUrl(abortSignal?: AbortSignal) {
    if (!this.isCloud()) {
      throw Error('region availability is only supported for LiveKit Cloud domains');
    }
    if (!this.regionSettings || Date.now() - this.lastUpdateAt > this.settingsCacheTime) {
      this.regionSettings = await this.fetchRegionSettings(abortSignal);
    }
    const regionsLeft = this.regionSettings.regions.filter(
      (region) => !this.attemptedRegions.find((attempted) => attempted.url === region.url),
    );
    if (regionsLeft.length > 0) {
      const nextRegion = regionsLeft[0];
      this.attemptedRegions.push(nextRegion);
      log.debug(`next region: ${nextRegion.region}`);
      return nextRegion.url;
    } else {
      return null;
    }
  }

  resetAttempts() {
    this.attemptedRegions = [];
  }

  /* @internal */
  async fetchRegionSettings(signal?: AbortSignal) {
    const regionSettingsResponse = await fetch(`${getCloudConfigUrl(this.serverUrl)}/regions`, {
      headers: { authorization: `Bearer ${this.token}` },
      signal,
    });
    if (regionSettingsResponse.ok) {
      const regionSettings = (await regionSettingsResponse.json()) as RegionSettings;
      this.lastUpdateAt = Date.now();
      return regionSettings;
    } else {
      throw new ConnectionError(
        `Could not fetch region settings: ${regionSettingsResponse.statusText}`,
        regionSettingsResponse.status === 401
          ? ConnectionErrorReason.NotAllowed
          : ConnectionErrorReason.InternalError,
        regionSettingsResponse.status,
      );
    }
  }

  setServerReportedRegions(regions: RegionSettings) {
    this.regionSettings = regions;
    this.lastUpdateAt = Date.now();
  }
}

function getCloudConfigUrl(serverUrl: URL) {
  return `${serverUrl.protocol.replace('ws', 'http')}//${serverUrl.host}/settings`;
}
