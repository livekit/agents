import { toHttpUrl, toWebsocketUrl } from '../room/utils';

export function createRtcUrl(url: string, searchParams: URLSearchParams) {
  const urlObj = new URL(toWebsocketUrl(url));
  searchParams.forEach((value, key) => {
    urlObj.searchParams.set(key, value);
  });
  return appendUrlPath(urlObj, 'rtc');
}

export function createValidateUrl(rtcWsUrl: string) {
  const urlObj = new URL(toHttpUrl(rtcWsUrl));
  return appendUrlPath(urlObj, 'validate');
}

function ensureTrailingSlash(path: string) {
  return path.endsWith('/') ? path : `${path}/`;
}

function appendUrlPath(urlObj: URL, path: string) {
  urlObj.pathname = `${ensureTrailingSlash(urlObj.pathname)}${path}`;
  return urlObj.toString();
}
