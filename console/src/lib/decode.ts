import type { CallRequest, CallMetadata, CallToken, TitleRequest } from "./types";

export function record(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
const optionalString = (value: unknown) => value == null || typeof value === "string";

/** Validate browser input before minting credentials or saving a title. */
export const decode = {
  callRequest(value: unknown): CallRequest {
    if (
      !record(value) ||
      !record(value.room_config) ||
      !Array.isArray(value.room_config.agents) ||
      !value.room_config.agents.length ||
      !value.room_config.agents.every(
        (agent) =>
          record(agent) &&
          typeof agent.agent_name === "string" &&
          agent.agent_name.trim() &&
          typeof agent.metadata === "string",
      ) ||
      ![value.room_name, value.participant_name, value.participant_identity].every(
        optionalString,
      ) ||
      (value.participant_attributes != null &&
        (!record(value.participant_attributes) ||
          !Object.values(value.participant_attributes).every((item) => typeof item === "string")))
    )
      throw new Error("Select an agent for this call");
    return value as CallRequest;
  },
  callMetadata(value: unknown): CallMetadata {
    if (
      !record(value) ||
      typeof value.endpoint !== "string" ||
      !value.endpoint.trim() ||
      ![value.session_id, value.sub].every(optionalString)
    )
      throw new Error("Select an endpoint for this call");
    return value as CallMetadata;
  },
  callToken(value: unknown): CallToken {
    if (
      !record(value) ||
      ![value.server_url, value.participant_token, value.session_id].every(
        (item) => typeof item === "string" && item.length > 0,
      )
    )
      throw new Error("The call could not be started");
    return value as CallToken;
  },
  titleRequest(value: unknown): TitleRequest {
    if (
      !record(value) ||
      typeof value.sessionId !== "string" ||
      !value.sessionId ||
      typeof value.firstMessage !== "string"
    )
      throw new Error("Choose a conversation and message");
    return value as TitleRequest;
  },
};
