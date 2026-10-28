import Ajv, { type ValidateFunction } from "ajv";
import type { AgentCard } from "./generated/AgentCard";
import type { StreamResponse } from "./generated/StreamResponse";
import type { ConsoleConfig } from "./generated/ConsoleConfig";
import type { Grant } from "./generated/Grant";
import type { SessionDump } from "./generated/SessionDump";
import type { SessionRow } from "./generated/SessionRow";
import type { History } from "./generated/History";
import type { Sessions } from "./generated/Sessions";
import type { CreatedSession } from "./generated/CreatedSession";
import type { QueryResult } from "./generated/QueryResult";
import type { Examples } from "./generated/Examples";
import type { Message } from "./generated/Message";
import message from "./generated/Message.json";
import type { CallRequest } from "./generated/CallRequest";
import type { CallMetadata } from "./generated/CallMetadata";
import type { CallToken } from "./generated/CallToken";
import type { TitleRequest } from "./generated/TitleRequest";
import callRequest from "./generated/CallRequest.json";
import callMetadata from "./generated/CallMetadata.json";
import callToken from "./generated/CallToken.json";
import titleRequest from "./generated/TitleRequest.json";
import agentCard from "./generated/AgentCard.json";
import streamResponse from "./generated/StreamResponse.json";
import config from "./generated/ConsoleConfig.json";
import grant from "./generated/Grant.json";
import sessionDump from "./generated/SessionDump.json";
import sessionRow from "./generated/SessionRow.json";
import history from "./generated/History.json";
import sessions from "./generated/Sessions.json";
import createdSession from "./generated/CreatedSession.json";
import queryResult from "./generated/QueryResult.json";
import examples from "./generated/Examples.json";

const ajv = new Ajv({ strict: false, validateFormats: false });
function decoder<T>(schema: object): (value: unknown) => T {
  const validate: ValidateFunction<T> = ajv.compile<T>(schema);
  return (value) => {
    if (!validate(value)) throw new Error(`Invalid response: ${ajv.errorsText(validate.errors)}`);
    return value;
  };
}
export const decode = {
  callRequest: decoder<CallRequest>(callRequest),
  callMetadata: decoder<CallMetadata>(callMetadata),
  callToken: decoder<CallToken>(callToken),
  titleRequest: decoder<TitleRequest>(titleRequest),
  message: decoder<Message>(message),
  card: decoder<AgentCard>(agentCard),
  stream: decoder<StreamResponse>(streamResponse),
  config: decoder<ConsoleConfig>(config),
  grant: decoder<Grant>(grant),
  dump: decoder<SessionDump>(sessionDump),
  session: decoder<SessionRow>(sessionRow),
  history: decoder<History>(history),
  sessions: decoder<Sessions>(sessions),
  created: decoder<CreatedSession>(createdSession),
  query: decoder<QueryResult>(queryResult),
  examples: decoder<Examples>(examples),
};
