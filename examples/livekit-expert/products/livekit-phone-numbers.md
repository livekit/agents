LiveKit Phone Numbers and telephony, including buying numbers, dispatch rules, SIP trunks, inbound and outbound calls, transfers, and DTMF.

OVERVIEW:
LiveKit Phone Numbers lets you purchase and manage US phone numbers for voice applications directly through LiveKit Cloud. It provides telephony infrastructure and phone number inventory without requiring separate SIP trunk configuration. You buy local or toll-free numbers directly through LiveKit and assign them to voice agents using dispatch rules. It currently supports inbound calling only, with outbound call support coming soon. You can manage phone numbers using the LiveKit Cloud dashboard, the LiveKit CLI, or the Phone Numbers APIs.

KEY BENEFITS:
Buy numbers directly from LiveKit without needing a third-party SIP provider. Setup is streamlined because there is no SIP trunk complexity for inbound calls. All calls get high-definition HD voice with clear professional audio quality. Management is unified through LiveKit Cloud for procuring numbers, configuring dispatch rules, and reviewing call metrics and logs. With LiveKit Phone Numbers, calls skip all trunking and go straight to LiveKit, which means there is no inbound trunk configuration or verification required.

SETTING UP A PHONE NUMBER:
There are three steps. First, search for an available number by country and area code using the dashboard, CLI, or API. Second, purchase the number. Third, assign the number to a dispatch rule so incoming calls get routed to a LiveKit room and your voice agent. You can do all three steps in the LiveKit Cloud dashboard under Telephony then Phone Numbers, or use the CLI with commands like lk number search, lk number buy, and lk number assign.

CLI REFERENCE FOR PHONE NUMBERS:
The LiveKit CLI provides phone number management commands prefixed with lk number. The main commands are lk number search to find available numbers by country and area code, lk number buy to purchase a number, lk number assign to assign a number to a dispatch rule, lk number unassign to remove a dispatch rule assignment, lk number list to show all purchased numbers, lk number release to release a number you no longer need, and lk number get to view details of a specific number.

PHONE NUMBERS API:
The PhoneNumberService APIs allow you to manage phone numbers programmatically. The main operations are SearchPhoneNumbers to find available numbers, PurchasePhoneNumber to buy a number, AssignPhoneNumber to assign a number to a dispatch rule, UnassignPhoneNumber to remove a dispatch rule assignment, ListPhoneNumbers to list all purchased numbers, ReleasePhoneNumber to release a number, and GetPhoneNumber to get details about a specific number. These APIs are available in Go, JavaScript, Python, Ruby, and Java server SDKs.

PRICING:
LiveKit Phone Numbers are metered by the minute of inbound call time, plus a small fixed monthly fee per number. Each individual resource usage is rounded up to a minimum increment of one minute for call time and one number for monthly rental. If you release a phone number before the end of the month, you are still billed for the entire month.

DISPATCH RULES:
A dispatch rule controls how callers are added as SIP participants in rooms. When an inbound call reaches LiveKit, the SIP service looks for a matching dispatch rule and uses it to place the caller into a LiveKit room. There are two main types. An individual dispatch rule creates a new room for each caller, with an optional room name prefix. A direct dispatch rule places all callers into a specific named room. Dispatch rules can also include agent dispatch configuration to specify which agent should be dispatched to the room, and they can optionally require a PIN for room access.

INBOUND CALL WORKFLOW:
When someone calls your LiveKit Phone Number, the call goes directly to LiveKit SIP without passing through a third-party trunk. LiveKit SIP finds a matching dispatch rule. A SIP participant is created for the caller and placed in a LiveKit room per the dispatch rule. The caller hears a dial tone until another participant, typically your voice agent, joins the room. If the dispatch rule has a PIN, the caller is prompted to enter it before joining.

THIRD-PARTY SIP TRUNKS:
If you prefer to use a third-party SIP provider like Twilio, Telnyx, or Plivo instead of LiveKit Phone Numbers, you need to set up an inbound trunk and dispatch rule. The inbound trunk authenticates calls from your provider. You configure your SIP provider to point to the LiveKit SIP endpoint, then create the inbound trunk and dispatch rule in LiveKit. This gives you more flexibility but requires more configuration. LiveKit supports providers including Twilio, Telnyx, Plivo, Vonage, and others.

OUTBOUND CALLING:
For outbound calls, you need a third-party SIP provider with an outbound trunk configured. You create a SIP participant using the CreateSIPParticipant API, specifying the outbound trunk ID, the phone number to dial, the room name, and participant details. The agent can initiate outbound calls programmatically. You can customize calls with features like custom caller ID, DTMF tones for extension codes, and dial tone playback while the call connects. LiveKit Phone Numbers does not yet support outbound calling, so you need a third-party provider for this.

AGENTS TELEPHONY INTEGRATION:
To connect a voice agent with telephony, first build your agent using the LiveKit Agents SDK, then set up either a LiveKit Phone Number or a third-party SIP trunk. For inbound calls, create a dispatch rule that dispatches your agent to the room when a call comes in. For outbound calls, your agent code creates a SIP participant to dial out. The agent name in the dispatch rule must match the name assigned to your agent in code.

DTMF SUPPORT:
LiveKit fully supports Dual-tone Multi-Frequency DTMF tones for integration with legacy IVR systems and receiving keypad input from callers. You can send DTMF tones using the publishDtmf API and receive them by listening to DTMF events. The Agents framework provides additional support including IVR detection, which you enable by setting ivr_detection to True in the AgentSession constructor. There is also a prebuilt GetDtmfTask for collecting digit input from callers, supporting both DTMF tones and spoken digits.

CALL TRANSFERS:
LiveKit supports two types of call transfers. Cold transfer, also called call forwarding, uses SIP REFER to transfer a call to another phone number or SIP endpoint. The current session ends after the transfer completes. You use the transfer_sip_participant method for this. Warm transfer is agent-assisted, where the AI agent dials a supervisor or human agent, provides conversation context, plays hold music for the original caller, and then merges the calls. There is a prebuilt WarmTransferTask that handles the entire workflow including creating a separate room for the human agent, dialing them, playing hold music, and handing off context.

HD VOICE:
LiveKit telephony supports high-definition voice using wideband codecs for superior call quality compared to traditional PSTN calls. This provides clearer audio for both agent dialogue and caller speech.

SECURE TRUNKING:
You can encrypt signaling and media traffic using TLS and SRTP to protect calls from eavesdropping and man-in-the-middle attacks. This is important for secure communications and compliance requirements.

REGION PINNING:
You can restrict network traffic to specific geographical regions to comply with local telephony regulations or data residency requirements. When configuring your SIP trunk, you can specify a regional endpoint instead of the default global endpoint.

NOISE CANCELLATION:
LiveKit provides background voice cancellation powered by Krisp for telephony calls. You enable it by setting krisp_enabled to True on the inbound trunk or SIP participant configuration. This removes background noise for clearer audio quality.

SIP PRIMER:
SIP stands for Session Initiation Protocol, and it is the standard protocol for establishing voice calls over the internet. The PSTN is the Public Switched Telephone Network, the global phone network. A SIP trunk connects your application to the PSTN through a SIP trunking provider. With LiveKit Phone Numbers, calls skip the trunking step and go straight to LiveKit. RTP is the protocol used to transmit the actual audio data during a call.

CONSIDERATIONS AND LIMITATIONS:
LiveKit Phone Numbers currently only supports inbound calling. Outbound calling support is coming soon. Forwarding calls using TransferSipParticipant is not yet supported with LiveKit Phone Numbers. Numbers are currently available in the United States only, both local and toll-free. If you release a number before end of month, you are still billed for the full month.

CLI TOOLS FOR SIP MANAGEMENT:
Beyond phone number commands, the LiveKit CLI has SIP management commands prefixed with lk sip. These include lk sip inbound create and lk sip inbound list for managing inbound trunks, lk sip outbound create and lk sip outbound list for managing outbound trunks, lk sip dispatch create and lk sip dispatch list for managing dispatch rules, and lk sip participant create for creating SIP participants to make outbound calls. Install the CLI from the LiveKit CLI getting started guide and keep it updated regularly.
