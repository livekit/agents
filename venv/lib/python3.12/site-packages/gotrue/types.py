from __future__ import annotations

from datetime import datetime
from time import time
from typing import Any, Callable, Dict, List, Union

from pydantic import BaseModel

try:
    # > 2
    from pydantic import model_validator

    model_validator_v1_v2_compat = model_validator(mode="before")
except ImportError:
    # < 2
    from pydantic import root_validator

    model_validator_v1_v2_compat = root_validator

from typing_extensions import Literal, NotRequired, TypedDict

Provider = Literal[
    "apple",
    "azure",
    "bitbucket",
    "discord",
    "facebook",
    "figma",
    "github",
    "gitlab",
    "google",
    "kakao",
    "keycloak",
    "linkedin",
    "notion",
    "slack",
    "spotify",
    "twitch",
    "twitter",
    "workos",
    "zoom",
]

EmailOtpType = Literal[
    "signup", "invite", "magiclink", "recovery", "email_change", "email"
]

AuthChangeEventMFA = Literal["MFA_CHALLENGE_VERIFIED"]

AuthFlowType = Literal["pkce", "implicit"]

AuthChangeEvent = Literal[
    "PASSWORD_RECOVERY",
    "SIGNED_IN",
    "SIGNED_OUT",
    "TOKEN_REFRESHED",
    "USER_UPDATED",
    "USER_DELETED",
    AuthChangeEventMFA,
]


class AMREntry(BaseModel):
    """
    An authentication methord reference (AMR) entry.

    An entry designates what method was used by the user to verify their
    identity and at what time.
    """

    method: Union[Literal["password", "otp", "oauth", "mfa/totp"], str]
    """
    Authentication method name.
    """
    timestamp: int
    """
    Timestamp when the method was successfully used. Represents number of
    seconds since 1st January 1970 (UNIX epoch) in UTC.
    """


class Options(TypedDict):
    redirect_to: NotRequired[str]
    data: NotRequired[Any]


class AuthResponse(BaseModel):
    user: Union[User, None] = None
    session: Union[Session, None] = None


class AuthOtpResponse(BaseModel):
    user: None = None
    session: None = None
    message_id: Union[str, None] = None


class OAuthResponse(BaseModel):
    provider: Provider
    url: str


class SSOResponse(BaseModel):
    url: str


class IdentitiesResponse(BaseModel):
    identities: List[UserIdentity]


class UserResponse(BaseModel):
    user: User


class Session(BaseModel):
    provider_token: Union[str, None] = None
    """
    The oauth provider token. If present, this can be used to make external API
    requests to the oauth provider used.
    """
    provider_refresh_token: Union[str, None] = None
    """
    The oauth provider refresh token. If present, this can be used to refresh
    the provider_token via the oauth provider's API.

    Not all oauth providers return a provider refresh token. If the
    provider_refresh_token is missing, please refer to the oauth provider's
    documentation for information on how to obtain the provider refresh token.
    """
    access_token: str
    refresh_token: str
    expires_in: int
    """
    The number of seconds until the token expires (since it was issued).
    Returned when a login is confirmed.
    """
    expires_at: Union[int, None] = None
    """
    A timestamp of when the token will expire. Returned when a login is confirmed.
    """
    token_type: str
    user: User

    @model_validator_v1_v2_compat
    def validator(cls, values: dict) -> dict:
        expires_in = values.get("expires_in")
        if expires_in and not values.get("expires_at"):
            values["expires_at"] = round(time()) + expires_in
        return values


class UserIdentity(BaseModel):
    id: str
    user_id: str
    identity_data: Dict[str, Any]
    provider: str
    created_at: datetime
    last_sign_in_at: Union[datetime, None] = None
    updated_at: Union[datetime, None] = None


class Factor(BaseModel):
    """
    A MFA factor.
    """

    id: str
    """
    ID of the factor.
    """
    friendly_name: Union[str, None] = None
    """
    Friendly name of the factor, useful to disambiguate between multiple factors.
    """
    factor_type: Union[Literal["totp"], str]
    """
    Type of factor. Only `totp` supported with this version but may change in
    future versions.
    """
    status: Literal["verified", "unverified"]
    """
    Factor's status.
    """
    created_at: datetime
    updated_at: datetime


class User(BaseModel):
    id: str
    app_metadata: Dict[str, Any]
    user_metadata: Dict[str, Any]
    aud: str
    confirmation_sent_at: Union[datetime, None] = None
    recovery_sent_at: Union[datetime, None] = None
    email_change_sent_at: Union[datetime, None] = None
    new_email: Union[str, None] = None
    invited_at: Union[datetime, None] = None
    action_link: Union[str, None] = None
    email: Union[str, None] = None
    phone: Union[str, None] = None
    created_at: datetime
    confirmed_at: Union[datetime, None] = None
    email_confirmed_at: Union[datetime, None] = None
    phone_confirmed_at: Union[datetime, None] = None
    last_sign_in_at: Union[datetime, None] = None
    role: Union[str, None] = None
    updated_at: Union[datetime, None] = None
    identities: Union[List[UserIdentity], None] = None
    factors: Union[List[Factor], None] = None


class UserAttributes(TypedDict):
    email: NotRequired[str]
    phone: NotRequired[str]
    password: NotRequired[str]
    data: NotRequired[Any]


class AdminUserAttributes(UserAttributes, TypedDict):
    user_metadata: NotRequired[Any]
    app_metadata: NotRequired[Any]
    email_confirm: NotRequired[bool]
    phone_confirm: NotRequired[bool]
    ban_duration: NotRequired[Union[str, Literal["none"]]]


class Subscription(BaseModel):
    id: str
    """
    The subscriber UUID. This will be set by the client.
    """
    callback: Callable[[AuthChangeEvent, Union[Session, None]], None]
    """
    The function to call every time there is an event.
    """
    unsubscribe: Callable[[], None]
    """
    Call this to remove the listener.
    """


class UpdatableFactorAttributes(TypedDict):
    friendly_name: str


class SignUpWithEmailAndPasswordCredentialsOptions(
    TypedDict,
):
    email_redirect_to: NotRequired[str]
    data: NotRequired[Any]
    captcha_token: NotRequired[str]


class SignUpWithEmailAndPasswordCredentials(TypedDict):
    email: str
    password: str
    options: NotRequired[SignUpWithEmailAndPasswordCredentialsOptions]


class SignUpWithPhoneAndPasswordCredentialsOptions(TypedDict):
    data: NotRequired[Any]
    captcha_token: NotRequired[str]


class SignUpWithPhoneAndPasswordCredentials(TypedDict):
    phone: str
    password: str
    options: NotRequired[SignUpWithPhoneAndPasswordCredentialsOptions]


SignUpWithPasswordCredentials = Union[
    SignUpWithEmailAndPasswordCredentials,
    SignUpWithPhoneAndPasswordCredentials,
]


class SignInWithPasswordCredentialsOptions(TypedDict):
    data: NotRequired[Any]
    captcha_token: NotRequired[str]


class SignInWithEmailAndPasswordCredentials(TypedDict):
    email: str
    password: str
    options: NotRequired[SignInWithPasswordCredentialsOptions]


class SignInWithPhoneAndPasswordCredentials(TypedDict):
    phone: str
    password: str
    options: NotRequired[SignInWithPasswordCredentialsOptions]


SignInWithPasswordCredentials = Union[
    SignInWithEmailAndPasswordCredentials,
    SignInWithPhoneAndPasswordCredentials,
]


class SignInWithEmailAndPasswordlessCredentialsOptions(TypedDict):
    email_redirect_to: NotRequired[str]
    should_create_user: NotRequired[bool]
    data: NotRequired[Any]
    captcha_token: NotRequired[str]


class SignInWithEmailAndPasswordlessCredentials(TypedDict):
    email: str
    options: NotRequired[SignInWithEmailAndPasswordlessCredentialsOptions]


class SignInWithPhoneAndPasswordlessCredentialsOptions(TypedDict):
    should_create_user: NotRequired[bool]
    data: NotRequired[Any]
    captcha_token: NotRequired[str]


class SignInWithPhoneAndPasswordlessCredentials(TypedDict):
    phone: str
    options: NotRequired[SignInWithPhoneAndPasswordlessCredentialsOptions]


SignInWithPasswordlessCredentials = Union[
    SignInWithEmailAndPasswordlessCredentials,
    SignInWithPhoneAndPasswordlessCredentials,
]


class SignInWithOAuthCredentialsOptions(TypedDict):
    redirect_to: NotRequired[str]
    scopes: NotRequired[str]
    query_params: NotRequired[Dict[str, str]]


class SignInWithOAuthCredentials(TypedDict):
    provider: Provider
    options: NotRequired[SignInWithOAuthCredentialsOptions]


class SignInWithSSOCredentials(TypedDict):
    provider_id: NotRequired[str]
    domain: NotRequired[str]
    options: NotRequired[SignInWithSSOOptions]


class SignInWithSSOOptions(TypedDict):
    redirect_to: NotRequired[str]
    skip_http_redirect: NotRequired[bool]


class VerifyOtpParamsOptions(TypedDict):
    redirect_to: NotRequired[str]
    captcha_token: NotRequired[str]


class VerifyEmailOtpParams(TypedDict):
    email: str
    token: str
    type: EmailOtpType
    options: NotRequired[VerifyOtpParamsOptions]


class VerifyMobileOtpParams(TypedDict):
    phone: str
    token: str
    type: Literal[
        "sms",
        "phone_change",
    ]
    options: NotRequired[VerifyOtpParamsOptions]


class VerifyTokenHashParams(TypedDict):
    token_hash: str
    type: EmailOtpType
    options: NotRequired[VerifyOtpParamsOptions]


VerifyOtpParams = Union[
    VerifyEmailOtpParams, VerifyMobileOtpParams, VerifyTokenHashParams
]


class GenerateLinkParamsOptions(TypedDict):
    redirect_to: NotRequired[str]


class GenerateLinkParamsWithDataOptions(GenerateLinkParamsOptions, TypedDict):
    data: NotRequired[Any]


class GenerateSignupLinkParams(TypedDict):
    type: Literal["signup"]
    email: str
    password: str
    options: NotRequired[GenerateLinkParamsWithDataOptions]


class GenerateInviteOrMagiclinkParams(TypedDict):
    type: Literal["invite", "magiclink"]
    email: str
    options: NotRequired[GenerateLinkParamsWithDataOptions]


class GenerateRecoveryLinkParams(TypedDict):
    type: Literal["recovery"]
    email: str
    options: NotRequired[GenerateLinkParamsOptions]


class GenerateEmailChangeLinkParams(TypedDict):
    type: Literal["email_change_current", "email_change_new"]
    email: str
    new_email: str
    options: NotRequired[GenerateLinkParamsOptions]


GenerateLinkParams = Union[
    GenerateSignupLinkParams,
    GenerateInviteOrMagiclinkParams,
    GenerateRecoveryLinkParams,
    GenerateEmailChangeLinkParams,
]

GenerateLinkType = Literal[
    "signup",
    "invite",
    "magiclink",
    "recovery",
    "email_change_current",
    "email_change_new",
]


class MFAEnrollParams(TypedDict):
    factor_type: Literal["totp"]
    issuer: NotRequired[str]
    friendly_name: NotRequired[str]


class MFAUnenrollParams(TypedDict):
    factor_id: str
    """
    ID of the factor being unenrolled.
    """


class CodeExchangeParams(TypedDict):
    code_verifier: str
    """
    Randomly generated string
    """
    auth_code: str
    """
    Code returned after completing one of the authorization flows
    """
    redirect_to: str
    """
    The URL to route to after a session is successfully obtained
    """


class MFAVerifyParams(TypedDict):
    factor_id: str
    """
    ID of the factor being verified.
    """
    challenge_id: str
    """
    ID of the challenge being verified.
    """
    code: str
    """
    Verification code provided by the user.
    """


class MFAChallengeParams(TypedDict):
    factor_id: str
    """
    ID of the factor to be challenged.
    """


class MFAChallengeAndVerifyParams(TypedDict):
    factor_id: str
    """
    ID of the factor being verified.
    """
    code: str
    """
    Verification code provided by the user.
    """


class AuthMFAVerifyResponse(BaseModel):
    access_token: str
    """
    New access token (JWT) after successful verification.
    """
    token_type: str
    """
    Type of token, typically `Bearer`.
    """
    expires_in: int
    """
    Number of seconds in which the access token will expire.
    """
    refresh_token: str
    """
    Refresh token you can use to obtain new access tokens when expired.
    """
    user: User
    """
    Updated user profile.
    """


class AuthMFAEnrollResponseTotp(BaseModel):
    qr_code: str
    """
    Contains a QR code encoding the authenticator URI. You can
    convert it to a URL by prepending `data:image/svg+xml;utf-8,` to
    the value. Avoid logging this value to the console.
    """
    secret: str
    """
    The TOTP secret (also encoded in the QR code). Show this secret
    in a password-style field to the user, in case they are unable to
    scan the QR code. Avoid logging this value to the console.
    """
    uri: str
    """
    The authenticator URI encoded within the QR code, should you need
    to use it. Avoid loggin this value to the console.
    """


class AuthMFAEnrollResponse(BaseModel):
    id: str
    """
    ID of the factor that was just enrolled (in an unverified state).
    """
    type: Literal["totp"]
    """
    Type of MFA factor. Only `totp` supported for now.
    """
    totp: AuthMFAEnrollResponseTotp
    """
    TOTP enrollment information.
    """


class AuthMFAUnenrollResponse(BaseModel):
    id: str
    """
    ID of the factor that was successfully unenrolled.
    """


class AuthMFAChallengeResponse(BaseModel):
    id: str
    """
    ID of the newly created challenge.
    """
    expires_at: int
    """
    Timestamp in UNIX seconds when this challenge will no longer be usable.
    """


class AuthMFAListFactorsResponse(BaseModel):
    all: List[Factor]
    """
    All available factors (verified and unverified).
    """
    totp: List[Factor]
    """
    Only verified TOTP factors. (A subset of `all`.)
    """


AuthenticatorAssuranceLevels = Literal["aal1", "aal2"]


class AuthMFAGetAuthenticatorAssuranceLevelResponse(BaseModel):
    current_level: Union[AuthenticatorAssuranceLevels, None] = None
    """
    Current AAL level of the session.
    """
    next_level: Union[AuthenticatorAssuranceLevels, None] = None
    """
    Next possible AAL level for the session. If the next level is higher
    than the current one, the user should go through MFA.
    """
    current_authentication_methods: List[AMREntry]
    """
    A list of all authentication methods attached to this session. Use
    the information here to detect the last time a user verified a
    factor, for example if implementing a step-up scenario.
    """


class AuthMFAAdminDeleteFactorResponse(BaseModel):
    id: str
    """
    ID of the factor that was successfully deleted.
    """


class AuthMFAAdminDeleteFactorParams(TypedDict):
    id: str
    """
    ID of the MFA factor to delete.
    """
    user_id: str
    """
    ID of the user whose factor is being deleted.
    """


class AuthMFAAdminListFactorsResponse(BaseModel):
    factors: List[Factor]
    """
    All factors attached to the user.
    """


class AuthMFAAdminListFactorsParams(TypedDict):
    user_id: str
    """
    ID of the user for which to list all MFA factors.
    """


class GenerateLinkProperties(BaseModel):
    """
    The properties related to the email link generated.
    """

    action_link: str
    """
    The email link to send to the user. The action_link follows the following format:

    auth/v1/verify?type={verification_type}&token={hashed_token}&redirect_to={redirect_to}
    """
    email_otp: str
    """
    The raw email OTP.
    You should send this in the email if you want your users to verify using an
    OTP instead of the action link.
    """
    hashed_token: str
    """
    The hashed token appended to the action link.
    """
    redirect_to: str
    """
    The URL appended to the action link.
    """
    verification_type: GenerateLinkType
    """
    The verification type that the email link is associated to.
    """


class GenerateLinkResponse(BaseModel):
    properties: GenerateLinkProperties
    user: User


class DecodedJWTDict(TypedDict):
    exp: NotRequired[int]
    aal: NotRequired[Union[AuthenticatorAssuranceLevels, None]]
    amr: NotRequired[Union[List[AMREntry], None]]


SignOutScope = Literal["global", "local", "others"]


class SignOutOptions(TypedDict):
    scope: NotRequired[SignOutScope]


for model in [
    AMREntry,
    AuthResponse,
    OAuthResponse,
    UserResponse,
    Session,
    UserIdentity,
    Factor,
    User,
    Subscription,
    AuthMFAVerifyResponse,
    AuthMFAEnrollResponseTotp,
    AuthMFAEnrollResponse,
    AuthMFAUnenrollResponse,
    AuthMFAChallengeResponse,
    AuthMFAListFactorsResponse,
    AuthMFAGetAuthenticatorAssuranceLevelResponse,
    AuthMFAAdminDeleteFactorResponse,
    AuthMFAAdminListFactorsResponse,
    GenerateLinkProperties,
]:
    try:
        # pydantic > 2
        model.model_rebuild()
    except AttributeError:
        # pydantic < 2
        model.update_forward_refs()
