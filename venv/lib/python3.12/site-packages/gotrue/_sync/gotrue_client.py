from __future__ import annotations

from contextlib import suppress
from functools import partial
from json import loads
from time import time
from typing import Callable, Dict, List, Tuple, Union
from urllib.parse import parse_qs, urlencode, urlparse
from uuid import uuid4

from ..constants import (
    DEFAULT_HEADERS,
    EXPIRY_MARGIN,
    GOTRUE_URL,
    MAX_RETRIES,
    RETRY_INTERVAL,
    STORAGE_KEY,
)
from ..errors import (
    AuthApiError,
    AuthImplicitGrantRedirectError,
    AuthInvalidCredentialsError,
    AuthRetryableError,
    AuthSessionMissingError,
)
from ..helpers import (
    decode_jwt_payload,
    generate_pkce_challenge,
    generate_pkce_verifier,
    model_dump,
    model_dump_json,
    model_validate,
    parse_auth_otp_response,
    parse_auth_response,
    parse_sso_response,
    parse_user_response,
)
from ..http_clients import SyncClient
from ..timer import Timer
from ..types import (
    AuthChangeEvent,
    AuthenticatorAssuranceLevels,
    AuthFlowType,
    AuthMFAChallengeResponse,
    AuthMFAEnrollResponse,
    AuthMFAGetAuthenticatorAssuranceLevelResponse,
    AuthMFAListFactorsResponse,
    AuthMFAUnenrollResponse,
    AuthMFAVerifyResponse,
    AuthOtpResponse,
    AuthResponse,
    CodeExchangeParams,
    DecodedJWTDict,
    IdentitiesResponse,
    MFAChallengeAndVerifyParams,
    MFAChallengeParams,
    MFAEnrollParams,
    MFAUnenrollParams,
    MFAVerifyParams,
    OAuthResponse,
    Options,
    Provider,
    Session,
    SignInWithOAuthCredentials,
    SignInWithPasswordCredentials,
    SignInWithPasswordlessCredentials,
    SignOutOptions,
    SignUpWithPasswordCredentials,
    Subscription,
    UserAttributes,
    UserResponse,
    VerifyOtpParams,
)
from .gotrue_admin_api import SyncGoTrueAdminAPI
from .gotrue_base_api import SyncGoTrueBaseAPI
from .gotrue_mfa_api import SyncGoTrueMFAAPI
from .storage import SyncMemoryStorage, SyncSupportedStorage


class SyncGoTrueClient(SyncGoTrueBaseAPI):
    def __init__(
        self,
        *,
        url: Union[str, None] = None,
        headers: Union[Dict[str, str], None] = None,
        storage_key: Union[str, None] = None,
        auto_refresh_token: bool = True,
        persist_session: bool = True,
        storage: Union[SyncSupportedStorage, None] = None,
        http_client: Union[SyncClient, None] = None,
        flow_type: AuthFlowType = "implicit",
    ) -> None:
        SyncGoTrueBaseAPI.__init__(
            self,
            url=url or GOTRUE_URL,
            headers=headers or DEFAULT_HEADERS,
            http_client=http_client,
        )
        self._storage_key = storage_key or STORAGE_KEY
        self._auto_refresh_token = auto_refresh_token
        self._persist_session = persist_session
        self._storage = storage or SyncMemoryStorage()
        self._in_memory_session: Union[Session, None] = None
        self._refresh_token_timer: Union[Timer, None] = None
        self._network_retries = 0
        self._state_change_emitters: Dict[str, Subscription] = {}
        self._flow_type = flow_type

        self.admin = SyncGoTrueAdminAPI(
            url=self._url,
            headers=self._headers,
            http_client=self._http_client,
        )
        self.mfa = SyncGoTrueMFAAPI()
        self.mfa.challenge = self._challenge
        self.mfa.challenge_and_verify = self._challenge_and_verify
        self.mfa.enroll = self._enroll
        self.mfa.get_authenticator_assurance_level = (
            self._get_authenticator_assurance_level
        )
        self.mfa.list_factors = self._list_factors
        self.mfa.unenroll = self._unenroll
        self.mfa.verify = self._verify

    # Initializations

    def initialize(self, *, url: Union[str, None] = None) -> None:
        if url and self._is_implicit_grant_flow(url):
            self.initialize_from_url(url)
        else:
            self.initialize_from_storage()

    def initialize_from_storage(self) -> None:
        return self._recover_and_refresh()

    def initialize_from_url(self, url: str) -> None:
        try:
            if self._is_implicit_grant_flow(url):
                session, redirect_type = self._get_session_from_url(url)
                self._save_session(session)
                self._notify_all_subscribers("SIGNED_IN", session)
                if redirect_type == "recovery":
                    self._notify_all_subscribers("PASSWORD_RECOVERY", session)
        except Exception as e:
            self._remove_session()
            raise e

    # Public methods

    def sign_up(
        self,
        credentials: SignUpWithPasswordCredentials,
    ) -> AuthResponse:
        """
        Creates a new user.
        """
        self._remove_session()
        email = credentials.get("email")
        phone = credentials.get("phone")
        password = credentials.get("password")
        options = credentials.get("options", {})
        redirect_to = options.get("redirect_to")
        data = options.get("data") or {}
        captcha_token = options.get("captcha_token")
        if email:
            response = self._request(
                "POST",
                "signup",
                body={
                    "email": email,
                    "password": password,
                    "data": data,
                    "gotrue_meta_security": {
                        "captcha_token": captcha_token,
                    },
                },
                redirect_to=redirect_to,
                xform=parse_auth_response,
            )
        elif phone:
            response = self._request(
                "POST",
                "signup",
                body={
                    "phone": phone,
                    "password": password,
                    "data": data,
                    "gotrue_meta_security": {
                        "captcha_token": captcha_token,
                    },
                },
                xform=parse_auth_response,
            )
        else:
            raise AuthInvalidCredentialsError(
                "You must provide either an email or phone number and a password"
            )
        if response.session:
            self._save_session(response.session)
            self._notify_all_subscribers("SIGNED_IN", response.session)
        return response

    def sign_in_with_password(
        self,
        credentials: SignInWithPasswordCredentials,
    ) -> AuthResponse:
        """
        Log in an existing user with an email or phone and password.
        """
        self._remove_session()
        email = credentials.get("email")
        phone = credentials.get("phone")
        password = credentials.get("password")
        options = credentials.get("options", {})
        data = options.get("data") or {}
        captcha_token = options.get("captcha_token")
        if email:
            response = self._request(
                "POST",
                "token",
                body={
                    "email": email,
                    "password": password,
                    "data": data,
                    "gotrue_meta_security": {
                        "captcha_token": captcha_token,
                    },
                },
                query={
                    "grant_type": "password",
                },
                xform=parse_auth_response,
            )
        elif phone:
            response = self._request(
                "POST",
                "token",
                body={
                    "phone": phone,
                    "password": password,
                    "data": data,
                    "gotrue_meta_security": {
                        "captcha_token": captcha_token,
                    },
                },
                query={
                    "grant_type": "password",
                },
                xform=parse_auth_response,
            )
        else:
            raise AuthInvalidCredentialsError(
                "You must provide either an email or phone number and a password"
            )
        if response.session:
            self._save_session(response.session)
            self._notify_all_subscribers("SIGNED_IN", response.session)
        return response

    def sign_in_with_sso(self, credentials: SignInWithSSOCredentials):
        """
        Attempts a single-sign on using an enterprise Identity Provider. A
        successful SSO attempt will redirect the current page to the identity
        provider authorization page. The redirect URL is implementation and SSO
        protocol specific.

        You can use it by providing a SSO domain. Typically you can extract this
        domain by asking users for their email address. If this domain is
        registered on the Auth instance the redirect will use that organization's
        currently active SSO Identity Provider for the login.
        If you have built an organization-specific login page, you can use the
        organization's SSO Identity Provider UUID directly instead.
        """
        self._remove_session()
        provider_id = credentials.get("provider_id")
        domain = credentials.get("domain")
        options = credentials.get("options", {})
        redirect_to = options.get("redirect_to")
        captcha_token = options.get("captcha_token")
        # HTTPX currently does not follow redirects: https://www.python-httpx.org/compatibility/
        # Additionally, unlike the JS client, Python is a server side language and it's not possible
        # to automatically redirect in browser for hte user
        skip_http_redirect = options.get("skip_http_redirect", True)

        if domain:
            return self._request(
                "POST",
                "sso",
                body={
                    "domain": domain,
                    "skip_http_redirect": skip_http_redirect,
                    "gotrue_meta_security": {
                        "captcha_token": captcha_token,
                    },
                },
                redirect_to=redirect_to,
                xform=parse_sso_response,
            )
        if provider_id:
            return self._request(
                "POST",
                "sso",
                body={
                    "provider_id": provider_id,
                    "skip_http_redirect": skip_http_redirect,
                    "gotrue_meta_security": {
                        "captcha_token": captcha_token,
                    },
                },
                redirect_to=redirect_to,
                xform=parse_sso_response,
            )
        raise AuthInvalidCredentialsError(
            "You must provide either a domain or provider_id"
        )

    def sign_in_with_oauth(
        self,
        credentials: SignInWithOAuthCredentials,
    ) -> OAuthResponse:
        """
        Log in an existing user via a third-party provider.
        """
        self._remove_session()

        provider = credentials.get("provider")
        options = credentials.get("options", {})
        redirect_to = options.get("redirect_to")
        scopes = options.get("scopes")
        params = options.get("query_params", {})
        if redirect_to:
            params["redirect_to"] = redirect_to
        if scopes:
            params["scopes"] = scopes
        url = self._get_url_for_provider(provider, params)
        return OAuthResponse(provider=provider, url=url)

    def link_identity(self, credentials):
        provider = credentials.get("provider")
        options = credentials.get("options", {})
        redirect_to = options.get("redirect_to")
        scopes = options.get("scopes")
        params = options.get("query_params", {})
        if redirect_to:
            params["redirect_to"] = redirect_to
        if scopes:
            params["scopes"] = scopes
        params["skip_browser_redirect"] = True

        url = self._get_url_for_provider(provider, params)
        return OAuthResponse(provider=provider, url=url)

    def get_user_identities(self):
        response = self.get_user()
        return (
            IdentitiesResponse(identities=response.user.identities)
            if response.user
            else AuthSessionMissingError()
        )

    def unlink_identity(self, identity):
        return self._request(
            "POST",
            f"/user/identities/{identity.id}",
        )

    def sign_in_with_otp(
        self,
        credentials: SignInWithPasswordlessCredentials,
    ) -> AuthOtpResponse:
        """
        Log in a user using magiclink or a one-time password (OTP).

        If the `{{ .ConfirmationURL }}` variable is specified in
        the email template, a magiclink will be sent.

        If the `{{ .Token }}` variable is specified in the email
        template, an OTP will be sent.

        If you're using phone sign-ins, only an OTP will be sent.
        You won't be able to send a magiclink for phone sign-ins.
        """
        self._remove_session()
        email = credentials.get("email")
        phone = credentials.get("phone")
        options = credentials.get("options", {})
        email_redirect_to = options.get("email_redirect_to")
        should_create_user = options.get("create_user", True)
        data = options.get("data")
        captcha_token = options.get("captcha_token")
        if email:
            return self._request(
                "POST",
                "otp",
                body={
                    "email": email,
                    "data": data,
                    "create_user": should_create_user,
                    "gotrue_meta_security": {
                        "captcha_token": captcha_token,
                    },
                },
                redirect_to=email_redirect_to,
                xform=parse_auth_otp_response,
            )
        if phone:
            return self._request(
                "POST",
                "otp",
                body={
                    "phone": phone,
                    "data": data,
                    "create_user": should_create_user,
                    "gotrue_meta_security": {
                        "captcha_token": captcha_token,
                    },
                },
                xform=parse_auth_otp_response,
            )
        raise AuthInvalidCredentialsError(
            "You must provide either an email or phone number"
        )

    def verify_otp(self, params: VerifyOtpParams) -> AuthResponse:
        """
        Log in a user given a User supplied OTP received via mobile.
        """
        self._remove_session()
        response = self._request(
            "POST",
            "verify",
            body={
                "gotrue_meta_security": {
                    "captcha_token": params.get("options", {}).get("captcha_token"),
                },
                **params,
            },
            redirect_to=params.get("options", {}).get("redirect_to"),
            xform=parse_auth_response,
        )
        if response.session:
            self._save_session(response.session)
            self._notify_all_subscribers("SIGNED_IN", response.session)
        return response

    def get_session(self) -> Union[Session, None]:
        """
        Returns the session, refreshing it if necessary.

        The session returned can be null if the session is not detected which
        can happen in the event a user is not signed-in or has logged out.
        """
        current_session: Union[Session, None] = None
        if self._persist_session:
            maybe_session = self._storage.get_item(self._storage_key)
            current_session = self._get_valid_session(maybe_session)
            if not current_session:
                self._remove_session()
        else:
            current_session = self._in_memory_session
        if not current_session:
            return None
        time_now = round(time())
        has_expired = (
            current_session.expires_at <= time_now + EXPIRY_MARGIN
            if current_session.expires_at
            else False
        )
        return (
            self._call_refresh_token(current_session.refresh_token)
            if has_expired
            else current_session
        )

    def get_user(self, jwt: Union[str, None] = None) -> Union[UserResponse, None]:
        """
        Gets the current user details if there is an existing session.

        Takes in an optional access token `jwt`. If no `jwt` is provided,
        `get_user()` will attempt to get the `jwt` from the current session.
        """
        if not jwt:
            session = self.get_session()
            if session:
                jwt = session.access_token
            else:
                return None
        return self._request("GET", "user", jwt=jwt, xform=parse_user_response)

    def update_user(self, attributes: UserAttributes) -> UserResponse:
        """
        Updates user data, if there is a logged in user.
        """
        session = self.get_session()
        if not session:
            raise AuthSessionMissingError()
        response = self._request(
            "PUT",
            "user",
            body=attributes,
            jwt=session.access_token,
            xform=parse_user_response,
        )
        session.user = response.user
        self._save_session(session)
        self._notify_all_subscribers("USER_UPDATED", session)
        return response

    def set_session(self, access_token: str, refresh_token: str) -> AuthResponse:
        """
        Sets the session data from the current session. If the current session
        is expired, `set_session` will take care of refreshing it to obtain a
        new session.

        If the refresh token in the current session is invalid and the current
        session has expired, an error will be thrown.

        If the current session does not contain at `expires_at` field,
        `set_session` will use the exp claim defined in the access token.

        The current session that minimally contains an access token,
        refresh token and a user.
        """
        time_now = round(time())
        expires_at = time_now
        has_expired = True
        session: Union[Session, None] = None
        if access_token and access_token.split(".")[1]:
            payload = self._decode_jwt(access_token)
            exp = payload.get("exp")
            if exp:
                expires_at = int(exp)
                has_expired = expires_at <= time_now
        if has_expired:
            if not refresh_token:
                raise AuthSessionMissingError()
            response = self._refresh_access_token(refresh_token)
            if not response.session:
                return AuthResponse()
            session = response.session
        else:
            response = self.get_user(access_token)
            session = Session(
                access_token=access_token,
                refresh_token=refresh_token,
                user=response.user,
                token_type="bearer",
                expires_in=expires_at - time_now,
                expires_at=expires_at,
            )
        self._save_session(session)
        self._notify_all_subscribers("TOKEN_REFRESHED", session)
        return AuthResponse(session=session, user=response.user)

    def refresh_session(
        self, refresh_token: Union[str, None] = None
    ) -> AuthResponse:
        """
        Returns a new session, regardless of expiry status.

        Takes in an optional current session. If not passed in, then refreshSession()
        will attempt to retrieve it from getSession(). If the current session's
        refresh token is invalid, an error will be thrown.
        """
        if not refresh_token:
            session = self.get_session()
            if session:
                refresh_token = session.refresh_token
        if not refresh_token:
            raise AuthSessionMissingError()
        session = self._call_refresh_token(refresh_token)
        return AuthResponse(session=session, user=session.user)

    def sign_out(self, options: SignOutOptions = {"scope": "global"}) -> None:
        """
        Inside a browser context, `sign_out` will remove the logged in user from the
        browser session and log them out - removing all items from localstorage and
        then trigger a `"SIGNED_OUT"` event.

        For server-side management, you can revoke all refresh tokens for a user by
        passing a user's JWT through to `api.sign_out`.

        There is no way to revoke a user's access token jwt until it expires.
        It is recommended to set a shorter expiry on the jwt for this reason.
        """
        with suppress(AuthApiError):
            session = self.get_session()
            access_token = session.access_token if session else None
            if access_token:
                self.admin.sign_out(access_token, options["scope"])

            if options["scope"] != "others":
                self._remove_session()
                self._notify_all_subscribers("SIGNED_OUT", None)

    def on_auth_state_change(
        self,
        callback: Callable[[AuthChangeEvent, Union[Session, None]], None],
    ) -> Subscription:
        """
        Receive a notification every time an auth event happens.
        """
        unique_id = str(uuid4())

        def _unsubscribe() -> None:
            self._state_change_emitters.pop(unique_id)

        subscription = Subscription(
            id=unique_id,
            callback=callback,
            unsubscribe=_unsubscribe,
        )
        self._state_change_emitters[unique_id] = subscription
        return subscription

    def reset_password_email(
        self,
        email: str,
        options: Options = {},
    ) -> None:
        """
        Sends a password reset request to an email address.
        """
        self._request(
            "POST",
            "recover",
            body={
                "email": email,
                "gotrue_meta_security": {
                    "captcha_token": options.get("captcha_token"),
                },
            },
            redirect_to=options.get("redirect_to"),
        )

    # MFA methods

    def _enroll(self, params: MFAEnrollParams) -> AuthMFAEnrollResponse:
        session = self.get_session()
        if not session:
            raise AuthSessionMissingError()
        response = self._request(
            "POST",
            "factors",
            body=params,
            jwt=session.access_token,
            xform=partial(model_validate, AuthMFAEnrollResponse),
        )
        if response.totp.qr_code:
            response.totp.qr_code = f"data:image/svg+xml;utf-8,{response.totp.qr_code}"
        return response

    def _challenge(self, params: MFAChallengeParams) -> AuthMFAChallengeResponse:
        session = self.get_session()
        if not session:
            raise AuthSessionMissingError()
        return self._request(
            "POST",
            f"factors/{params.get('factor_id')}/challenge",
            jwt=session.access_token,
            xform=partial(model_validate, AuthMFAChallengeResponse),
        )

    def _challenge_and_verify(
        self,
        params: MFAChallengeAndVerifyParams,
    ) -> AuthMFAVerifyResponse:
        response = self._challenge(
            {
                "factor_id": params.get("factor_id"),
            }
        )
        return self._verify(
            {
                "factor_id": params.get("factor_id"),
                "challenge_id": response.id,
                "code": params.get("code"),
            }
        )

    def _verify(self, params: MFAVerifyParams) -> AuthMFAVerifyResponse:
        session = self.get_session()
        if not session:
            raise AuthSessionMissingError()
        response = self._request(
            "POST",
            f"factors/{params.get('factor_id')}/verify",
            body=params,
            jwt=session.access_token,
            xform=partial(model_validate, AuthMFAVerifyResponse),
        )
        session = model_validate(Session, model_dump(response))
        self._save_session(session)
        self._notify_all_subscribers("MFA_CHALLENGE_VERIFIED", session)
        return response

    def _unenroll(self, params: MFAUnenrollParams) -> AuthMFAUnenrollResponse:
        session = self.get_session()
        if not session:
            raise AuthSessionMissingError()
        return self._request(
            "DELETE",
            f"factors/{params.get('factor_id')}",
            jwt=session.access_token,
            xform=partial(AuthMFAUnenrollResponse, model_validate),
        )

    def _list_factors(self) -> AuthMFAListFactorsResponse:
        response = self.get_user()
        all = response.user.factors or []
        totp = [f for f in all if f.factor_type == "totp" and f.status == "verified"]
        return AuthMFAListFactorsResponse(all=all, totp=totp)

    def _get_authenticator_assurance_level(
        self,
    ) -> AuthMFAGetAuthenticatorAssuranceLevelResponse:
        session = self.get_session()
        if not session:
            return AuthMFAGetAuthenticatorAssuranceLevelResponse(
                current_level=None,
                next_level=None,
                current_authentication_methods=[],
            )
        payload = self._decode_jwt(session.access_token)
        current_level: Union[AuthenticatorAssuranceLevels, None] = None
        if payload.get("aal"):
            current_level = payload.get("aal")
        verified_factors = [
            f for f in session.user.factors or [] if f.status == "verified"
        ]
        next_level = "aal2" if verified_factors else current_level
        current_authentication_methods = payload.get("amr") or []
        return AuthMFAGetAuthenticatorAssuranceLevelResponse(
            current_level=current_level,
            next_level=next_level,
            current_authentication_methods=current_authentication_methods,
        )

    # Private methods

    def _remove_session(self) -> None:
        if self._persist_session:
            self._storage.remove_item(self._storage_key)
        else:
            self._in_memory_session = None
        if self._refresh_token_timer:
            self._refresh_token_timer.cancel()
            self._refresh_token_timer = None

    def _get_session_from_url(
        self,
        url: str,
    ) -> Tuple[Session, Union[str, None]]:
        if not self._is_implicit_grant_flow(url):
            raise AuthImplicitGrantRedirectError("Not a valid implicit grant flow url.")
        result = urlparse(url)
        params = parse_qs(result.query)
        error_description = self._get_param(params, "error_description")
        if error_description:
            error_code = self._get_param(params, "error_code")
            error = self._get_param(params, "error")
            if not error_code:
                raise AuthImplicitGrantRedirectError("No error_code detected.")
            if not error:
                raise AuthImplicitGrantRedirectError("No error detected.")
            raise AuthImplicitGrantRedirectError(
                error_description,
                {"code": error_code, "error": error},
            )
        provider_token = self._get_param(params, "provider_token")
        provider_refresh_token = self._get_param(params, "provider_refresh_token")
        access_token = self._get_param(params, "access_token")
        if not access_token:
            raise AuthImplicitGrantRedirectError("No access_token detected.")
        expires_in = self._get_param(params, "expires_in")
        if not expires_in:
            raise AuthImplicitGrantRedirectError("No expires_in detected.")
        refresh_token = self._get_param(params, "refresh_token")
        if not refresh_token:
            raise AuthImplicitGrantRedirectError("No refresh_token detected.")
        token_type = self._get_param(params, "token_type")
        if not token_type:
            raise AuthImplicitGrantRedirectError("No token_type detected.")
        time_now = round(time())
        expires_at = time_now + int(expires_in)
        user = self.get_user(access_token)
        session = Session(
            provider_token=provider_token,
            provider_refresh_token=provider_refresh_token,
            access_token=access_token,
            expires_in=int(expires_in),
            expires_at=expires_at,
            refresh_token=refresh_token,
            token_type=token_type,
            user=user.user,
        )
        redirect_type = self._get_param(params, "type")
        return session, redirect_type

    def _recover_and_refresh(self) -> None:
        raw_session = self._storage.get_item(self._storage_key)
        current_session = self._get_valid_session(raw_session)
        if not current_session:
            if raw_session:
                self._remove_session()
            return
        time_now = round(time())
        expires_at = current_session.expires_at
        if expires_at and expires_at < time_now + EXPIRY_MARGIN:
            refresh_token = current_session.refresh_token
            if self._auto_refresh_token and refresh_token:
                self._network_retries += 1
                try:
                    self._call_refresh_token(refresh_token)
                    self._network_retries = 0
                except Exception as e:
                    if (
                        isinstance(e, AuthRetryableError)
                        and self._network_retries < MAX_RETRIES
                    ):
                        if self._refresh_token_timer:
                            self._refresh_token_timer.cancel()
                        self._refresh_token_timer = Timer(
                            (RETRY_INTERVAL ** (self._network_retries * 100)),
                            self._recover_and_refresh,
                        )
                        self._refresh_token_timer.start()
                        return
            self._remove_session()
            return
        if self._persist_session:
            self._save_session(current_session)
        self._notify_all_subscribers("SIGNED_IN", current_session)

    def _call_refresh_token(self, refresh_token: str) -> Session:
        if not refresh_token:
            raise AuthSessionMissingError()
        response = self._refresh_access_token(refresh_token)
        if not response.session:
            raise AuthSessionMissingError()
        self._save_session(response.session)
        self._notify_all_subscribers("TOKEN_REFRESHED", response.session)
        return response.session

    def _refresh_access_token(self, refresh_token: str) -> AuthResponse:
        return self._request(
            "POST",
            "token",
            query={"grant_type": "refresh_token"},
            body={"refresh_token": refresh_token},
            xform=parse_auth_response,
        )

    def _save_session(self, session: Session) -> None:
        if not self._persist_session:
            self._in_memory_session = session
        expire_at = session.expires_at
        if expire_at:
            time_now = round(time())
            expire_in = expire_at - time_now
            refresh_duration_before_expires = (
                EXPIRY_MARGIN if expire_in > EXPIRY_MARGIN else 0.5
            )
            value = (expire_in - refresh_duration_before_expires) * 1000
            self._start_auto_refresh_token(value)
        if self._persist_session and session.expires_at:
            self._storage.set_item(self._storage_key, model_dump_json(session))

    def _start_auto_refresh_token(self, value: float) -> None:
        if self._refresh_token_timer:
            self._refresh_token_timer.cancel()
            self._refresh_token_timer = None
        if value <= 0 or not self._auto_refresh_token:
            return

        def refresh_token_function():
            self._network_retries += 1
            try:
                session = self.get_session()
                if session:
                    self._call_refresh_token(session.refresh_token)
                    self._network_retries = 0
            except Exception as e:
                if (
                    isinstance(e, AuthRetryableError)
                    and self._network_retries < MAX_RETRIES
                ):
                    self._start_auto_refresh_token(
                        RETRY_INTERVAL ** (self._network_retries * 100)
                    )

        self._refresh_token_timer = Timer(value, refresh_token_function)
        self._refresh_token_timer.start()

    def _notify_all_subscribers(
        self,
        event: AuthChangeEvent,
        session: Union[Session, None],
    ) -> None:
        for subscription in self._state_change_emitters.values():
            subscription.callback(event, session)

    def _get_valid_session(
        self,
        raw_session: Union[str, None],
    ) -> Union[Session, None]:
        if not raw_session:
            return None
        data = loads(raw_session)
        if not data:
            return None
        if not data.get("access_token"):
            return None
        if not data.get("refresh_token"):
            return None
        if not data.get("expires_at"):
            return None
        try:
            expires_at = int(data["expires_at"])
            data["expires_at"] = expires_at
        except ValueError:
            return None
        try:
            return model_validate(Session, data)
        except Exception:
            return None

    def _get_param(
        self,
        query_params: Dict[str, List[str]],
        name: str,
    ) -> Union[str, None]:
        return query_params[name][0] if name in query_params else None

    def _is_implicit_grant_flow(self, url: str) -> bool:
        result = urlparse(url)
        params = parse_qs(result.query)
        return "access_token" in params or "error_description" in params

    def _get_url_for_provider(
        self,
        provider: Provider,
        params: Dict[str, str],
    ) -> str:
        if self._flow_type == "pkce":
            code_verifier = generate_pkce_verifier()
            code_challenge = generate_pkce_challenge(code_verifier)
            self._storage.set_item(
                f"{self._storage_key}-code-verifier", code_verifier
            )
            code_challenge_method = (
                "plain" if code_verifier == code_challenge else "s256"
            )
            params["code_challenge"] = code_challenge
            params["code_challenge_method"] = code_challenge_method

        params["provider"] = provider
        query = urlencode(params)
        return f"{self._url}/authorize?{query}"

    def _decode_jwt(self, jwt: str) -> DecodedJWTDict:
        """
        Decodes a JWT (without performing any validation).
        """
        return decode_jwt_payload(jwt)

    def exchange_code_for_session(self, params: CodeExchangeParams):
        code_verifier = params.get("code_verifier") or self._storage.get_item(
            f"{self._storage_key}-code-verifier"
        )
        response = self._request(
            "POST",
            "token?grant_type=pkce",
            body={
                "auth_code": params.get("auth_code"),
                "code_verifier": code_verifier,
            },
            redirect_to=params.get("redirect_to"),
            xform=parse_auth_response,
        )
        self._storage.remove_item(f"{self._storage_key}-code-verifier")
        if response.session:
            self._save_session(response.session)
            self._notify_all_subscribers("SIGNED_IN", response.session)
        return response
