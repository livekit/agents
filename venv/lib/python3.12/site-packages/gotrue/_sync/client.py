from __future__ import annotations

from functools import partial
from json import dumps, loads
from threading import Timer
from time import time
from typing import Any, Callable, Dict, Optional, Tuple, Union, cast
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from ..constants import COOKIE_OPTIONS, DEFAULT_HEADERS, GOTRUE_URL, STORAGE_KEY
from ..exceptions import APIError
from ..helpers import model_dump, model_validate
from ..types import (
    AuthChangeEvent,
    CookieOptions,
    Provider,
    Session,
    Subscription,
    User,
    UserAttributes,
    UserAttributesDict,
)
from .api import SyncGoTrueAPI
from .storage import SyncMemoryStorage, SyncSupportedStorage


class SyncGoTrueClient:
    def __init__(
        self,
        *,
        url: str = GOTRUE_URL,
        headers: Dict[str, str] = {},
        auto_refresh_token: bool = True,
        persist_session: bool = True,
        local_storage: SyncSupportedStorage = SyncMemoryStorage(),
        cookie_options: CookieOptions = CookieOptions.parse_obj(COOKIE_OPTIONS),
        api: Optional[SyncGoTrueAPI] = None,
        replace_default_headers: bool = False,
    ) -> None:
        """Create a new client

        url : str
            The URL of the GoTrue server.
        headers : Dict[str, str]
            Any additional headers to send to the GoTrue server.
        auto_refresh_token : bool
            Set to "true" if you want to automatically refresh the token before
            expiring.
        persist_session : bool
            Set to "true" if you want to automatically save the user session
            into local storage.
        local_storage : SupportedStorage
            The storage engine to use for persisting the session.
        cookie_options : CookieOptions
            The options for the cookie.
        """
        if url.startswith("http://"):
            print(
                "Warning:\n\nDO NOT USE HTTP IN PRODUCTION FOR GOTRUE EVER!\n"
                "GoTrue REQUIRES HTTPS to work securely."
            )
        self.state_change_emitters: Dict[str, Subscription] = {}
        self.refresh_token_timer: Optional[Timer] = None
        self.current_user: Optional[User] = None
        self.current_session: Optional[Session] = None
        self.auto_refresh_token = auto_refresh_token
        self.persist_session = persist_session
        self.local_storage = local_storage
        empty_or_default_headers = {} if replace_default_headers else DEFAULT_HEADERS
        args = {
            "url": url,
            "headers": {**empty_or_default_headers, **headers},
            "cookie_options": cookie_options,
        }
        self.api = api or SyncGoTrueAPI(**args)

    def __enter__(self) -> SyncGoTrueClient:
        return self

    def __exit__(self, exc_t, exc_v, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        self.api.close()

    def init_recover(self) -> None:
        """Recover the current session from local storage."""
        self._recover_session()
        self._recover_and_refresh()

    def sign_up(
        self,
        *,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        password: Optional[str] = None,
        redirect_to: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Union[Session, User]:
        """Creates a new user. If email and phone are provided, email will be
        used and phone will be ignored.

        Parameters
        ---------
        email : Optional[str]
            The user's email address.
        phone : Optional[str]
            The user's phone number.
        password : Optional[str]
            The user's password.
        redirect_to : Optional[str]
            A URL or mobile address to send the user to after they are confirmed.
        data : Optional[Dict[str, Any]]
            Optional user metadata.

        Returns
        -------
        response : Union[Session, User]
            A logged-in session if the server has "autoconfirm" ON
            A user if the server has "autoconfirm" OFF

        Raises
        ------
        APIError
            If an error occurs.
        """
        self._remove_session()

        if email and password:
            response = self.api.sign_up_with_email(
                email=email,
                password=password,
                redirect_to=redirect_to,
                data=data,
            )
        elif phone and password:
            response = self.api.sign_up_with_phone(
                phone=phone, password=password, data=data
            )
        elif not password:
            raise ValueError("Password must be defined, can't be None.")
        else:
            raise ValueError("Email or phone must be defined, both can't be None.")

        if isinstance(response, Session):
            # The user has confirmed their email or the underlying DB doesn't
            # require email confirmation.
            self._save_session(session=response)
            self._notify_all_subscribers(event=AuthChangeEvent.SIGNED_IN)
        return response

    def sign_in(
        self,
        *,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        password: Optional[str] = None,
        refresh_token: Optional[str] = None,
        provider: Optional[Provider] = None,
        redirect_to: Optional[str] = None,
        scopes: Optional[str] = None,
        create_user: bool = False,
    ) -> Optional[Union[Session, str]]:
        """Log in an existing user, or login via a third-party provider.
        If email and phone are provided, email will be used and phone will be ignored.

        Parameters
        ---------
        email : Optional[str]
            The user's email address.
        phone : Optional[str]
            The user's phone number.
        password : Optional[str]
            The user's password.
        refresh_token : Optional[str]
            A valid refresh token that was returned on login.
        provider : Optional[Provider]
            One of the providers supported by GoTrue.
        redirect_to : Optional[str]
            A URL or mobile address to send the user to after they are confirmed.
        scopes : Optional[str]
            A space-separated list of scopes granted to the OAuth application.

        Returns
        -------
        response : Optional[Union[Session, str]]
            If only email are provided between the email and password,
            None is returned and send magic link to email

            If email and password are provided, a logged-in session is returned.

            If only phone are provided between the phone and password,
            None is returned and send message to phone

            If phone and password are provided, a logged-in session is returned.

            If refresh_token is provided, a logged-in session is returned.

            If provider is provided, an redirect URL is returned.

            Otherwise, error is raised.

        Raises
        ------
        APIError
            If an error occurs.
        """
        self._remove_session()
        if email:
            if password:
                response = self._handle_email_sign_in(
                    email=email,
                    password=password,
                    redirect_to=redirect_to,
                )
            else:
                response = self.api.send_magic_link_email(
                    email=email, create_user=create_user
                )
        elif phone:
            if password:
                response = self._handle_phone_sign_in(phone=phone, password=password)
            else:
                response = self.api.send_mobile_otp(
                    phone=phone, create_user=create_user
                )
        elif refresh_token:
            # current_session and current_user will be updated to latest
            # on _call_refresh_token using the passed refresh_token
            self._call_refresh_token(refresh_token=refresh_token)
            response = self.current_session
        elif provider:
            response = self._handle_provider_sign_in(
                provider=provider,
                redirect_to=redirect_to,
                scopes=scopes,
            )
        else:
            raise ValueError(
                "Email, phone, refresh_token, or provider must be defined, "
                "all can't be None."
            )
        return response

    def verify_otp(
        self,
        *,
        phone: str,
        token: str,
        redirect_to: Optional[str] = None,
    ) -> Union[Session, User]:
        """Log in a user given a User supplied OTP received via mobile.

        Parameters
        ----------
        phone : str
            The user's phone number.
        token : str
            The user's OTP.
        redirect_to : Optional[str]
            A URL or mobile address to send the user to after they are confirmed.

        Returns
        -------
        response : Union[Session, User]
            A logged-in session if the server has "autoconfirm" ON
            A user if the server has "autoconfirm" OFF

        Raises
        ------
        APIError
            If an error occurs.
        """
        self._remove_session()
        response = self.api.verify_mobile_otp(
            phone=phone,
            token=token,
            redirect_to=redirect_to,
        )
        if isinstance(response, Session):
            self._save_session(session=response)
            self._notify_all_subscribers(event=AuthChangeEvent.SIGNED_IN)
        return response

    def user(self) -> Optional[User]:
        """Returns the user data, if there is a logged in user."""
        return self.current_user

    def session(self) -> Optional[Session]:
        """Returns the session data, if there is an active session."""
        return self.current_session

    def refresh_session(self) -> Session:
        """Force refreshes the session.

        Force refreshes the session including the user data incase it was
        updated in a different session.
        """
        if not self.current_session:
            raise ValueError("Not logged in.")
        return self._call_refresh_token()

    def update(self, *, attributes: Union[UserAttributesDict, UserAttributes]) -> User:
        """Updates user data, if there is a logged in user.

        Parameters
        ----------
        attributes : UserAttributesDict | UserAttributes
            Attributes to update, could be: email, password, email_change_token, data

        Returns
        -------
        response : User
            The updated user data.

        Raises
        ------
        APIError
            If an error occurs.
        """
        if not self.current_session:
            raise ValueError("Not logged in.")

        if isinstance(attributes, dict):
            attributes_to_update = UserAttributes(**attributes)
        else:
            attributes_to_update = attributes

        response = self.api.update_user(
            jwt=self.current_session.access_token,
            attributes=attributes_to_update,
        )
        self.current_session.user = response
        self._save_session(session=self.current_session)
        self._notify_all_subscribers(event=AuthChangeEvent.USER_UPDATED)
        return response

    def set_session(self, *, refresh_token: str) -> Session:
        """Sets the session data from refresh_token and returns current Session

        Parameters
        ----------
        refresh_token : str
            A JWT token

        Returns
        -------
        response : Session
            A logged-in session

        Raises
        ------
        APIError
            If an error occurs.
        """
        response = self.api.refresh_access_token(refresh_token=refresh_token)
        self._save_session(session=response)
        self._notify_all_subscribers(event=AuthChangeEvent.SIGNED_IN)
        return response

    def set_auth(self, *, access_token: str) -> Session:
        """Overrides the JWT on the current client. The JWT will then be sent in
        all subsequent network requests.

        Parameters
        ----------
        access_token : str
            A JWT token

        Returns
        -------
        response : Session
            A logged-in session

        Raises
        ------
        APIError
            If an error occurs.
        """
        session = Session(
            access_token=access_token,
            token_type="bearer",
            user=None,
            expires_in=None,
            expires_at=None,
            refresh_token=None,
            provider_token=None,
        )
        if self.current_session:
            session.expires_in = self.current_session.expires_in
            session.expires_at = self.current_session.expires_at
            session.refresh_token = self.current_session.refresh_token
            session.provider_token = self.current_session.provider_token
        self._save_session(session=session)
        return session

    def get_session_from_url(
        self,
        *,
        url: str,
        store_session: bool = False,
    ) -> Session:
        """Gets the session data from a URL string.

        Parameters
        ----------
        url : str
            The URL string.
        store_session : bool
            Optionally store the session in the browser

        Returns
        -------
        response : Session
            A logged-in session

        Raises
        ------
        APIError
            If an error occurs.
        """
        data = urlparse(url)
        query = parse_qs(data.query)
        error_description = query.get("error_description")
        access_token = query.get("access_token")
        expires_in = query.get("expires_in")
        refresh_token = query.get("refresh_token")
        token_type = query.get("token_type")
        if error_description:
            raise APIError(error_description[0], 400)
        if not access_token or not access_token[0]:
            raise APIError("No access_token detected.", 400)
        if not refresh_token or not refresh_token[0]:
            raise APIError("No refresh_token detected.", 400)
        if not token_type or not token_type[0]:
            raise APIError("No token_type detected.", 400)
        if not expires_in or not expires_in[0]:
            raise APIError("No expires_in detected.", 400)
        try:
            expires_at = round(time()) + int(expires_in[0])
        except ValueError:
            raise APIError("Invalid expires_in.", 400)
        response = self.api.get_user(jwt=access_token[0])
        provider_token = query.get("provider_token")
        session = Session(
            access_token=access_token[0],
            token_type=token_type[0],
            user=response,
            expires_in=int(expires_in[0]),
            expires_at=expires_at,
            refresh_token=refresh_token[0],
            provider_token=provider_token[0] if provider_token else None,
        )
        if store_session:
            self._save_session(session=session)
            recovery_mode = query.get("type")
            self._notify_all_subscribers(event=AuthChangeEvent.SIGNED_IN)
            if recovery_mode and recovery_mode[0] == "recovery":
                self._notify_all_subscribers(event=AuthChangeEvent.PASSWORD_RECOVERY)
        return session

    def sign_out(self) -> None:
        """Log the user out."""
        access_token: Optional[str] = None
        if self.current_session:
            access_token = self.current_session.access_token
        self._remove_session()
        self._notify_all_subscribers(event=AuthChangeEvent.SIGNED_OUT)
        if access_token:
            self.api.sign_out(jwt=access_token)

    def _unsubscribe(self, *, id: str) -> None:
        """Unsubscribe from a subscription."""
        self.state_change_emitters.pop(id)

    def on_auth_state_change(
        self,
        *,
        callback: Callable[[AuthChangeEvent, Optional[Session]], None],
    ) -> Subscription:
        """Receive a notification every time an auth event happens.

        Parameters
        ----------
        callback : Callable[[AuthChangeEvent, Optional[Session]], None]
            The callback to call when an auth event happens.

        Returns
        -------
        subscription : Subscription
            A subscription object which can be used to unsubscribe itself.

        Raises
        ------
        APIError
            If an error occurs.
        """
        unique_id = uuid4()
        subscription = Subscription(
            id=unique_id,
            callback=callback,
            unsubscribe=partial(self._unsubscribe, id=unique_id.hex),
        )
        self.state_change_emitters[unique_id.hex] = subscription
        return subscription

    def _handle_email_sign_in(
        self,
        *,
        email: str,
        password: str,
        redirect_to: Optional[str],
    ) -> Session:
        """Sign in with email and password."""
        response = self.api.sign_in_with_email(
            email=email,
            password=password,
            redirect_to=redirect_to,
        )
        self._save_session(session=response)
        self._notify_all_subscribers(event=AuthChangeEvent.SIGNED_IN)
        return response

    def _handle_phone_sign_in(self, *, phone: str, password: str) -> Session:
        """Sign in with phone and password."""
        response = self.api.sign_in_with_phone(phone=phone, password=password)
        self._save_session(session=response)
        self._notify_all_subscribers(event=AuthChangeEvent.SIGNED_IN)
        return response

    def _handle_provider_sign_in(
        self,
        *,
        provider: Provider,
        redirect_to: Optional[str],
        scopes: Optional[str],
    ) -> str:
        """Sign in with provider."""
        return self.api.get_url_for_provider(
            provider=provider,
            redirect_to=redirect_to,
            scopes=scopes,
        )

    def _recover_common(self) -> Optional[Tuple[Session, int, int]]:
        """Recover common logic"""
        json = self.local_storage.get_item(STORAGE_KEY)
        if not json:
            return
        data = loads(json)
        session_raw = data.get("session")
        expires_at_raw = data.get("expires_at")
        if (
            expires_at_raw
            and isinstance(expires_at_raw, int)
            and session_raw
            and isinstance(session_raw, dict)
        ):
            session = model_validate(Session, session_raw)
            expires_at = int(expires_at_raw)
            time_now = round(time())
            return session, expires_at, time_now

    def _recover_session(self) -> None:
        """Attempts to get the session from LocalStorage"""
        result = self._recover_common()
        if not result:
            return
        session, expires_at, time_now = result
        if expires_at >= time_now:
            self._save_session(session=session)
            self._notify_all_subscribers(event=AuthChangeEvent.SIGNED_IN)

    def _recover_and_refresh(self) -> None:
        """Recovers the session from LocalStorage and refreshes"""
        result = self._recover_common()
        if not result:
            return
        session, expires_at, time_now = result
        if expires_at < time_now and self.auto_refresh_token and session.refresh_token:
            try:
                self._call_refresh_token(refresh_token=session.refresh_token)
            except APIError:
                self._remove_session()
        elif expires_at < time_now or not session or not session.user:
            self._remove_session()
        else:
            self._save_session(session=session)
            self._notify_all_subscribers(event=AuthChangeEvent.SIGNED_IN)

    def _call_refresh_token(self, *, refresh_token: Optional[str] = None) -> Session:
        if refresh_token is None:
            if self.current_session:
                refresh_token = self.current_session.refresh_token
            else:
                raise ValueError("No current session and refresh_token not supplied.")
        response = self.api.refresh_access_token(refresh_token=cast(str, refresh_token))
        self._save_session(session=response)
        self._notify_all_subscribers(event=AuthChangeEvent.TOKEN_REFRESHED)
        self._notify_all_subscribers(event=AuthChangeEvent.SIGNED_IN)
        return response

    def _notify_all_subscribers(self, *, event: AuthChangeEvent) -> None:
        """Notify all subscribers that auth event happened."""
        for value in self.state_change_emitters.values():
            value.callback(event, self.current_session)

    def _save_session(self, *, session: Session) -> None:
        """Save session to client."""
        self.current_session = session
        self.current_user = session.user
        if session.expires_at:
            time_now = round(time())
            expire_in = session.expires_at - time_now
            refresh_duration_before_expires = 60 if expire_in > 60 else 0.5
            self._start_auto_refresh_token(
                value=(expire_in - refresh_duration_before_expires)
            )
        if self.persist_session and session.expires_at:
            self._persist_session(session=session)

    def _persist_session(self, *, session: Session) -> None:
        data = {"session": model_dump(session), "expires_at": session.expires_at}
        self.local_storage.set_item(STORAGE_KEY, dumps(data, default=str))

    def _remove_session(self) -> None:
        """Remove the session."""
        self.current_session = None
        self.current_user = None
        if self.refresh_token_timer:
            self.refresh_token_timer.cancel()
        self.local_storage.remove_item(STORAGE_KEY)

    def _start_auto_refresh_token(self, *, value: float) -> None:
        if self.refresh_token_timer:
            self.refresh_token_timer.cancel()
        if value <= 0 or not self.auto_refresh_token:
            return
        self.refresh_token_timer = Timer(value, self._call_refresh_token)
        self.refresh_token_timer.start()
