from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import parse_obj_as

from ..exceptions import APIError
from ..helpers import check_response, encode_uri_component
from ..http_clients import SyncClient
from ..types import (
    CookieOptions,
    LinkType,
    Provider,
    Session,
    User,
    UserAttributes,
    determine_session_or_user_model_from_response,
)


class SyncGoTrueAPI:
    def __init__(
        self,
        *,
        url: str,
        headers: Dict[str, str],
        cookie_options: CookieOptions,
        http_client: Optional[SyncClient] = None,
    ) -> None:
        """Initialise API class."""
        self.url = url
        self.headers = headers
        self.cookie_options = cookie_options
        self.http_client = http_client or SyncClient()

    def __enter__(self) -> SyncGoTrueAPI:
        return self

    def __exit__(self, exc_t, exc_v, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        self.http_client.aclose()

    def create_user(self, *, attributes: UserAttributes) -> User:
        """Creates a new user.

        This function should only be called on a server.
        Never expose your `service_role` key in the browser.

        Parameters
        ----------
        attributes: UserAttributes
            The data you want to create the user with.

        Returns
        -------
        response : User
            The created user

        Raises
        ------
        APIError
            If an error occurs.
        """
        headers = self.headers
        data = attributes.dict()
        url = f"{self.url}/admin/users"
        response = self.http_client.post(url, json=data, headers=headers)
        return User.parse_response(response)

    def list_users(self) -> List[User]:
        """Get a list of users.

        This function should only be called on a server.
        Never expose your `service_role` key in the browser.

        Returns
        -------
        response : List[User]
            A list of users

        Raises
        ------
        APIError
            If an error occurs.
        """
        headers = self.headers
        url = f"{self.url}/admin/users"
        response = self.http_client.get(url, headers=headers)
        check_response(response)
        users = response.json().get("users")
        if users is None:
            raise APIError("No users found in response", 400)
        if not isinstance(users, list):
            raise APIError("Expected a list of users", 400)
        return parse_obj_as(List[User], users)

    def sign_up_with_email(
        self,
        *,
        email: str,
        password: str,
        redirect_to: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Union[Session, User]:
        """Creates a new user using their email address.

        Parameters
        ----------
        email : str
            The email address of the user.
        password : str
            The password of the user.
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
        headers = self.headers
        query_string = ""
        if redirect_to:
            redirect_to_encoded = encode_uri_component(redirect_to)
            query_string = f"?redirect_to={redirect_to_encoded}"
        data = {"email": email, "password": password, "data": data}
        url = f"{self.url}/signup{query_string}"
        response = self.http_client.post(url, json=data, headers=headers)
        SessionOrUserModel = determine_session_or_user_model_from_response(response)
        return SessionOrUserModel.parse_response(response)

    def sign_in_with_email(
        self,
        *,
        email: str,
        password: str,
        redirect_to: Optional[str] = None,
    ) -> Session:
        """Logs in an existing user using their email address.

        Parameters
        ----------
        email : str
            The email address of the user.
        password : str
            The password of the user.
        redirect_to : Optional[str]
            A URL or mobile address to send the user to after they are confirmed.

        Returns
        -------
        response : Session
            A logged-in session

        Raises
        ------
        APIError
            If an error occurs.
        """

        headers = self.headers
        query_string = "?grant_type=password"
        if redirect_to:
            redirect_to_encoded = encode_uri_component(redirect_to)
            query_string += f"&redirect_to={redirect_to_encoded}"
        data = {"email": email, "password": password}
        url = f"{self.url}/token{query_string}"
        response = self.http_client.post(url, json=data, headers=headers)
        return Session.parse_response(response)

    def sign_up_with_phone(
        self,
        *,
        phone: str,
        password: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> Union[Session, User]:
        """Signs up a new user using their phone number and a password.

        Parameters
        ----------
        phone : str
            The phone number of the user.
        password : str
            The password of the user.
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
        headers = self.headers
        data = {"phone": phone, "password": password, "data": data}
        url = f"{self.url}/signup"
        response = self.http_client.post(url, json=data, headers=headers)
        SessionOrUserModel = determine_session_or_user_model_from_response(response)
        return SessionOrUserModel.parse_response(response)

    def sign_in_with_phone(
        self,
        *,
        phone: str,
        password: str,
    ) -> Session:
        """Logs in an existing user using their phone number and password.

        Parameters
        ----------
        phone : str
            The phone number of the user.
        password : str
            The password of the user.

        Returns
        -------
        response : Session
            A logged-in session

        Raises
        ------
        APIError
            If an error occurs.
        """
        data = {"phone": phone, "password": password}
        url = f"{self.url}/token?grant_type=password"
        headers = self.headers
        response = self.http_client.post(url, json=data, headers=headers)
        return Session.parse_response(response)

    def send_magic_link_email(
        self,
        *,
        email: str,
        create_user: bool,
        redirect_to: Optional[str] = None,
    ) -> None:
        """Sends a magic login link to an email address.

        Parameters
        ----------
        email : str
            The email address of the user.
        redirect_to : Optional[str]
            A URL or mobile address to send the user to after they are confirmed.

        Raises
        ------
        APIError
            If an error occurs.
        """
        headers = self.headers
        query_string = ""
        if redirect_to:
            redirect_to_encoded = encode_uri_component(redirect_to)
            query_string = f"?redirect_to={redirect_to_encoded}"
        data = {"email": email, "create_user": create_user}
        url = f"{self.url}/magiclink{query_string}"
        response = self.http_client.post(url, json=data, headers=headers)
        return check_response(response)

    def send_mobile_otp(self, *, phone: str, create_user: bool) -> None:
        """Sends a mobile OTP via SMS. Will register the account if it doesn't already exist

        Parameters
        ----------
        phone : str
            The user's phone number WITH international prefix

        Raises
        ------
        APIError
            If an error occurs.
        """
        headers = self.headers
        data = {"phone": phone, "create_user": create_user}
        url = f"{self.url}/otp"
        response = self.http_client.post(url, json=data, headers=headers)
        return check_response(response)

    def verify_mobile_otp(
        self,
        *,
        phone: str,
        token: str,
        redirect_to: Optional[str] = None,
    ) -> Union[Session, User]:
        """Send User supplied Mobile OTP to be verified

        Parameters
        ----------
        phone : str
            The user's phone number WITH international prefix
        token : str
            Token that user was sent to their mobile phone
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
        headers = self.headers
        data = {
            "phone": phone,
            "token": token,
            "type": "sms",
        }
        if redirect_to:
            redirect_to_encoded = encode_uri_component(redirect_to)
            data["redirect_to"] = redirect_to_encoded
        url = f"{self.url}/verify"
        response = self.http_client.post(url, json=data, headers=headers)
        SessionOrUserModel = determine_session_or_user_model_from_response(response)
        return SessionOrUserModel.parse_response(response)

    def invite_user_by_email(
        self,
        *,
        email: str,
        redirect_to: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> User:
        """Sends an invite link to an email address.

        Parameters
        ----------
        email : str
            The email address of the user.
        redirect_to : Optional[str]
            A URL or mobile address to send the user to after they are confirmed.
        data : Optional[Dict[str, Any]]
            Optional user metadata.

        Returns
        -------
        response : User
            A user

        Raises
        ------
        APIError
            If an error occurs.
        """
        headers = self.headers
        query_string = ""
        if redirect_to:
            redirect_to_encoded = encode_uri_component(redirect_to)
            query_string = f"?redirect_to={redirect_to_encoded}"
        data = {"email": email, "data": data}
        url = f"{self.url}/invite{query_string}"
        response = self.http_client.post(url, json=data, headers=headers)
        return User.parse_response(response)

    def reset_password_for_email(
        self,
        *,
        email: str,
        redirect_to: Optional[str] = None,
    ) -> None:
        """Sends a reset request to an email address.

        Parameters
        ----------
        email : str
            The email address of the user.
        redirect_to : Optional[str]
            A URL or mobile address to send the user to after they are confirmed.

        Raises
        ------
        APIError
            If an error occurs.
        """
        headers = self.headers
        query_string = ""
        if redirect_to:
            redirect_to_encoded = encode_uri_component(redirect_to)
            query_string = f"?redirect_to={redirect_to_encoded}"
        data = {"email": email}
        url = f"{self.url}/recover{query_string}"
        response = self.http_client.post(url, json=data, headers=headers)
        return check_response(response)

    def _create_request_headers(self, *, jwt: str) -> Dict[str, str]:
        """Create temporary object.

        Create a temporary object with all configured headers and adds the
        Authorization token to be used on request methods.

        Parameters
        ----------
        jwt : str
            A valid, logged-in JWT.

        Returns
        -------
        headers : dict of str
            The headers required for a successful request statement with the
            supabase backend.
        """
        headers = {**self.headers, "Authorization": f"Bearer {jwt}"}
        return headers

    def sign_out(self, *, jwt: str) -> None:
        """Removes a logged-in session.

        Parameters
        ----------
        jwt : str
            A valid, logged-in JWT.
        """
        headers = self._create_request_headers(jwt=jwt)
        url = f"{self.url}/logout"
        self.http_client.post(url, headers=headers)

    def get_url_for_provider(
        self,
        *,
        provider: Provider,
        redirect_to: Optional[str] = None,
        scopes: Optional[str] = None,
    ) -> str:
        """Generates the relevant login URL for a third-party provider.

        Parameters
        ----------
        provider : Provider
            One of the providers supported by GoTrue.
        redirect_to : Optional[str]
            A URL or mobile address to send the user to after they are confirmed.
        scopes : Optional[str]
            A space-separated list of scopes granted to the OAuth application.

        Returns
        -------
        url : str
            The URL to redirect the user to.

        Raises
        ------
        APIError
            If an error occurs.
        """
        url_params = [f"provider={encode_uri_component(provider)}"]
        if redirect_to:
            redirect_to_encoded = encode_uri_component(redirect_to)
            url_params.append(f"redirect_to={redirect_to_encoded}")
        if scopes:
            url_params.append(f"scopes={encode_uri_component(scopes)}")
        return f"{self.url}/authorize?{'&'.join(url_params)}"

    def get_user(self, *, jwt: str) -> User:
        """Gets the user details.

        Parameters
        ----------
        jwt : str
            A valid, logged-in JWT.

        Returns
        -------
        response : User
            A user

        Raises
        ------
        APIError
            If an error occurs.
        """
        headers = self._create_request_headers(jwt=jwt)
        url = f"{self.url}/user"
        response = self.http_client.get(url, headers=headers)
        return User.parse_response(response)

    def update_user(
        self,
        *,
        jwt: str,
        attributes: UserAttributes,
    ) -> User:
        """
        Updates the user data.

        Parameters
        ----------
        jwt : str
            A valid, logged-in JWT.
        attributes : UserAttributes
            The data you want to update.

        Returns
        -------
        response : User
            A user

        Raises
        ------
        APIError
            If an error occurs.
        """
        headers = self._create_request_headers(jwt=jwt)
        data = attributes.dict()
        url = f"{self.url}/user"
        response = self.http_client.put(url, json=data, headers=headers)
        return User.parse_response(response)

    def delete_user(
        self, *, uid: str, jwt: str, should_soft_delete: bool = False
    ) -> None:
        """Delete a user. Requires a `service_role` key.

        This function should only be called on a server.
        Never expose your `service_role` key in the browser.

        Parameters
        ----------
        uid : str
            The user uid you want to remove.
        jwt : str
            A valid, logged-in JWT.
        should_soft_delete : bool
            If true, then the user will be soft-deleted from the auth schema.

        Returns
        -------
        response : User
            A user

        Raises
        ------
        APIError
            If an error occurs.
        """
        headers = self._create_request_headers(jwt=jwt)
        url = f"{self.url}/admin/users/{uid}"
        body = {
            "should_soft_delete": should_soft_delete,
        }
        response = self.http_client.delete(url, json=body, headers=headers)
        return check_response(response)

    def refresh_access_token(self, *, refresh_token: str) -> Session:
        """Generates a new JWT.

        Parameters
        ----------
        refresh_token : str
            A valid refresh token that was returned on login.

        Returns
        -------
        response : Session
            A session

        Raises
        ------
        APIError
            If an error occurs.
        """
        data = {"refresh_token": refresh_token}
        url = f"{self.url}/token?grant_type=refresh_token"
        headers = self.headers
        response = self.http_client.post(url, json=data, headers=headers)
        return Session.parse_response(response)

    def generate_link(
        self,
        *,
        type: LinkType,
        email: str,
        password: Optional[str] = None,
        redirect_to: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Union[Session, User]:
        """
        Generates links to be sent via email or other.

        Parameters
        ----------
        type : LinkType
            The link type ("signup" or "magiclink" or "recovery" or "invite").
        email : str
            The user's email.
        password : Optional[str]
            User password. For signup only.
        redirect_to : Optional[str]
            The link type ("signup" or "magiclink" or "recovery" or "invite").
        data : Optional[Dict[str, Any]]
            Optional user metadata. For signup only.

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
        headers = self.headers
        data = {
            "type": type,
            "email": email,
            "data": data,
        }
        if password:
            data["password"] = password
        if redirect_to:
            redirect_to_encoded = encode_uri_component(redirect_to)
            data["redirect_to"] = redirect_to_encoded
        url = f"{self.url}/admin/generate_link"
        response = self.http_client.post(url, json=data, headers=headers)
        SessionOrUserModel = determine_session_or_user_model_from_response(response)
        return SessionOrUserModel.parse_response(response)

    def set_auth_cookie(self, *, req, res):
        """Stub for parity with JS api."""
        raise NotImplementedError("set_auth_cookie not implemented.")

    def get_user_by_cookie(self, *, req):
        """Stub for parity with JS api."""
        raise NotImplementedError("get_user_by_cookie not implemented.")
