from __future__ import annotations

from functools import partial
from typing import Dict, List, Union

from ..helpers import model_validate, parse_link_response, parse_user_response
from ..http_clients import SyncClient
from ..types import (
    AdminUserAttributes,
    AuthMFAAdminDeleteFactorParams,
    AuthMFAAdminDeleteFactorResponse,
    AuthMFAAdminListFactorsParams,
    AuthMFAAdminListFactorsResponse,
    GenerateLinkParams,
    GenerateLinkResponse,
    Options,
    SignOutScope,
    User,
    UserResponse,
)
from .gotrue_admin_mfa_api import SyncGoTrueAdminMFAAPI
from .gotrue_base_api import SyncGoTrueBaseAPI


class SyncGoTrueAdminAPI(SyncGoTrueBaseAPI):
    def __init__(
        self,
        *,
        url: str = "",
        headers: Dict[str, str] = {},
        http_client: Union[SyncClient, None] = None,
    ) -> None:
        SyncGoTrueBaseAPI.__init__(
            self,
            url=url,
            headers=headers,
            http_client=http_client,
        )
        self.mfa = SyncGoTrueAdminMFAAPI()
        self.mfa.list_factors = self._list_factors
        self.mfa.delete_factor = self._delete_factor

    def sign_out(self, jwt: str, scope: SignOutScope = "global") -> None:
        """
        Removes a logged-in session.
        """
        return self._request(
            "POST",
            "logout",
            query={"scope": scope},
            jwt=jwt,
            no_resolve_json=True,
        )

    def invite_user_by_email(
        self,
        email: str,
        options: Options = {},
    ) -> UserResponse:
        """
        Sends an invite link to an email address.
        """
        return self._request(
            "POST",
            "invite",
            body={"email": email, "data": options.get("data")},
            redirect_to=options.get("redirect_to"),
            xform=parse_user_response,
        )

    def generate_link(self, params: GenerateLinkParams) -> GenerateLinkResponse:
        """
        Generates email links and OTPs to be sent via a custom email provider.
        """
        return self._request(
            "POST",
            "admin/generate_link",
            body={
                "type": params.get("type"),
                "email": params.get("email"),
                "password": params.get("password"),
                "new_email": params.get("new_email"),
                "data": params.get("options", {}).get("data"),
            },
            redirect_to=params.get("options", {}).get("redirect_to"),
            xform=parse_link_response,
        )

    # User Admin API

    def create_user(self, attributes: AdminUserAttributes) -> UserResponse:
        """
        Creates a new user.

        This function should only be called on a server.
        Never expose your `service_role` key in the browser.
        """
        return self._request(
            "POST",
            "admin/users",
            body=attributes,
            xform=parse_user_response,
        )

    def list_users(self, page: int = None, per_page: int = None) -> List[User]:
        """
        Get a list of users.

        This function should only be called on a server.
        Never expose your `service_role` key in the browser.
        """
        return self._request(
            "GET",
            "admin/users",
            query={"page": page, "per_page": per_page},
            xform=lambda data: [model_validate(User, user) for user in data["users"]]
            if "users" in data
            else [],
        )

    def get_user_by_id(self, uid: str) -> UserResponse:
        """
        Get user by id.

        This function should only be called on a server.
        Never expose your `service_role` key in the browser.
        """
        return self._request(
            "GET",
            f"admin/users/{uid}",
            xform=parse_user_response,
        )

    def update_user_by_id(
        self,
        uid: str,
        attributes: AdminUserAttributes,
    ) -> UserResponse:
        """
        Updates the user data.

        This function should only be called on a server.
        Never expose your `service_role` key in the browser.
        """
        return self._request(
            "PUT",
            f"admin/users/{uid}",
            body=attributes,
            xform=parse_user_response,
        )

    def delete_user(self, id: str, should_soft_delete: bool = False) -> None:
        """
        Delete a user. Requires a `service_role` key.

        This function should only be called on a server.
        Never expose your `service_role` key in the browser.
        """
        body = {"should_soft_delete": should_soft_delete}
        return self._request("DELETE", f"admin/users/{id}", body=body)

    def _list_factors(
        self,
        params: AuthMFAAdminListFactorsParams,
    ) -> AuthMFAAdminListFactorsResponse:
        return self._request(
            "GET",
            f"admin/users/{params.get('user_id')}/factors",
            xform=partial(model_validate, AuthMFAAdminListFactorsResponse),
        )

    def _delete_factor(
        self,
        params: AuthMFAAdminDeleteFactorParams,
    ) -> AuthMFAAdminDeleteFactorResponse:
        return self._request(
            "DELETE",
            f"admin/users/{params.get('user_id')}/factors/{params.get('factor_id')}",
            xform=partial(model_validate, AuthMFAAdminDeleteFactorResponse),
        )
