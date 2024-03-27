from ..types import (
    AuthMFAAdminDeleteFactorParams,
    AuthMFAAdminDeleteFactorResponse,
    AuthMFAAdminListFactorsParams,
    AuthMFAAdminListFactorsResponse,
)


class SyncGoTrueAdminMFAAPI:
    """
    Contains the full multi-factor authentication administration API.
    """

    def list_factors(
        self,
        params: AuthMFAAdminListFactorsParams,
    ) -> AuthMFAAdminListFactorsResponse:
        """
        Lists all factors attached to a user.
        """
        raise NotImplementedError()  # pragma: no cover

    def delete_factor(
        self,
        params: AuthMFAAdminDeleteFactorParams,
    ) -> AuthMFAAdminDeleteFactorResponse:
        """
        Deletes a factor on a user. This will log the user out of all active
        sessions (if the deleted factor was verified). There's no need to delete
        unverified factors.
        """
        raise NotImplementedError()  # pragma: no cover
