from ..types import (
    AuthMFAChallengeResponse,
    AuthMFAEnrollResponse,
    AuthMFAGetAuthenticatorAssuranceLevelResponse,
    AuthMFAListFactorsResponse,
    AuthMFAUnenrollResponse,
    AuthMFAVerifyResponse,
    MFAChallengeAndVerifyParams,
    MFAChallengeParams,
    MFAEnrollParams,
    MFAUnenrollParams,
    MFAVerifyParams,
)


class AsyncGoTrueMFAAPI:
    """
    Contains the full multi-factor authentication API.
    """

    async def enroll(self, params: MFAEnrollParams) -> AuthMFAEnrollResponse:
        """
        Starts the enrollment process for a new Multi-Factor Authentication
        factor. This method creates a new factor in the 'unverified' state.
        Present the QR code or secret to the user and ask them to add it to their
        authenticator app. Ask the user to provide you with an authenticator code
        from their app and verify it by calling challenge and then verify.

        The first successful verification of an unverified factor activates the
        factor. All other sessions are logged out and the current one gets an
        `aal2` authenticator level.
        """
        raise NotImplementedError()  # pragma: no cover

    async def challenge(self, params: MFAChallengeParams) -> AuthMFAChallengeResponse:
        """
        Prepares a challenge used to verify that a user has access to a MFA
        factor. Provide the challenge ID and verification code by calling `verify`.
        """
        raise NotImplementedError()  # pragma: no cover

    async def challenge_and_verify(
        self,
        params: MFAChallengeAndVerifyParams,
    ) -> AuthMFAVerifyResponse:
        """
        Helper method which creates a challenge and immediately uses the given code
        to verify against it thereafter. The verification code is provided by the
        user by entering a code seen in their authenticator app.
        """
        raise NotImplementedError()  # pragma: no cover

    async def verify(self, params: MFAVerifyParams) -> AuthMFAVerifyResponse:
        """
        Verifies a verification code against a challenge. The verification code is
        provided by the user by entering a code seen in their authenticator app.
        """
        raise NotImplementedError()  # pragma: no cover

    async def unenroll(self, params: MFAUnenrollParams) -> AuthMFAUnenrollResponse:
        """
        Unenroll removes a MFA factor. Unverified factors can safely be ignored
        and it's not necessary to unenroll them. Unenrolling a verified MFA factor
        cannot be done from a session with an `aal1` authenticator level.
        """
        raise NotImplementedError()  # pragma: no cover

    async def list_factors(self) -> AuthMFAListFactorsResponse:
        """
        Returns the list of MFA factors enabled for this user. For most use cases
        you should consider using `get_authenticator_assurance_level`.

        This uses a cached version of the factors and avoids incurring a network call.
        If you need to update this list, call `get_user` first.
        """
        raise NotImplementedError()  # pragma: no cover

    async def get_authenticator_assurance_level(
        self,
    ) -> AuthMFAGetAuthenticatorAssuranceLevelResponse:
        """
        Returns the Authenticator Assurance Level (AAL) for the active session.

        - `aal1` (or `null`) means that the user's identity has been verified only
        with a conventional login (email+password, OTP, magic link, social login,
        etc.).
        - `aal2` means that the user's identity has been verified both with a
        conventional login and at least one MFA factor.

        Although this method returns a promise, it's fairly quick (microseconds)
        and rarely uses the network. You can use this to check whether the current
        user needs to be shown a screen to verify their MFA factors.
        """
        raise NotImplementedError()  # pragma: no cover
