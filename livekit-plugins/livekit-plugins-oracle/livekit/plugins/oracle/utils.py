# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Oracle Corporation and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module provides utilities used throughout the Oracle LiveKit plug-in code.

Author: Keith Schnable (at Oracle Corporation)
Date: 2025-08-12
"""

from __future__ import annotations

from enum import Enum

import oci


class AuthenticationType(Enum):
    """Authentication types as enumerator."""

    API_KEY = "API_KEY"
    SECURITY_TOKEN = "SECURITY_TOKEN"
    INSTANCE_PRINCIPAL = "INSTANCE_PRINCIPAL"
    RESOURCE_PRINCIPAL = "RESOURCE_PRINCIPAL"


def get_config_and_signer(
    *,
    authentication_type: AuthenticationType = None,
    authentication_configuration_file_spec: str = None,
    authentication_profile_name: str = None,
):
    config = {}
    signer = None

    # API_KEY
    if authentication_type == AuthenticationType.API_KEY:
        config = oci.config.from_file(
            authentication_configuration_file_spec, authentication_profile_name
        )

    # SECURITY_TOKEN
    elif authentication_type == AuthenticationType.SECURITY_TOKEN:
        config = oci.config.from_file(
            authentication_configuration_file_spec, authentication_profile_name
        )
        with open(config["security_token_file"]) as f:
            token = f.readline()
        private_key = oci.signer.load_private_key_from_file(config["key_file"])
        signer = oci.auth.signers.SecurityTokenSigner(token=token, private_key=private_key)

    # INSTANCE_PRINCIPAL
    elif authentication_type == AuthenticationType.INSTANCE_PRINCIPAL:
        signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()

    # RESOURCE_PRINCIPAL
    elif authentication_type == AuthenticationType.RESOURCE_PRINCIPAL:
        signer = oci.auth.signers.get_resource_principals_signer()

    return {"config": config, "signer": signer}
