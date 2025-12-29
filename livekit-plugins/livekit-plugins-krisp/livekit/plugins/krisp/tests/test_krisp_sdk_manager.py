# Copyright 2023 LiveKit, Inc.
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

"""Unit tests for Krisp SDK Manager (singleton with reference counting)."""

import pytest

from livekit.plugins.krisp.krisp_instance import (
    KRISP_SAMPLE_RATES,
    KrispSDKManager,
    int_to_krisp_sample_rate,
)


class TestKrispSDKManager:
    """Tests for KrispSDKManager singleton."""

    def test_reference_counting(self):
        """Test that SDK manager properly tracks references."""
        # Initial state
        initial_count = KrispSDKManager.get_reference_count()
        _ = KrispSDKManager.is_initialized()  # Check but don't use

        # Acquire first reference
        KrispSDKManager.acquire()
        assert KrispSDKManager.get_reference_count() == initial_count + 1
        assert KrispSDKManager.is_initialized()

        # Acquire second reference
        KrispSDKManager.acquire()
        assert KrispSDKManager.get_reference_count() == initial_count + 2
        assert KrispSDKManager.is_initialized()

        # Release first reference
        KrispSDKManager.release()
        assert KrispSDKManager.get_reference_count() == initial_count + 1
        assert KrispSDKManager.is_initialized()

        # Release second reference
        KrispSDKManager.release()
        assert KrispSDKManager.get_reference_count() == initial_count
        # Note: SDK might still be initialized from other tests or previous calls

    def test_multiple_acquire_release_cycles(self):
        """Test multiple acquire/release cycles."""
        initial_count = KrispSDKManager.get_reference_count()

        for _ in range(3):
            KrispSDKManager.acquire()
            assert KrispSDKManager.get_reference_count() > initial_count
            assert KrispSDKManager.is_initialized()
            KrispSDKManager.release()
            assert KrispSDKManager.get_reference_count() == initial_count


class TestSampleRateConversion:
    """Tests for sample rate conversion utilities."""

    def test_supported_sample_rates(self):
        """Test conversion of all supported sample rates."""
        for rate_hz, krisp_enum in KRISP_SAMPLE_RATES.items():
            result = int_to_krisp_sample_rate(rate_hz)
            assert result == krisp_enum

    def test_unsupported_sample_rate(self):
        """Test that unsupported rates raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported sample rate"):
            int_to_krisp_sample_rate(22050)  # Not supported

        with pytest.raises(ValueError, match="Unsupported sample rate"):
            int_to_krisp_sample_rate(96000)  # Not supported

    def test_sample_rate_error_message(self):
        """Test that error message includes helpful information."""
        try:
            int_to_krisp_sample_rate(11025)
        except ValueError as e:
            assert "11025" in str(e)
            assert "Supported rates" in str(e)
            # Should list at least some supported rates
            assert "16000" in str(e)
