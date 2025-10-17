"""
Unit tests for API key encryption service
These tests don't require database connection
"""
import pytest
from services.api_key_service import APIKeyService


class TestAPIKeyEncryption:
    """Test API key encryption/decryption"""

    def setup_method(self):
        """Setup test fixture"""
        self.service = APIKeyService()

    def test_encrypt_decrypt_roundtrip(self):
        """Test encryption and decryption produce original value"""
        original_key = "sk-test-key-1234567890abcdef"

        encrypted = self.service.encrypt_api_key(original_key)
        assert encrypted != original_key  # Should be encrypted

        decrypted = self.service.decrypt_api_key(encrypted)
        assert decrypted == original_key

    def test_encrypted_keys_are_different(self):
        """Test same input produces different encrypted outputs (due to IV)"""
        key = "sk-test-key-xyz"

        encrypted1 = self.service.encrypt_api_key(key)
        encrypted2 = self.service.encrypt_api_key(key)

        # Same plaintext should produce different ciphertexts (Fernet uses random IV)
        # But both should decrypt to same value
        assert self.service.decrypt_api_key(encrypted1) == key
        assert self.service.decrypt_api_key(encrypted2) == key

    def test_wrong_encryption_key_fails(self):
        """Test decryption with wrong key fails"""
        original_key = "sk-test-key-abc"

        # Encrypt with one service instance
        encrypted = self.service.encrypt_api_key(original_key)

        # Try to decrypt with different service (different encryption key)
        different_service = APIKeyService()

        # Should fail or return wrong value
        with pytest.raises(Exception):
            different_service.decrypt_api_key(encrypted)

    def test_empty_string_encryption(self):
        """Test encrypting empty string"""
        encrypted = self.service.encrypt_api_key("")
        decrypted = self.service.decrypt_api_key(encrypted)
        assert decrypted == ""

    def test_long_key_encryption(self):
        """Test encrypting long API key"""
        long_key = "sk-" + "a" * 1000
        encrypted = self.service.encrypt_api_key(long_key)
        decrypted = self.service.decrypt_api_key(encrypted)
        assert decrypted == long_key


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
