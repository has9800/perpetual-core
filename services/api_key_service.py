"""
User API Key Management Service
Securely stores and retrieves user API keys for external providers
"""
from cryptography.fernet import Fernet
from typing import Optional, Dict
from supabase import Client
import os
import base64


class APIKeyService:
    """Service for managing user API keys with encryption"""

    def __init__(self):
        """Initialize encryption"""
        # Get encryption key from environment
        # Generate with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
        encryption_key = os.getenv("API_KEY_ENCRYPTION_KEY")

        if not encryption_key:
            # For development - generate temporary key
            # IMPORTANT: In production, this MUST be set in environment!
            print("⚠️  WARNING: API_KEY_ENCRYPTION_KEY not set! Using temporary key (dev only)")
            encryption_key = Fernet.generate_key().decode()

        self.cipher = Fernet(encryption_key.encode())

    def encrypt_api_key(self, api_key: str) -> str:
        """
        Encrypt an API key

        Args:
            api_key: Plain text API key

        Returns:
            Encrypted API key (base64 encoded)
        """
        encrypted = self.cipher.encrypt(api_key.encode())
        return encrypted.decode()

    def decrypt_api_key(self, encrypted_key: str) -> str:
        """
        Decrypt an API key

        Args:
            encrypted_key: Encrypted API key

        Returns:
            Plain text API key
        """
        decrypted = self.cipher.decrypt(encrypted_key.encode())
        return decrypted.decode()

    async def store_user_api_key(
        self,
        supabase: Client,
        user_id: str,
        provider: str,
        api_key: str
    ) -> bool:
        """
        Store user's API key for a provider

        Args:
            supabase: Supabase client
            user_id: User ID
            provider: Provider name (openai, anthropic, etc.)
            api_key: Plain text API key to store

        Returns:
            True if successful
        """
        try:
            # Encrypt the API key
            encrypted = self.encrypt_api_key(api_key)

            # Upsert to database
            result = supabase.table("user_provider_keys").upsert({
                "user_id": user_id,
                "provider": provider,
                "encrypted_key": encrypted,
                "updated_at": "now()"
            }, on_conflict="user_id,provider").execute()

            return True

        except Exception as e:
            print(f"Error storing API key: {e}")
            return False

    async def get_user_api_key(
        self,
        supabase: Client,
        user_id: str,
        provider: str
    ) -> Optional[str]:
        """
        Retrieve user's API key for a provider

        Args:
            supabase: Supabase client
            user_id: User ID
            provider: Provider name

        Returns:
            Decrypted API key or None if not found
        """
        try:
            # Query database
            result = supabase.table("user_provider_keys")\
                .select("encrypted_key")\
                .eq("user_id", user_id)\
                .eq("provider", provider)\
                .single()\
                .execute()

            if not result.data:
                return None

            # Decrypt and return
            encrypted = result.data["encrypted_key"]
            return self.decrypt_api_key(encrypted)

        except Exception as e:
            print(f"Error retrieving API key: {e}")
            return None

    async def delete_user_api_key(
        self,
        supabase: Client,
        user_id: str,
        provider: str
    ) -> bool:
        """
        Delete user's API key for a provider

        Args:
            supabase: Supabase client
            user_id: User ID
            provider: Provider name

        Returns:
            True if successful
        """
        try:
            supabase.table("user_provider_keys")\
                .delete()\
                .eq("user_id", user_id)\
                .eq("provider", provider)\
                .execute()

            return True

        except Exception as e:
            print(f"Error deleting API key: {e}")
            return False

    async def list_user_providers(
        self,
        supabase: Client,
        user_id: str
    ) -> list[str]:
        """
        List all providers for which user has stored API keys

        Args:
            supabase: Supabase client
            user_id: User ID

        Returns:
            List of provider names
        """
        try:
            result = supabase.table("user_provider_keys")\
                .select("provider")\
                .eq("user_id", user_id)\
                .execute()

            return [row["provider"] for row in result.data]

        except Exception as e:
            print(f"Error listing providers: {e}")
            return []


# Singleton instance
_api_key_service: Optional[APIKeyService] = None


def get_api_key_service() -> APIKeyService:
    """Get or create API key service instance"""
    global _api_key_service
    if _api_key_service is None:
        _api_key_service = APIKeyService()
    return _api_key_service
