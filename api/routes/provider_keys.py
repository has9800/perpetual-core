"""
Provider API Key Management Endpoints
Allows users to manage their encrypted API keys for external providers
"""
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Optional
from api.dependencies import get_supabase
from services.api_key_service import get_api_key_service
from services.provider_config import list_supported_models, PROVIDERS

router = APIRouter(prefix="/provider-keys", tags=["provider-keys"])


class AddProviderKeyRequest(BaseModel):
    """Request to add a provider API key"""
    provider: str = Field(..., description="Provider name (openai, anthropic, xai, etc.)")
    api_key: str = Field(..., description="API key for the provider")


class ProviderKeyResponse(BaseModel):
    """Response after adding/deleting a key"""
    success: bool
    message: str
    provider: Optional[str] = None


class ProviderListResponse(BaseModel):
    """List of providers with keys"""
    providers: List[str]


class SupportedProvidersResponse(BaseModel):
    """List of all supported providers and their models"""
    providers: dict


@router.post("/add", response_model=ProviderKeyResponse)
async def add_provider_key(
    request: Request,
    key_request: AddProviderKeyRequest,
    supabase = Depends(get_supabase),
    api_key_service = Depends(get_api_key_service)
):
    """
    Add or update API key for a provider

    Stores the API key encrypted. Users provide their own API keys
    for external providers (OpenAI, Anthropic, etc.)
    """
    user = request.state.user

    # Validate provider exists
    if key_request.provider not in PROVIDERS:
        raise HTTPException(
            400,
            f"Unknown provider: {key_request.provider}. "
            f"Supported providers: {', '.join(PROVIDERS.keys())}"
        )

    try:
        success = await api_key_service.store_user_api_key(
            supabase=supabase,
            user_id=user['user_id'],
            provider=key_request.provider,
            api_key=key_request.api_key
        )

        if success:
            return ProviderKeyResponse(
                success=True,
                message=f"API key for {key_request.provider} added successfully",
                provider=key_request.provider
            )
        else:
            raise HTTPException(500, "Failed to store API key")

    except Exception as e:
        print(f"Error adding provider key: {e}")
        raise HTTPException(500, f"Failed to add API key: {str(e)}")


@router.delete("/delete/{provider}", response_model=ProviderKeyResponse)
async def delete_provider_key(
    request: Request,
    provider: str,
    supabase = Depends(get_supabase),
    api_key_service = Depends(get_api_key_service)
):
    """
    Delete API key for a provider

    Removes the encrypted API key from storage
    """
    user = request.state.user

    try:
        success = await api_key_service.delete_user_api_key(
            supabase=supabase,
            user_id=user['user_id'],
            provider=provider
        )

        if success:
            return ProviderKeyResponse(
                success=True,
                message=f"API key for {provider} deleted successfully",
                provider=provider
            )
        else:
            raise HTTPException(500, "Failed to delete API key")

    except Exception as e:
        print(f"Error deleting provider key: {e}")
        raise HTTPException(500, f"Failed to delete API key: {str(e)}")


@router.get("/list", response_model=ProviderListResponse)
async def list_provider_keys(
    request: Request,
    supabase = Depends(get_supabase),
    api_key_service = Depends(get_api_key_service)
):
    """
    List all providers for which user has stored API keys

    Does not return the actual keys, just the provider names
    """
    user = request.state.user

    try:
        providers = await api_key_service.list_user_providers(
            supabase=supabase,
            user_id=user['user_id']
        )

        return ProviderListResponse(providers=providers)

    except Exception as e:
        print(f"Error listing provider keys: {e}")
        raise HTTPException(500, f"Failed to list providers: {str(e)}")


@router.get("/supported", response_model=SupportedProvidersResponse)
async def get_supported_providers():
    """
    Get list of all supported providers and their models

    Public endpoint - no auth required
    """
    try:
        providers_info = {
            name: {
                "models": config.models,
                "format": config.format,
                "supports_streaming": config.supports_streaming
            }
            for name, config in PROVIDERS.items()
        }

        return SupportedProvidersResponse(providers=providers_info)

    except Exception as e:
        print(f"Error getting supported providers: {e}")
        raise HTTPException(500, f"Failed to get providers: {str(e)}")
