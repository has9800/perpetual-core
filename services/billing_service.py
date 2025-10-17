"""
Billing service using Polar.sh
"""
import httpx
from datetime import datetime
from typing import Dict, Optional
from config.settings import get_settings
from models.database import UsageLog

settings = get_settings()


class BillingService:
    """Polar.sh billing integration"""
    
    def __init__(self):
        self.polar_api_url = "https://api.polar.sh/v1"
        self.headers = {
            "Authorization": f"Bearer {settings.POLAR_API_KEY}",
            "Content-Type": "application/json"
        }
        self.organization_id = settings.POLAR_ORGANIZATION_ID
        
        # Pricing (per 1M tokens)
        self.pricing = {
            'input_tokens': 0.10,   # $0.10 per 1M input tokens
            'output_tokens': 0.30,  # $0.30 per 1M output tokens
            'retrieval': 0.05       # $0.05 per 1K retrieval calls
        }
    
    def calculate_cost(self, usage: Dict) -> float:
        """
        Calculate cost for usage
        
        Args:
            usage: Dict with input_tokens, output_tokens, retrieval_calls
            
        Returns:
            Cost in USD
        """
        input_cost = (usage.get('input_tokens', 0) / 1_000_000) * self.pricing['input_tokens']
        output_cost = (usage.get('output_tokens', 0) / 1_000_000) * self.pricing['output_tokens']
        retrieval_cost = (usage.get('retrieval_calls', 0) / 1_000) * self.pricing['retrieval']
        
        return round(input_cost + output_cost + retrieval_cost, 6)
    
    async def track_usage(
        self,
        user_id: str,
        conversation_id: str,
        usage: Dict,
        supabase_client
    ):
        """
        Track usage in Supabase for billing
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            usage: Usage dict
            supabase_client: Supabase client instance
        """
        cost = self.calculate_cost(usage)
        
        # Log usage to Supabase
        usage_log = {
            'user_id': user_id,
            'conversation_id': conversation_id,
            'timestamp': datetime.utcnow().isoformat(),
            'input_tokens': usage.get('input_tokens', 0),
            'output_tokens': usage.get('output_tokens', 0),
            'total_tokens': usage.get('total_tokens', 0),
            'retrieval_calls': usage.get('retrieval_calls', 0),
            'retrieval_latency_ms': usage.get('retrieval_latency_ms', 0),
            'cost_usd': cost,
            'model': usage.get('model', 'unknown'),
            'endpoint': usage.get('endpoint', '/v1/chat/completions'),
            'status_code': usage.get('status_code', 200),
            'latency_ms': usage.get('latency_ms', 0)
        }
        
        supabase_client.table('usage_logs').insert(usage_log).execute()
        
        # Update user totals
        supabase_client.rpc('increment_user_usage', {
            'user_id_param': user_id,
            'tokens_param': usage.get('total_tokens', 0),
            'cost_param': cost
        }).execute()
    
    async def create_invoice(
        self,
        user_id: str,
        period_start: datetime,
        period_end: datetime,
        supabase_client
    ) -> Dict:
        """
        Create invoice in Polar.sh for user's usage
        
        Args:
            user_id: User ID
            period_start: Billing period start
            period_end: Billing period end
            supabase_client: Supabase client
            
        Returns:
            Invoice dict with invoice_id, amount, etc.
        """
        # Get usage for period
        usage_result = supabase_client.table('usage_logs').select(
            'input_tokens, output_tokens, retrieval_calls, cost_usd'
        ).eq('user_id', user_id).gte(
            'timestamp', period_start.isoformat()
        ).lte(
            'timestamp', period_end.isoformat()
        ).execute()
        
        if not usage_result.data:
            return {'success': False, 'message': 'No usage in period'}
        
        # Sum totals
        total_input = sum(r['input_tokens'] for r in usage_result.data)
        total_output = sum(r['output_tokens'] for r in usage_result.data)
        total_retrieval = sum(r['retrieval_calls'] for r in usage_result.data)
        total_cost = sum(r['cost_usd'] for r in usage_result.data)
        
        # Get user email
        user_result = supabase_client.table('users').select('email').eq(
            'id', user_id
        ).single().execute()
        
        email = user_result.data['email']
        
        # Create invoice in Polar.sh
        async with httpx.AsyncClient() as client:
            invoice_data = {
                "organization_id": self.organization_id,
                "customer_email": email,
                "amount": int(total_cost * 100),  # Cents
                "currency": "USD",
                "description": f"Perpetual AI Usage ({period_start.date()} to {period_end.date()})",
                "metadata": {
                    "user_id": user_id,
                    "input_tokens": total_input,
                    "output_tokens": total_output,
                    "retrieval_calls": total_retrieval,
                    "period_start": period_start.isoformat(),
                    "period_end": period_end.isoformat()
                }
            }
            
            response = await client.post(
                f"{self.polar_api_url}/invoices",
                json=invoice_data,
                headers=self.headers
            )
            
            if response.status_code == 201:
                invoice = response.json()
                return {
                    'success': True,
                    'invoice_id': invoice['id'],
                    'amount': total_cost,
                    'invoice_url': invoice.get('hosted_invoice_url')
                }
            else:
                return {
                    'success': False,
                    'error': response.text
                }
    
    async def check_user_balance(self, user_id: str, supabase_client) -> bool:
        """
        Check if user has sufficient balance
        
        Args:
            user_id: User ID
            supabase_client: Supabase client
            
        Returns:
            True if user has balance or is on paid tier
        """
        user_result = supabase_client.table('users').select(
            'tier, current_balance_usd'
        ).eq('id', user_id).single().execute()
        
        user = user_result.data
        
        # Enterprise always passes
        if user['tier'] == 'enterprise':
            return True
        
        # Pro needs positive balance
        if user['tier'] == 'pro':
            return user['current_balance_usd'] > 0
        
        # Free tier has limits but no balance check
        return True


# Singleton
_billing_service: Optional[BillingService] = None

def get_billing_service() -> BillingService:
    """Get or create BillingService singleton"""
    global _billing_service
    if _billing_service is None:
        _billing_service = BillingService()
    return _billing_service
