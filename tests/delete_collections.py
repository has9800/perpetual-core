from qdrant_client import QdrantClient
import os

client = QdrantClient(
    url=os.getenv("https://886eb61f-20e6-493e-b467-fbbcad529efd.us-west-2-0.aws.cloud.qdrant.io"),
    api_key=os.getenv("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.R4UyNOac6Gvf1yDferYocoEivA8cohdd5zf5aSgmvoE")
)

client.delete_collection("conversations")
print("✅ Deleted collection")
