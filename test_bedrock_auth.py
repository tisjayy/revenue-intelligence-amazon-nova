"""
Test Bedrock API Key Authentication
"""

import os
import requests
import json

# Set your bearer token
bearer_token = os.environ.get('AWS_BEARER_TOKEN_BEDROCK')

if not bearer_token:
    print("❌ ERROR: AWS_BEARER_TOKEN_BEDROCK not set")
    print("\nRun this first:")
    print('$env:AWS_BEARER_TOKEN_BEDROCK="your-api-key-here"')
    exit(1)

# Bedrock endpoint
region = 'us-east-1'
model_id = 'us.amazon.nova-lite-v1:0'  # Amazon Nova Lite
endpoint = f'https://bedrock-runtime.{region}.amazonaws.com/model/{model_id}/invoke'

# Test request
headers = {
    'Authorization': f'Bearer {bearer_token}',
    'Content-Type': 'application/json',
    'Accept': 'application/json'
}

body = {
    "messages": [
        {
            "role": "user",
            "content": [{"text": "Say 'Authentication successful!' if you can read this."}]
        }
    ],
    "inferenceConfig": {
        "maxTokens": 100,
        "temperature": 0.7
    }
}

print("Testing Bedrock API Key authentication...")
print(f"Endpoint: {endpoint}")
print(f"Model: {model_id}")

try:
    response = requests.post(endpoint, headers=headers, json=body)
    
    if response.status_code == 200:
        result = response.json()
        if 'output' in result and 'message' in result['output']:
            message = result['output']['message']
            if 'content' in message:
                print(f"\n✅ SUCCESS: {message['content'][0]['text']}")
        else:
            print(f"\n✅ Response received: {result}")
    else:
        print(f"\n❌ ERROR {response.status_code}: {response.text}")
        
except Exception as e:
    print(f"\n❌ ERROR: {e}")
