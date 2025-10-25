"""Test script for hallucination detection"""
import requests
from requests.auth import HTTPBasicAuth

# Valid usage
session = requests.Session()
response = session.get('https://api.example.com', auth=HTTPBasicAuth('user', 'pass'))

# Potential hallucination - fake method
session.magic_request('https://fake.com')

# Another hallucination - wrong parameters
requests.get('https://example.com', fake_param=True, another_fake='value')
