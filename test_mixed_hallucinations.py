# Test script with mix of valid and invalid code
import datetime
import json

import requests


def test_mixed_functions():
    # Valid code
    response = requests.get("https://api.example.com/data")
    current_time = datetime.datetime.now()
    data = json.loads('{"key": "value"}')

    # Invalid code (hallucinations)
    fake_response = requests.fetch("https://api.example.com/data")  # fetch method doesn't exist
    fake_time = datetime.current()  # current method doesn't exist  
    fake_data = json.parse('{"key": "value"}')  # parse method doesn't exist

    return data, fake_data

if __name__ == "__main__":
    test_mixed_functions()
