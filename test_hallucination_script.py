# Test script with known hallucinations for testing enhanced detection
import datetime

import requests


def test_function():
    # Known hallucination 1: Response object doesn't have extract_json_data method
    response = requests.get("https://api.example.com/data")
    data = response.extract_json_data()

    # Known hallucination 2: datetime object doesn't have add_days method
    current_time = datetime.datetime.now()
    future_time = current_time.add_days(7)

    # Known hallucination 3: requests.post doesn't have auto_retry parameter
    result = requests.post(
        "https://api.example.com/submit", 
        json=data, 
        auto_retry=True,
        retry_attempts=3,
    )

    return result

if __name__ == "__main__":
    test_function()
