import json
import requests
import pandas as pd

# The exact URL for the Home & Kitchen 5-core subset
url = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/benchmark/5core.jsonl"
output_file = "diderot_data.jsonl"
limit = 1000000

print(f"Connecting directly to: {url}")

# We stream the request so we don't crash your RAM
response = requests.get(url, stream=True)

if response.status_code == 200:
    print("Connection Successful! Downloading 1,000,000 records...")
    
    with open(output_file, 'wb') as f:
        count = 0
        # Iterate over lines in the stream
        for line in response.iter_lines():
            if line:
                f.write(line + b'\n')
                count += 1
                
                if count % 100000 == 0:
                    print(f"Progress: {count} records saved...")
                
                if count >= limit:
                    break
    print(f"DONE! {count} records saved to {output_file}")
else:
    print(f"Failed to connect. Error code: {response.status_code}")