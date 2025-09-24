#!/usr/bin/env python3
"""Debug why gasoline is being detected as pressurized_gas"""

import json
from pathlib import Path
from nepassist_langgraph_agent import generate_verified_tanks_json

# Test specifically with Gasoline
test_csv = """Site,Fuel Type,Volume
Test Station,Gasoline,5000
"""

test_file = Path("debug_gasoline.csv")
test_file.write_text(test_csv)

result = generate_verified_tanks_json.func(
    input_file=str(test_file),
    output_file="debug_gasoline.json",
    standardize_first=True
)

data = json.loads(result)
print(f"Status: {data.get('status')}")

if data.get('status') == 'success':
    with open("debug_gasoline.json", 'r') as f:
        tank_data = json.load(f)

    tank = tank_data['tanks'][0]
    print(f"Tank name: {tank['name']}")
    print(f"Tank type: {tank['type']}")
    print(f"Expected: diesel")
    print(f"Match: {tank['type'] == 'diesel'}")

# Clean up
test_file.unlink()
Path("debug_gasoline.json").unlink(missing_ok=True)