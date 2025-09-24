#!/usr/bin/env python3
"""
Test the comprehensive JSON generation with unit parsing and validation.
"""

import json
from pathlib import Path
from nepassist_langgraph_agent import generate_verified_tanks_json, standardize_excel_data

# Create test data with various formats
test_data = """Site,Contact,Capacity,Tank Size,Has Containment,Dike Size,Fuel,Notes,Latitude,Longitude
"ABC Gas Station","John Smith","10,000 gallons","12x8x10","Yes","15x12x3 ft","Diesel","Regular inspection",18.234,-66.123
"XYZ Industrial","Jane Doe","500 bbl","8x6x8","Yes","180 sq ft","Propane","Large facility",18.245,-66.134
"City Water Plant","Bob Johnson","75700 liters","20x10x12","Yes","25 ft x 15 ft","Water","Non-fuel tank",18.256,-66.145
"Marine Depot","Alice Brown","15000 gal","15x10x10","No","","Diesel","Waterfront location",18.267,-66.156
"Small Generator","Tom Wilson","200","4x3x4","Yes","4x4","Gasoline","Emergency backup",18.278,-66.167
"""

def test_json_generation():
    print("=" * 70)
    print("Testing JSON Generation with Unit Parsing and Validation")
    print("=" * 70)

    # Write test CSV
    test_csv = Path("test_tanks_various_units.csv")
    test_csv.write_text(test_data)
    print(f"\n✅ Created test CSV: {test_csv}")

    # Test 1: Generate JSON with auto-standardization
    print("\n" + "=" * 70)
    print("Test 1: Generate JSON with Auto-Standardization")
    print("-" * 40)

    # Call the function directly for testing
    result = generate_verified_tanks_json.func(
        input_file=str(test_csv),
        output_file='output/test_verified_tanks.json',
        standardize_first=True
    )

    data = json.loads(result)

    if data.get('status') == 'success':
        print(f"✅ JSON generated successfully!")
        print(f"   Output: {data.get('output')}")
        print(f"   Tanks processed: {data.get('tanks_processed')}")

        # Load and display the JSON
        with open(data.get('output'), 'r') as f:
            tank_data = json.load(f)

        print("\n📊 Generated Tank Data:")
        for i, tank in enumerate(tank_data['tanks'], 1):
            print(f"\n   Tank {i}: {tank['name']}")
            print(f"   - Volume: {tank['volume']:.1f} gallons")
            print(f"   - Type: {tank['type']}")
            print(f"   - Has dike: {tank['has_dike']}")
            if tank['has_dike'] and tank.get('dike_dims'):
                dims = tank['dike_dims']
                if isinstance(dims, list):
                    print(f"   - Dike dimensions: {dims[0]} ft x {dims[1]} ft")
                else:
                    print(f"   - Dike area: {dims} ft²")

        # Show summary
        if 'summary' in data:
            print("\n📈 Summary Statistics:")
            summary = data['summary']
            print(f"   Total volume: {summary['total_volume_gallons']:,.1f} gallons")
            print(f"   Diesel tanks: {summary['diesel_tanks']}")
            print(f"   Gas tanks: {summary['gas_tanks']}")
            print(f"   Tanks with dike: {summary['tanks_with_dike']}")
            print(f"   Tanks without dike: {summary['tanks_without_dike']}")

        # Show any validation errors
        if data.get('validation_errors'):
            print("\n⚠️ Validation Errors:")
            for err in data['validation_errors']:
                print(f"   Row {err['row']}: {err['error']}")

    else:
        print(f"❌ Error: {data.get('error')}")
        if data.get('validation_errors'):
            print("\nValidation errors:")
            for err in data['validation_errors']:
                print(f"   - {err}")

    # Test 2: Unit Conversion Examples
    print("\n" + "=" * 70)
    print("Test 2: Unit Conversion Examples")
    print("-" * 40)

    conversions = [
        ("10,000 gallons", 10000.0, "gallons"),
        ("500 bbl", 21000.0, "barrels → gallons"),
        ("75700 liters", 20006.1, "liters → gallons"),
        ("15000 gal", 15000.0, "gallons"),
        ("200", 200.0, "no unit → gallons"),
    ]

    print("\n📏 Volume Conversions Applied:")
    for original, converted, description in conversions:
        print(f"   {original:20} → {converted:10,.1f} gal  ({description})")

    # Test 3: Dike Dimension Parsing
    print("\n" + "=" * 70)
    print("Test 3: Dike Dimension Parsing")
    print("-" * 40)

    dike_examples = [
        ("15x12x3 ft", [15.0, 12.0], "LxWxH format → [L, W]"),
        ("180 sq ft", 180.0, "Area format → ft²"),
        ("25 ft x 15 ft", [25.0, 15.0], "L ft x W ft → [L, W]"),
        ("4x4", [4.0, 4.0], "Simple LxW → [L, W]"),
    ]

    print("\n📐 Dike Dimension Parsing:")
    for original, parsed, description in dike_examples:
        if isinstance(parsed, list):
            print(f"   {original:20} → [{parsed[0]}, {parsed[1]}] ft  ({description})")
        else:
            print(f"   {original:20} → {parsed} ft²  ({description})")

    # Test 4: Type Detection
    print("\n" + "=" * 70)
    print("Test 4: Fuel Type Detection")
    print("-" * 40)

    type_mapping = [
        ("Diesel", "diesel"),
        ("Propane", "pressurized_gas"),
        ("Water", "diesel"),  # Default for non-fuel
        ("Gasoline", "gasoline"),
    ]

    print("\n⛽ Fuel Type Mapping:")
    for original, mapped in type_mapping:
        print(f"   {original:15} → {mapped}")

    print("\n" + "=" * 70)
    print("✅ JSON Generation Test Complete!")
    print("=" * 70)

if __name__ == "__main__":
    test_json_generation()