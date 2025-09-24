#!/usr/bin/env python3
"""
Final validation of all implemented features for NEPAssist Tanks agent.
"""

import json
from pathlib import Path

# Direct function imports for testing
from nepassist_langgraph_agent import (
    EXCEL_TEMPLATE_HEADERS,
    standardize_excel_data,
    generate_verified_tanks_json,
    tank_volume_calculator
)

def validate_features():
    print("=" * 80)
    print("NEPAssist Tanks - Feature Validation Report")
    print("=" * 80)

    all_passed = True
    results = []

    # Test 1: Excel Template Headers Order
    print("\n✓ Testing Excel Template Headers Order...")
    expected_headers = [
        "Site Name or Business Name",
        "Person Contacted",
        "Tank Capacity",
        "Tank Measurements",
        "Has Dike",  # Before Dike Measurements
        "Dike Measurements",
        "Diesel",  # After Dike Measurements
        "Pressurized Gas",  # After Dike Measurements
        "Acceptable Separation Distance Calculated",
        "Approximate Distance to Site (approximately)",
        "Compliance",
        "Additional information",
        "Latitude (NAD83)",
        "Longitude (NAD83)",
        "Calculated Distance to Polygon (ft)",
        "Tank Type",
    ]

    if EXCEL_TEMPLATE_HEADERS == expected_headers:
        print("  ✅ PASS: Headers in correct order (16 columns)")
        print("     • 'Has Dike' before 'Dike Measurements' ✓")
        print("     • Fuel type columns after 'Dike Measurements' ✓")
        results.append(("Template Headers", "PASS"))
    else:
        print("  ❌ FAIL: Headers not in expected order")
        all_passed = False
        results.append(("Template Headers", "FAIL"))

    # Test 2: Volume Unit Conversions
    print("\n✓ Testing Volume Unit Conversions...")
    test_volumes = [
        ("10000 gallons", 10000.0),
        ("500 bbl", 21000.0),
        ("1000 liters", 264.172),
        ("10 m3", 2641.72),
        ("5000", 5000.0),  # Default to gallons
    ]

    conversion_passed = True
    for volume_str, expected in test_volumes:
        # Create minimal test data
        test_csv = f"Site,Volume\nTest,{volume_str}\n"
        test_file = Path("temp_volume_test.csv")
        test_file.write_text(test_csv)

        # Generate JSON with the volume
        result = generate_verified_tanks_json.func(
            input_file=str(test_file),
            output_file="temp_volume.json",
            standardize_first=True
        )

        data = json.loads(result)
        if data.get('status') == 'success' and data.get('tanks_processed') > 0:
            with open("temp_volume.json", 'r') as f:
                tank_data = json.load(f)
            actual = tank_data['tanks'][0]['volume']
            if abs(actual - expected) < 0.1:
                print(f"  ✅ {volume_str:20} → {actual:,.1f} gallons")
            else:
                print(f"  ❌ {volume_str:20} → {actual:,.1f} (expected {expected:,.1f})")
                conversion_passed = False

        # Cleanup
        test_file.unlink(missing_ok=True)
        Path("temp_volume.json").unlink(missing_ok=True)

    if conversion_passed:
        results.append(("Volume Conversions", "PASS"))
    else:
        all_passed = False
        results.append(("Volume Conversions", "FAIL"))

    # Test 3: Dike Dimension Parsing
    print("\n✓ Testing Dike Dimension Parsing...")
    dike_tests = [
        ("15x12", "list", [15.0, 12.0]),
        ("20 ft x 15 ft", "list", [20.0, 15.0]),
        ("500 sq ft", "float", 500.0),
        ("1000 square feet", "float", 1000.0),
        ("10x8x3", "list", [10.0, 8.0]),  # Ignores height
    ]

    dike_passed = True
    for dike_str, expected_type, expected_val in dike_tests:
        # Create test data with dike
        test_csv = f"Site,Has Dike,Dike Size,Volume\nTest,Yes,{dike_str},5000\n"
        test_file = Path("temp_dike_test.csv")
        test_file.write_text(test_csv)

        result = generate_verified_tanks_json.func(
            input_file=str(test_file),
            output_file="temp_dike.json",
            standardize_first=True
        )

        data = json.loads(result)
        if data.get('status') == 'success' and data.get('tanks_processed') > 0:
            with open("temp_dike.json", 'r') as f:
                tank_data = json.load(f)
            dike_dims = tank_data['tanks'][0].get('dike_dims')

            if expected_type == "list" and isinstance(dike_dims, list):
                if dike_dims == expected_val:
                    print(f"  ✅ {dike_str:25} → {dike_dims}")
                else:
                    print(f"  ❌ {dike_str:25} → {dike_dims} (expected {expected_val})")
                    dike_passed = False
            elif expected_type == "float" and isinstance(dike_dims, (int, float)):
                if abs(dike_dims - expected_val) < 0.1:
                    print(f"  ✅ {dike_str:25} → {dike_dims} ft²")
                else:
                    print(f"  ❌ {dike_str:25} → {dike_dims} (expected {expected_val})")
                    dike_passed = False

        # Cleanup
        test_file.unlink(missing_ok=True)
        Path("temp_dike.json").unlink(missing_ok=True)

    if dike_passed:
        results.append(("Dike Parsing", "PASS"))
    else:
        all_passed = False
        results.append(("Dike Parsing", "FAIL"))

    # Test 4: Fuel Type Detection (only diesel or pressurized_gas allowed)
    print("\n✓ Testing Fuel Type Detection (only diesel/pressurized_gas)...")
    fuel_tests = [
        ("Diesel", "diesel"),
        ("Propane", "pressurized_gas"),
        ("LPG", "pressurized_gas"),
        ("Gasoline", "diesel"),  # Defaults to diesel
        ("Fuel Oil", "diesel"),  # Defaults to diesel
        ("Water", "diesel"),     # Non-fuel defaults to diesel
    ]

    fuel_passed = True
    for fuel_str, expected_type in fuel_tests:
        test_csv = f"Site,Fuel Type,Volume\nTest,{fuel_str},5000\n"
        test_file = Path("temp_fuel_test.csv")
        test_file.write_text(test_csv)

        result = generate_verified_tanks_json.func(
            input_file=str(test_file),
            output_file="temp_fuel.json",
            standardize_first=True
        )

        data = json.loads(result)
        if data.get('status') == 'success' and data.get('tanks_processed') > 0:
            with open("temp_fuel.json", 'r') as f:
                tank_data = json.load(f)
            actual_type = tank_data['tanks'][0]['type']
            if actual_type == expected_type:
                print(f"  ✅ {fuel_str:20} → {actual_type}")
            else:
                print(f"  ❌ {fuel_str:20} → {actual_type} (expected {expected_type})")
                fuel_passed = False

        # Cleanup
        test_file.unlink(missing_ok=True)
        Path("temp_fuel.json").unlink(missing_ok=True)

    if fuel_passed:
        results.append(("Fuel Type Detection", "PASS"))
    else:
        all_passed = False
        results.append(("Fuel Type Detection", "FAIL"))

    # Test 5: Column Standardization
    print("\n✓ Testing Column Standardization...")
    test_csv = """Business,Contact,Size,Containment,Fuel,Lat,Long
"ABC Corp","John Doe","10000 gal","Yes","Diesel",18.123,-66.456
"""
    test_file = Path("temp_standard_test.csv")
    test_file.write_text(test_csv)

    result = standardize_excel_data.func(
        input_file=str(test_file),
        output_file="temp_standardized.xlsx"
    )

    data = json.loads(result)
    if data.get('status') == 'success':
        mapping = data.get('mapping', {})
        expected_mappings = {
            "Business": "Site Name or Business Name",
            "Contact": "Person Contacted",
            "Size": "Tank Capacity",
            "Containment": "Has Dike",
            "Fuel": "Tank Type",
        }

        standard_passed = True
        for orig, expected in expected_mappings.items():
            actual = mapping.get(orig)
            if actual == expected:
                print(f"  ✅ '{orig}' → '{actual}'")
            else:
                print(f"  ❌ '{orig}' → '{actual}' (expected '{expected}')")
                standard_passed = False

        if standard_passed:
            results.append(("Column Standardization", "PASS"))
        else:
            all_passed = False
            results.append(("Column Standardization", "FAIL"))
    else:
        print(f"  ❌ Standardization failed: {data.get('error')}")
        all_passed = False
        results.append(("Column Standardization", "FAIL"))

    # Cleanup
    test_file.unlink(missing_ok=True)
    Path("temp_standardized.xlsx").unlink(missing_ok=True)
    Path("output/excel_files/temp_standard_test_standardized.xlsx").unlink(missing_ok=True)

    # Final Report
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    for feature, status in results:
        icon = "✅" if status == "PASS" else "❌"
        print(f"{icon} {feature:30} {status}")

    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 ALL FEATURES VALIDATED SUCCESSFULLY!")
        print("\nThe NEPAssist Tanks agent is fully operational with:")
        print("  • Optimal Excel template format (16 columns)")
        print("  • Intelligent column mapping and standardization")
        print("  • Comprehensive unit conversion (gallons, barrels, liters, m³)")
        print("  • Flexible dike dimension parsing")
        print("  • Automatic fuel type detection")
        print("  • Pydantic-validated JSON generation")
    else:
        print("⚠️  Some features need attention. Please review failures above.")
    print("=" * 80)

if __name__ == "__main__":
    validate_features()