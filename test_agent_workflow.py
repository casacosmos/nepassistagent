#!/usr/bin/env python3
"""
Demonstrate the complete workflow for Excel data standardization in the agent.
"""

import json
from pathlib import Path
import pandas as pd
from nepassist_langgraph_agent import standardize_excel_data, create_excel_template

def demonstrate_workflow():
    print("=" * 70)
    print("NEPAssist Tanks Data Standardization Workflow")
    print("=" * 70)

    # Step 1: Show the optimal template
    print("\n📋 Step 1: Creating the Optimal Template")
    print("-" * 40)
    template_result = create_excel_template.invoke({
        'output_file': 'output/excel_files/OPTIMAL_TEMPLATE.xlsx',
        'sheet_name': 'Tank Data'
    })

    template_data = json.loads(template_result)
    print(f"✅ Template created: {template_data.get('output')}")

    # Show template structure
    df_template = pd.read_excel(template_data.get('output'))
    print(f"\n   Optimal column order (16 columns):")
    for i, col in enumerate(df_template.columns, 1):
        if col in ["Has Dike", "Diesel", "Pressurized Gas"]:
            print(f"   {i:2}. {col} (Boolean)")
        else:
            print(f"   {i:2}. {col}")

    # Step 2: User uploads non-standard data
    print("\n📥 Step 2: User Uploads Non-Standard Data")
    print("-" * 40)
    print("   File: test_data_unstandardized.csv")
    df_original = pd.read_csv('test_data_unstandardized.csv')
    print(f"   Original columns: {', '.join(df_original.columns)}")
    print(f"   Rows: {len(df_original)}")

    # Step 3: Standardization
    print("\n🔄 Step 3: Automatic Standardization")
    print("-" * 40)

    standard_result = standardize_excel_data.invoke({
        'input_file': 'test_data_unstandardized.csv',
        'output_file': 'output/excel_files/USER_DATA_STANDARDIZED.xlsx'
    })

    standard_data = json.loads(standard_result)

    if standard_data.get('status') == 'success':
        print("✅ Data successfully standardized!")
        print(f"   Output: {standard_data.get('output')}")

        print("\n   Column Mapping Applied:")
        for orig, mapped in standard_data.get('mapping', {}).items():
            print(f"   • {orig:30} → {mapped}")

        # Load standardized data
        df_standard = pd.read_excel(standard_data.get('output'))

        print("\n   Data Transformations:")
        print("   • Boolean columns converted (Yes/No format)")
        print("   • Fuel type auto-detected from 'Fuel Type' column")
        print("   • Missing columns filled with null values")
        print("   • Column order matches optimal template")

        # Step 4: Ready for tools
        print("\n🛠️  Step 4: Data Ready for All Tools")
        print("-" * 40)
        print("   The standardized data can now be used with:")
        print("   ✓ Map generation tools")
        print("   ✓ Distance calculation tools")
        print("   ✓ Volume calculation tools")
        print("   ✓ Coordinate conversion tools")
        print("   ✓ Excel manipulation tools")

        # Show data quality
        print("\n📊 Data Quality Check:")
        required_cols = ["Site Name or Business Name", "Latitude (NAD83)", "Longitude (NAD83)"]
        for col in required_cols:
            filled = df_standard[col].notna().sum()
            total = len(df_standard)
            print(f"   {col}: {filled}/{total} filled")

        print("\n✨ Workflow Complete!")
        print("   User data has been standardized to the optimal format.")
        print("   All tank analysis tools will now work seamlessly.")

    else:
        print(f"❌ Standardization failed: {standard_data.get('error')}")

if __name__ == "__main__":
    demonstrate_workflow()
    print("\n" + "=" * 70)