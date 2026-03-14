import os
import csv
import glob
from pathlib import Path

def combine_csv_results():
    """
    Combine all CSV files in the results directory into one master CSV file
    """
    # Define paths
    results_dir = Path("Call_recordings_AI/results")
    output_file = results_dir / "combined_analysis_results.csv"
    
    # Check if results directory exists
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return False
    
    # Find all CSV files in the results directory
    csv_files = list(results_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"❌ No CSV files found in {results_dir}")
        return False
    
    print(f"📁 Found {len(csv_files)} CSV files to combine:")
    for csv_file in csv_files:
        print(f"   • {csv_file.name}")
    
    # Combine all CSV files
    combined_rows = []
    header_written = False
    
    for csv_file in csv_files:
        print(f"📖 Reading: {csv_file.name}")
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                
                # Skip header row for all files except the first one
                if not header_written:
                    header = next(reader)
                    combined_rows.append(header)
                    header_written = True
                else:
                    # Skip header for subsequent files
                    next(reader)
                
                # Add all data rows
                for row in reader:
                    combined_rows.append(row)
                    
        except Exception as e:
            print(f"❌ Error reading {csv_file.name}: {e}")
            continue
    
    # Write combined data to output file
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)
            writer.writerows(combined_rows)
        
        print(f"✅ Successfully combined {len(combined_rows)-1} rows into: {output_file}")
        print(f"📊 Total files processed: {len(csv_files)}")
        print(f"📈 Total data rows: {len(combined_rows)-1}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error writing combined file: {e}")
        return False

def list_csv_files():
    """
    List all CSV files in the results directory
    """
    results_dir = Path("Call_recordings_AI/results")
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    csv_files = list(results_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"📁 No CSV files found in {results_dir}")
        return
    
    print(f"📁 Found {len(csv_files)} CSV files:")
    for i, csv_file in enumerate(csv_files, 1):
        # Count rows in each file
        try:
            with open(csv_file, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                row_count = sum(1 for row in reader) - 1  # Subtract header
                print(f"   {i}. {csv_file.name} ({row_count} data rows)")
        except Exception as e:
            print(f"   {i}. {csv_file.name} (Error reading: {e})")

def main():
    """
    Main function with menu options
    """
    print("🔄 CSV Results Combiner")
    print("=" * 30)
    
    while True:
        print("\nOptions:")
        print("1. List CSV files")
        print("2. Combine all CSV files")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            print("\n" + "="*30)
            list_csv_files()
            print("="*30)
            
        elif choice == "2":
            print("\n" + "="*30)
            success = combine_csv_results()
            if success:
                print("\n🎉 CSV combination completed successfully!")
            else:
                print("\n❌ CSV combination failed!")
            print("="*30)
            
        elif choice == "3":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main() 