import duckdb

csv_path = "accepted_2007_to_2018Q4.csv"

# Quick check: Count total rows in the CSV using DuckDB (takes < 2 seconds)
print("--- Counting Rows ---")
row_count = duckdb.query(f"SELECT COUNT(*) FROM '{csv_path}'").fetchone()[0]
print(f"Total Records: {row_count:,}")

# Look at the target variable: 'loan_status'
print("\n--- Loan Status Breakdown ---")
status_counts = duckdb.query(f"""
    SELECT loan_status, COUNT(*) as count 
    FROM '{csv_path}' 
    GROUP BY loan_status 
    ORDER BY count DESC
""").df()

print(status_counts)