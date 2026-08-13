import duckdb

raw_data = "accepted_2007_to_2018Q4.csv"
clean_data = "baseline_data.csv"

print("Filtering for finished loans only and ignoring weird text rows...")

# all_varchar=True tells the database to read everything as a string to avoid crashing
query = f"""
COPY (
    SELECT 
        *,
        CASE WHEN loan_status = 'Charged Off' THEN 1 ELSE 0 END AS is_default
    FROM read_csv_auto('{raw_data}', all_varchar=True)
    WHERE loan_status IN ('Fully Paid', 'Charged Off')
) TO '{clean_data}' (HEADER, DELIMITER ',');
"""

duckdb.execute(query)
print(f"Success! Saved the clean data to {clean_data}")