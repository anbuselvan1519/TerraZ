import pandas as pd

# ===============================
# 1. Load all files with full paths
# ===============================
base_path = r"C:\Users\admin\OneDrive\Documents\EV_mineral_exploration\rdbms"

deposits = pd.read_csv(base_path + r"\Deposits.txt", sep="\t", low_memory=False)
commodities = pd.read_csv(base_path + r"\Commodity.txt", sep="\t", low_memory=False)
location = pd.read_csv(base_path + r"\Location.txt", sep="\t", low_memory=False)
coords = pd.read_csv(base_path + r"\Coords.txt", sep="\t", low_memory=False)
districts = pd.read_csv(base_path + r"\Districts.txt", sep="\t", low_memory=False)
names = pd.read_csv(base_path + r"\Names.txt", sep="\t", low_memory=False)

grades = pd.read_excel(r"C:\Users\admin\OneDrive\Documents\EV_mineral_exploration\grade.xlsx")

# ===============================
# 2. Normalize column names
# ===============================
for df in [deposits, commodities, location, coords, districts, names, grades]:
    df.columns = df.columns.str.strip().str.lower()

# ===============================
# 3. Prepare grades table
# ===============================
grades.rename(columns={col: col.lower() for col in grades.columns}, inplace=True)
if "dep_id" not in grades.columns:
    grades.rename(columns={"qdep_id": "dep_id"}, inplace=True)

# Keep only q01–q26
grade_cols = ["dep_id"] + [f"q{i:02d}" for i in range(1, 27)]
grades = grades[grade_cols]

# ===============================
# 4. Merge everything
# ===============================
merged = pd.merge(commodities[["dep_id", "commod"]], deposits[["dep_id"]], on="dep_id", how="left")
merged = pd.merge(merged, grades, on="dep_id", how="left")
merged = pd.merge(merged, location[["dep_id", "region", "country", "state_prov", "county"]], on="dep_id", how="left")
merged = pd.merge(merged, coords[["dep_id", "wgs84_lat", "wgs84_lon"]], on="dep_id", how="left")
merged = pd.merge(merged, names[["dep_id", "name"]], on="dep_id", how="left")
merged = pd.merge(merged, districts[["dep_id", "district"]], on="dep_id", how="left")

# ===============================
# 5. Compute final grade
# ===============================
def assign_grade(row):
    try:
        score = sum([row[f"q{i:02d}"] if pd.notnull(row[f"q{i:02d}"]) and row[f"q{i:02d}"] != "NULL" else 0 for i in range(1, 27)])
    except Exception:
        score = 0
    if score >= 20:
        return "A"
    elif score >= 15:
        return "B"
    elif score >= 10:
        return "C"
    elif score >= 5:
        return "D"
    else:
        return "E"

merged["final_grade"] = merged.apply(assign_grade, axis=1)

# ===============================
# 6. Keep only required columns
# ===============================
final_dataset = merged[[
    "dep_id", "commod", "final_grade",
    "wgs84_lat", "wgs84_lon", "region", "country", "state_prov", "county",
    "name","district"
]]

# ===============================
# 7. Clean missing values
# ===============================
final_dataset = final_dataset.replace([0, "0", "", " ", None, pd.NA, pd.NaT], "NULL")
final_dataset = final_dataset.fillna("NULL")

# ===============================
# 8. Ensure row count 1000–5000
# ===============================
rows = len(final_dataset)
if rows < 1000:
    print(f"⚠️ Dataset too small ({rows} rows). Consider including more dep_ids or relaxing filters.")
elif rows > 5000:
    final_dataset = final_dataset.sample(5000, random_state=42)

# ===============================
# 9. Save final dataset
# ===============================
output_path = r"C:\Users\admin\OneDrive\Documents\EV_mineral_exploration\clean_mineral_dataset.csv"
final_dataset.to_csv(output_path, index=False)

print("✅ Final dataset saved:", final_dataset.shape)
print(final_dataset.head(10))
