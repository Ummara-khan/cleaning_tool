import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import json
import re
import ast
from dateutil import parser

st.set_page_config(page_title="Universal Data Cleaning Tool", layout="wide")
st.title("🧹 Data Cleaning Tool")

# ---------- Utilities ----------
def load_data(uploaded_file):
    """Smart loader for CSV, Excel, JSON, TXT with auto JSON normalization"""
    try:
        if uploaded_file.name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".json"):
            import json
            raw = json.load(uploaded_file)
            
            # Detect type of JSON and normalize accordingly
            if isinstance(raw, dict):
                if "users" in raw:
                    return normalize_nested_json(raw)  # Existing users JSON handler
                elif "items" in raw:
                    return normalize_items_json(raw)   # New items JSON handler
                else:
                    # Fallback: normalize any dict
                    return pd.json_normalize(raw)
            elif isinstance(raw, list):
                return pd.json_normalize(raw)
            else:
                st.error("Unsupported JSON structure.")
                return None

        elif uploaded_file.name.endswith(".txt"):
            return pd.read_csv(uploaded_file, delimiter="\t")
        else:
            st.error("Unsupported file format.")
            return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


# ---------- Sidebar ----------
# Sidebar with black background
st.markdown(
    """
    <style>
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: black;
    }
    /* Sidebar header text */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] h4 {
        color: white;
        text-align: center;
    }
    /* Sidebar text */
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] label {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.sidebar.title("")
st.sidebar.markdown("---")
st.sidebar.header("")

# ---------- Flexible Cleaning Functions ----------
def standardize_missing(df):
    placeholders = ["unknown", "", "none", "n/a", "na", "null", "??", "invalid_email@", "[' messy ']"]
    df = df.replace(placeholders, np.nan, regex=False)
    return df

def detect_text_columns(df):
    return df.select_dtypes(include="object").columns.tolist()

def normalize_text(df, text_columns):
    for col in text_columns:
        df[col] = df[col].apply(lambda x: str(x).strip() if pd.notna(x) else np.nan)
        if col.lower() in ["color"]:
            df[col] = df[col].str.lower().replace({'blúe':'blue'})
        if col.lower() in ["origin"]:
            df[col] = df[col].str.upper()
    return df

def detect_email_columns(df):
    return [col for col in df.columns if "email" in col.lower()]

def validate_emails(df, email_columns):
    for col in email_columns:
        df[col] = df[col].apply(lambda x: x if pd.notna(x) and re.match(r"[^@]+@[^@]+\.[^@]+", str(x)) else np.nan)
    return df

def detect_list_columns(df, sample_size=100):
    list_cols = []
    for col in df.columns:
        if df[col].dtype == "object":
            sample = df[col].dropna().astype(str).head(sample_size)
            if all(s.startswith("[") and s.endswith("]") for s in sample if s):
                list_cols.append(col)
    return list_cols

def parse_list_columns(df, list_columns):
    for col in list_columns:
        df[col] = df[col].replace(["None", "[' Messy ']"], "[]", regex=False)
        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
    return df

def clean_zipcode_column(df):
    for col in df.columns:
        if "zip" in col.lower() or "postal" in col.lower():
            df[col] = df[col].replace(["Abcde"], np.nan)
            df[col] = df[col].apply(lambda x: str(int(float(x))) if pd.notna(x) and str(x).replace(".", "").isdigit() else np.nan)
    return df

def detect_date_columns(df, sample_size=50):
    date_cols = []
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().astype(str).head(sample_size)
            parsed = 0
            for val in sample:
                try:
                    parser.parse(val)
                    parsed += 1
                except:
                    continue
            if parsed / max(1, len(sample)) > 0.6:
                date_cols.append(col)
    return date_cols

def standardize_dates(df):
    date_cols = detect_date_columns(df)
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")
    return df

def correct_dtypes(df):
    for col in df.columns:
        col_lower = col.lower()
        # ID columns to string
        if "id" in col_lower:
            df[col] = df[col].astype(str)
        # Numeric columns
        elif any(k in col_lower for k in ["salary", "age", "weight", "gsm"]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        # Dates handled elsewhere
        # Boolean columns
        elif df[col].dtype == object and set(df[col].dropna().unique()) <= {True, False, "True", "False", "true", "false"}:
            df[col] = df[col].replace({"true": True, "false": False, "True": True, "False": False})
        # Leave other object columns as string
        elif df[col].dtype == object:
            df[col] = df[col].astype(str)
    return df



def stringify_unhashables(df):
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
            df[col] = df[col].astype(str)
    return df



def normalize_nested_json(data):
    """
    Flatten JSON and preserve numeric fields like age and salary.
    Handle booleans and dates without destroying valid data.
    """
    if isinstance(data, dict) and "users" in data:
        data = data["users"]
    elif isinstance(data, dict):
        data = [data]

    df = pd.json_normalize(data)

    for col in df.columns:
        # Convert numeric columns safely
        if col.lower() in ["age", "salary"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Parse dates only for date columns
        elif any(word in col.lower() for word in ["date", "time", "dob", "join"]):
            df[col] = pd.to_datetime(df[col], errors="coerce")

        # Convert booleans safely
        elif df[col].dtype == object:
            df[col] = df[col].replace({"true": True, "false": False, "True": True, "False": False})

    return df



def normalize_items_json(data):
    """
    Flatten 'items' JSON array into a DataFrame.
    - Preserves numeric fields
    - Parses dates
    - Keeps text fields clean
    - Converts booleans
    - Handles missing or nested values safely
    """
    # If it's wrapped in a top-level 'items'
    if isinstance(data, dict) and "items" in data:
        data = data["items"]
    elif isinstance(data, dict):
        data = [data]
    
    df = pd.json_normalize(data)

    # Clean columns safely
    for col in df.columns:
        col_lower = col.lower()
        
        # Numeric fields
        if any(k in col_lower for k in ["price", "quantity", "sales_order", "oz"]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Date fields
        elif any(k in col_lower for k in ["date", "trx_date", "time"]):
            df[col] = pd.to_datetime(df[col], errors="coerce")
        
        # Boolean-like fields
        elif df[col].dtype == object:
            df[col] = df[col].replace({"true": True, "false": False, "True": True, "False": False})
            # Remove placeholders for missing values
            df[col] = df[col].replace(["unknown", "none", "n/a", "na", "null", "??", ""], np.nan)
        
        # List/dict objects -> string to prevent unhashable errors
        if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
            df[col] = df[col].astype(str)
    
    # Strip text columns
    text_cols = df.select_dtypes(include="object").columns
    for col in text_cols:
        df[col] = df[col].apply(lambda x: str(x).strip() if pd.notna(x) else np.nan)

    # Remove duplicates
    df = df.drop_duplicates()

    return df



# ---------- General Cleaning ----------
def standardize_missing(df):
    placeholders = ["unknown", "", "none", "n/a", "na", "null", "??", "invalid-email@", "invalid-phone"]
    df = df.replace(placeholders, np.nan)
    return df

def stringify_unhashables(df):
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
            df[col] = df[col].astype(str)
    return df


# ---------- Auto Cleaning ----------
def normalize_nested_json(data):
    """
    Flatten JSON and preserve numeric fields like age and salary.
    Handle booleans and dates without destroying valid data.
    """
    if isinstance(data, dict) and "users" in data:
        data = data["users"]
    elif isinstance(data, dict):
        data = [data]

    df = pd.json_normalize(data)

    for col in df.columns:
        # Convert numeric columns safely
        if col.lower() in ["age", "salary"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Parse dates only for date columns
        elif any(word in col.lower() for word in ["date", "time", "dob", "join"]):
            df[col] = pd.to_datetime(df[col], errors="coerce")

        # Convert booleans safely
        elif df[col].dtype == object:
            df[col] = df[col].replace({"true": True, "false": False, "True": True, "False": False})

    return df


def auto_clean(df):
    # Replace placeholders but keep numeric columns like age intact
    placeholders = ["unknown", "", "none", "n/a", "na", "null", "??", "invalid_email@", "invalid-phone"]
    for col in df.columns:
        if col.lower() not in ["age", "salary"]:
            df[col] = df[col].replace(placeholders, np.nan)

    df = stringify_unhashables(df)
    df = df.drop_duplicates()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    text_cols = detect_text_columns(df)
    df = normalize_text(df, text_cols)

    email_cols = detect_email_columns(df)
    df = validate_emails(df, email_cols)

    list_cols = detect_list_columns(df)
    df = parse_list_columns(df, list_cols)

    df = clean_zipcode_column(df)
    df = standardize_dates(df)

    # Correct dtypes safely
    df = correct_dtypes(df)
    return df

import pandas as pd
import numpy as np
import json
from pandas import json_normalize

def flatten_nested_json_safe(data, sep="_"):
    """
    Flatten nested JSON dynamically without exploding lists.
    Preserves numeric fields, handles booleans, dates, and placeholders.
    Lists are converted to JSON strings to avoid row duplication.
    """
    # If top-level dict with one key pointing to list, use it
    if isinstance(data, dict):
        if any(isinstance(v, list) for v in data.values()):
            for k, v in data.items():
                if isinstance(v, list):
                    data = v
                    break
        else:
            data = [data]

    # Recursive function to flatten dicts
    def recursive_flatten(row, parent_key=""):
        items = {}
        if isinstance(row, dict):
            for k, v in row.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.update(recursive_flatten(v, new_key))
                elif isinstance(v, list):
                    # Convert list to JSON string to avoid exploding
                    items[new_key] = json.dumps(v)
                else:
                    items[new_key] = v
        else:
            items[parent_key] = row
        return items

    # Apply recursive flattening to all rows
    flattened_records = [recursive_flatten(r) for r in data]
    df = pd.DataFrame(flattened_records)

    # Convert numeric-like columns safely
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    # Replace placeholders dynamically
    placeholders = ["unknown", "", "none", "n/a", "na", "null", "??", "invalid_email@", "invalid-phone"]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].replace(placeholders, np.nan)
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df.drop_duplicates().reset_index(drop=True)
    return df


# ---------- Profiling & Export ----------
def profile_data(df):
    profile = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing_pct = round(df[col].isna().mean() * 100, 2)
        col_values = df[col].apply(lambda x: str(x) if isinstance(x, (list, dict)) else x)
        unique_vals = col_values.nunique()
        sample_vals = col_values.dropna().unique()[:5]
        sample_display = ", ".join(map(str, sample_vals))
        profile.append([col, dtype, missing_pct, unique_vals, sample_display])
    return pd.DataFrame(profile, columns=["Column", "Type", "Missing %", "Unique Values", "Sample Values"])


def safe_display(df):
    df_display = df.copy()
    
    for col in df_display.columns:
        df_display[col] = df_display[col].apply(lambda x: (
            json.dumps(x) if isinstance(x, (list, dict, np.ndarray)) else  # convert list/dict/array to JSON string
            "" if pd.isna(x) else  # replace NaN with empty string
            str(x)  # convert other types to string
        ))
    
    st.dataframe(df_display)

def export_downloads(df):
    export_df = df.copy()

    # Convert all object columns to string (optional)
    for col in export_df.columns:
        if export_df[col].dtype == "object":
            export_df[col] = export_df[col].astype(str)

    # Make all datetime columns timezone-naive for Excel
    for col in export_df.select_dtypes(include=["datetime64[ns, UTC]", "datetimetz"]).columns:
        export_df[col] = export_df[col].dt.tz_localize(None)

    # CSV download
    buffer_csv = io.BytesIO()
    export_df.to_csv(buffer_csv, index=False)
    st.download_button("⬇️ Download CSV", buffer_csv.getvalue(), 
                       file_name="cleaned_data.csv", mime="text/csv")

    # Excel download
    buffer_excel = io.BytesIO()
    with pd.ExcelWriter(buffer_excel, engine="xlsxwriter") as writer:
        export_df.to_excel(writer, index=False)
    st.download_button("⬇️ Download Excel", buffer_excel.getvalue(),
                       file_name="cleaned_data.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # JSON download
    buffer_json = io.BytesIO()
    buffer_json.write(export_df.to_json(orient="records").encode())
    st.download_button("⬇️ Download JSON", buffer_json.getvalue(), 
                       file_name="cleaned_data.json", mime="application/json")


# ---------- Dashboard ----------
def show_dashboard_charts(df):
    st.subheader("📊 Data Profiling Dashboard (3×3)")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    plots_done = 0
    if len(numeric_cols) > 0:
        sns.histplot(df[numeric_cols[0]].dropna(), kde=True, ax=axes[plots_done])
        axes[plots_done].set_title(f"Distribution: {numeric_cols[0]}")
        plots_done += 1
    if len(numeric_cols) > 1:
        sns.boxplot(y=df[numeric_cols[1]].dropna(), ax=axes[plots_done])
        axes[plots_done].set_title(f"Boxplot: {numeric_cols[1]}")
        plots_done += 1
    if len(numeric_cols) >= 2:
        sns.heatmap(df[numeric_cols].corr(), annot=False, cmap="coolwarm", ax=axes[plots_done])
        axes[plots_done].set_title("Correlation Heatmap")
        plots_done += 1
    if len(cat_cols) > 0:
        sns.countplot(y=df[cat_cols[0]], ax=axes[plots_done],
                      order=df[cat_cols[0]].value_counts().index[:10])
        axes[plots_done].set_title(f"Top 10 Categories: {cat_cols[0]}")
        plots_done += 1
    if len(cat_cols) > 0:
        df[cat_cols[0]].value_counts().head(5).plot.pie(autopct="%1.1f%%", ax=axes[plots_done])
        axes[plots_done].set_ylabel("")
        axes[plots_done].set_title(f"Category Distribution: {cat_cols[0]}")
        plots_done += 1
    if len(numeric_cols) >= 2:
        sns.scatterplot(x=df[numeric_cols[0]], y=df[numeric_cols[1]], ax=axes[plots_done])
        axes[plots_done].set_title(f"{numeric_cols[0]} vs {numeric_cols[1]}")
        plots_done += 1
    if len(numeric_cols) > 0:
        df[numeric_cols[0]].dropna().reset_index(drop=True).plot(ax=axes[plots_done])
        axes[plots_done].set_title(f"Trend: {numeric_cols[0]}")
        plots_done += 1
    if len(numeric_cols) > 0 and len(cat_cols) > 0:
        df.groupby(cat_cols[0])[numeric_cols[0]].mean().nlargest(10).plot(kind="bar", ax=axes[plots_done])
        axes[plots_done].set_title(f"Mean {numeric_cols[0]} by {cat_cols[0]}")
        plots_done += 1
    sns.heatmap(df.isnull(), cbar=False, ax=axes[plots_done])
    axes[plots_done].set_title("Missing Values Heatmap")
    plots_done += 1
    for i in range(plots_done, 9):
        axes[i].axis("off")
    plt.tight_layout()
    st.pyplot(fig)

# ---------- Main App ----------
uploaded_file = st.file_uploader("📂 Upload your data file (CSV, Excel, JSON, TXT)",
                                 type=["csv", "xlsx", "xls", "json", "txt"])

if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        st.subheader("📊 Raw Data Preview")
        safe_display(df.head())


        st.subheader("🧹 Cleaned Data")
        cleaned_df = auto_clean(df)
        st.dataframe(cleaned_df.head())
        total_rows = df.shape[0]
        cleaned_rows = cleaned_df.shape[0]
        total_cols = cleaned_df.shape[1]
        duplicates_removed = total_rows - cleaned_rows
        total_nulls = cleaned_df.isnull().sum().sum()
        percent_nulls = round((total_nulls / (cleaned_rows * total_cols)) * 100, 2) if cleaned_rows > 0 else 0
        # ---------- Custom KPI Boxes ----------
        st.subheader("📈 Data Health Report")
        kpi_html = f"""
        <div style="display:flex; gap:15px; margin-bottom:15px; flex-wrap: wrap;">
            <div style="flex:1; background:white; padding:20px; border:1px solid black; border-radius:10px; text-align:center;">
                <div style="font-size:24px; font-weight:bold;">{cleaned_rows}</div>
                <div style="font-size:14px;">Rows After Cleaning</div>
            </div>
            <div style="flex:1; background:white; padding:20px; border:1px solid black; border-radius:10px; text-align:center;">
                <div style="font-size:24px; font-weight:bold;">{total_cols}</div>
                <div style="font-size:14px;">Columns</div>
            </div>
            <div style="flex:1; background:white; padding:20px; border:1px solid black; border-radius:10px; text-align:center;">
                <div style="font-size:24px; font-weight:bold;">{duplicates_removed}</div>
                <div style="font-size:14px;">Duplicates Removed</div>
            </div>
            <div style="flex:1; background:white; padding:20px; border:1px solid black; border-radius:10px; text-align:center;">
                <div style="font-size:24px; font-weight:bold;">{total_nulls}</div>
                <div style="font-size:14px;">Total Null Values</div>
            </div>
            <div style="flex:1; background:white; padding:20px; border:1px solid black; border-radius:10px; text-align:center;">
                <div style="font-size:24px; font-weight:bold;">{percent_nulls}%</div>
                <div style="font-size:14px;">Null Percentage</div>
            </div>
        </div>
        """
        st.markdown(kpi_html, unsafe_allow_html=True)

        st.subheader("🔍 Column Profiling Report")
        profile_df = profile_data(cleaned_df)
        st.dataframe(profile_df)

        show_dashboard_charts(cleaned_df)

        st.subheader("💾 Download Cleaned Data")
        export_downloads(cleaned_df)
