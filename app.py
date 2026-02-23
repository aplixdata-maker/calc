import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# --- Configuration ---
st.set_page_config(
    page_title="Meesho Profit/loss calculator",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- SECURITY: Password Authentication ---
def check_password():
    """Returns `True` if the user had a correct password."""
    def password_entered():
        if st.session_state["username"] in st.secrets["passwords"] and \
           st.session_state["password"] == st.secrets["passwords"][st.session_state["username"]]:
            st.session_state["password_correct"] = True
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if st.session_state["password_correct"]:
        return True

    st.markdown("### 🔒 Please Login")
    with st.form("credentials_form"):
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        submit_button = st.form_submit_button("Log In", on_click=password_entered)

    if submit_button and not st.session_state["password_correct"]:
        st.error("😕 User not known or password incorrect")
    return False

# --- Main Processing Logic ---
def process_data(orders_file, same_month_file, next_month_file, cost_file, packaging_cost_value, misc_cost_value):
    try:
        # --- A. Read Orders ---
        df_orders = pd.read_csv(orders_file)

        # --- B. Read Order Payments (Dynamic Column Mapping) ---
        def read_payment_sheet(file):
            df_raw = pd.read_excel(file, sheet_name='Order Payments', header=1)
            # Find columns dynamically to avoid hardcoded letter errors
            cols_to_extract = {
                "Sub Order No": "Sub Order No",
                "Live Order Status": "Live Order Status",
                "Final Settlement Amount": "Final Settlement Amount"
            }
            # Verify columns exist
            for key, val in cols_to_extract.items():
                if val not in df_raw.columns:
                    raise ValueError(f"Column '{val}' not found in sheet 'Order Payments'")
            
            return df_raw[list(cols_to_extract.values())].copy()

        df_same = read_payment_sheet(same_month_file)
        df_next = read_payment_sheet(next_month_file)

        # --- C. Read Cost File ---
        if cost_file.name.endswith('.csv'):
            df_cost = pd.read_csv(cost_file)
        else:
            df_cost = pd.read_excel(cost_file)

        # --- D. Read Ads Cost (Dynamic Column Mapping) ---
        def get_ads_sum(file):
            file.seek(0)
            try:
                df_ads = pd.read_excel(file, sheet_name='Ads Cost', header=1)
                # Find column that contains "Total Ads Cost"
                ads_col = [c for c in df_ads.columns if 'Total Ads Cost' in str(c)][0]
                return pd.to_numeric(df_ads[ads_col], errors='coerce').sum()
            except:
                return 0

        same_ads_sum = get_ads_sum(same_month_file)
        next_ads_sum = get_ads_sum(next_month_file)

    except Exception as e:
        st.error(f"Error reading files: {e}")
        return None, None, None

    # --- Data Processing ---
    df_orders_raw = df_orders[["Sub Order No", "SKU", "Quantity"]].copy()
    df_orders_raw['Quantity'] = pd.to_numeric(df_orders_raw['Quantity'], errors='coerce').fillna(0)

    # Combine payment statuses
    df_order_status_pool = pd.concat([df_same, df_next], ignore_index=True)
    
    # Prepare Pivot Tables for Payments
    def get_pivot(df, col_name):
        df['Final Settlement Amount'] = pd.to_numeric(df['Final Settlement Amount'], errors='coerce').fillna(0)
        pivot = pd.pivot_table(df, values='Final Settlement Amount', index=['Sub Order No'], aggfunc='sum').reset_index()
        pivot.rename(columns={'Final Settlement Amount': col_name}, inplace=True)
        return pivot

    df_pivot_same = get_pivot(df_same, 'same month pay')
    df_pivot_next = get_pivot(df_next, 'next month pay')

    # --- Merging Data ---
    df_orders_final = pd.merge(df_orders_raw, df_pivot_same, on='Sub Order No', how='left')
    df_orders_final = pd.merge(df_orders_final, df_pivot_next, on='Sub Order No', how='left')
    
    df_orders_final['total'] = df_orders_final[['same month pay', 'next month pay']].sum(axis=1, skipna=True)
    
    # Get Status (Last known status for the Sub Order No)
    status_lookup = df_order_status_pool[['Sub Order No', 'Live Order Status']].drop_duplicates(subset=['Sub Order No'], keep='last')
    df_orders_final = pd.merge(df_orders_final, status_lookup, on='Sub Order No', how='left')
    df_orders_final.rename(columns={'Live Order Status': 'status'}, inplace=True)

    # --- COST LOGIC ---
    cost_lookup = df_cost.iloc[:, :2].copy()
    cost_lookup.columns = ['SKU_Lookup', 'Cost_Value']
    df_orders_final['SKU'] = df_orders_final['SKU'].astype(str)
    cost_lookup['SKU_Lookup'] = cost_lookup['SKU_Lookup'].astype(str)
    
    df_orders_final = pd.merge(df_orders_final, cost_lookup, left_on='SKU', right_on='SKU_Lookup', how='left')

    # Identify Missing SKUs
    missing_cost_mask = df_orders_final['Cost_Value'].isna()
    missing_details_df = df_orders_final.loc[missing_cost_mask, ['Sub Order No', 'SKU', 'status', 'Quantity', 'total']].copy()
    
    df_orders_final['Cost_Value'] = df_orders_final['Cost_Value'].fillna(0)

    # Product and Packaging Cost Logic
    clean_status = df_orders_final['status'].fillna('Unknown').str.strip()
    cond_delivered = clean_status.isin(['Delivered', 'Exchange'])
    cond_pkg = clean_status.isin(['Delivered', 'Exchange', 'Return'])

    df_orders_final['cost'] = np.where(cond_delivered, df_orders_final['Cost_Value'], 0)
    df_orders_final['actual cost'] = df_orders_final['cost'] * df_orders_final['Quantity']
    df_orders_final['packaging cost'] = np.where(cond_pkg, packaging_cost_value, 0)

    # --- Stats Calculation ---
    stats = {
        "Total Payments": df_orders_final['total'].sum(),
        "Total Actual Cost": df_orders_final['actual cost'].sum(),
        "Total Packaging Cost": df_orders_final['packaging cost'].sum(),
        "Same Month Ads Cost": same_ads_sum,
        "Next Month Ads Cost": next_ads_sum,
        "Miscellaneous Cost": misc_cost_value,
        "Profit / Loss": df_orders_final['total'].sum() - df_orders_final['actual cost'].sum() - df_orders_final['packaging cost'].sum() - abs(same_ads_sum) - misc_cost_value,
        "count_total": len(df_orders_final),
        "count_delivered": len(df_orders_final[clean_status == 'Delivered']),
        "count_return": len(df_orders_final[clean_status == 'Return']),
        "count_rto": len(df_orders_final[clean_status == 'RTO']),
        "count_Exchange": len(df_orders_final[clean_status == 'Exchange']),
        "count_cancelled": len(df_orders_final[clean_status == 'Cancelled']),
        "count_Shipped": len(df_orders_final[clean_status == 'Shipped']),
        "count_ready_to_ship": len(df_orders_final[clean_status == 'Ready_to_ship'])
    }

    # --- Export Prep ---
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_orders_final.drop(columns=['SKU_Lookup', 'Cost_Value'], errors='ignore').to_excel(writer, sheet_name='orders', index=False)
        pd.DataFrame(list(stats.items()), columns=['Metric', 'Value']).to_excel(writer, sheet_name='summary', index=False)
        
        # Cost sheet for audit
        pkg_filter = clean_status.isin(['Delivered', 'Return', 'Exchange'])
        df_pkg = df_orders_final[pkg_filter][['Sub Order No', 'SKU', 'status', 'actual cost']].copy()
        df_pkg.to_excel(writer, sheet_name='Cost Audit', index=False)

    output.seek(0)
    return output, stats, missing_details_df

# --- Streamlit App ---
if check_password():
    st.title("📊 Meesho Profit/Loss Dashboard")
    
    with st.expander("Step 1: Upload Files", expanded=True):
        c1, c2 = st.columns(2)
        orders_file = c1.file_uploader("Orders File (CSV)", type=['csv'])
        cost_file = c1.file_uploader("Cost File (CSV/XLSX)", type=['csv', 'xlsx'])
        same_month_file = c2.file_uploader("Current Month Payment (XLSX)", type=['xlsx'])
        next_month_file = c2.file_uploader("Next Month Payment (XLSX)", type=['xlsx'])

    with st.expander("Step 2: Settings"):
        c1, c2 = st.columns(2)
        pack_cost = c1.number_input("Packaging Cost", value=5.0)
        misc_cost = c2.number_input("Misc Cost", value=0.0)

    if orders_file and cost_file and same_month_file and next_month_file:
        if st.button("Calculate", type="primary"):
            # --- Show processing icon ---
            with st.spinner('Crunching numbers... Please wait.'):
                excel_data, stats, missing = process_data(orders_file, same_month_file, next_month_file, cost_file, pack_cost, misc_cost)
            
            if stats:
                st.success("Analysis Complete!")
                st.metric("Total Profit/Loss", f"₹{stats['Profit / Loss']:,.2f}")
                
                cols = st.columns(4)
                cols[0].metric("Payments", f"₹{stats['Total Payments']:,.2f}")
                cols[1].metric("Prod Cost", f"₹{stats['Total Actual Cost']:,.2f}")
                cols[2].metric("Pkg Cost", f"₹{stats['Total Packaging Cost']:,.2f}")
                cols[3].metric("Ads", f"₹{abs(stats['Same Month Ads Cost']):,.2f}")
                
                if not missing.empty:
                    st.warning(f"Found {len(missing)} missing SKUs in cost sheet.")
                    st.dataframe(missing)
                
                st.download_button("Download Report", excel_data, "Report.xlsx")
