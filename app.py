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
    return False

# --- Main Processing Logic ---
def process_data(orders_file, same_month_file, next_month_file, cost_file, packaging_cost_value, misc_cost_value):
    try:
        # --- A. Read Orders ---
        df_orders = pd.read_csv(orders_file)
        
        # Capture the original status from the first column of the Orders CSV
        df_orders_status_orig = df_orders.iloc[:, [0, 1]].copy()
        df_orders_status_orig.columns = ['orig_status', 'Sub Order No']
        df_orders_status_orig['Sub Order No'] = df_orders_status_orig['Sub Order No'].astype(str).str.strip()

        # --- B. Read Order Payments ---
        excel_cols = ["Sub Order No", "Live Order Status", "Final Settlement Amount"]
        # Standard Meesho columns: A=Sub Order, H=Status, N=Settlement
        df_same = pd.read_excel(same_month_file, sheet_name='Order Payments', header=1, usecols='A,H,N')
        df_next = pd.read_excel(next_month_file, sheet_name='Order Payments', header=1, usecols='A,H,N')
        df_same.columns = excel_cols
        df_next.columns = excel_cols

        # --- C. Read Cost File ---
        if cost_file.name.endswith('.csv'):
            df_cost = pd.read_csv(cost_file)
        else:
            df_cost = pd.read_excel(cost_file)

        # --- D. Read Ads Cost ---
        same_month_file.seek(0)
        try:
            df_same_ads = pd.read_excel(same_month_file, sheet_name='Ads Cost', header=1, usecols="H")
            same_ads_sum = pd.to_numeric(df_same_ads.iloc[:, 0], errors='coerce').sum()
        except:
            same_ads_sum = 0
            
        next_month_file.seek(0)
        try:
            df_next_ads = pd.read_excel(next_month_file, sheet_name='Ads Cost', header=1, usecols="H")
            next_ads_sum = pd.to_numeric(df_next_ads.iloc[:, 0], errors='coerce').sum()
        except:
            next_ads_sum = 0

    except Exception as e:
        st.error(f"Error reading files: {e}")
        return None, None, None

    # --- Data Cleaning ---
    df_orders_raw = df_orders[["Sub Order No", "SKU", "Quantity"]].copy()
    df_orders_raw['Sub Order No'] = df_orders_raw['Sub Order No'].astype(str).str.strip()
    df_orders_raw['SKU'] = df_orders_raw['SKU'].astype(str).str.strip()
    df_orders_raw['Quantity'] = pd.to_numeric(df_orders_raw['Quantity'], errors='coerce').fillna(0)

    # Prepare Payments for Pivot
    df_all_payments = pd.concat([df_same, df_next], ignore_index=True)
    df_all_payments['Sub Order No'] = df_all_payments['Sub Order No'].astype(str).str.strip()
    df_all_payments['Final Settlement Amount'] = pd.to_numeric(df_all_payments['Final Settlement Amount'], errors='coerce').fillna(0)

    df_pivot_same = df_same.copy()
    df_pivot_same['Sub Order No'] = df_pivot_same['Sub Order No'].astype(str).str.strip()
    df_pivot_same = df_pivot_same.groupby('Sub Order No')['Final Settlement Amount'].sum().reset_index()
    df_pivot_same.rename(columns={'Final Settlement Amount': 'same month pay'}, inplace=True)

    df_pivot_next = df_next.copy()
    df_pivot_next['Sub Order No'] = df_pivot_next['Sub Order No'].astype(str).str.strip()
    df_pivot_next = df_pivot_next.groupby('Sub Order No')['Final Settlement Amount'].sum().reset_index()
    df_pivot_next.rename(columns={'Final Settlement Amount': 'next month pay'}, inplace=True)

    # --- Merging ---
    df_orders_final = df_orders_raw.copy()
    df_orders_final = pd.merge(df_orders_final, df_pivot_same, on='Sub Order No', how='left')
    df_orders_final = pd.merge(df_orders_final, df_pivot_next, on='Sub Order No', how='left')
    df_orders_final['total'] = df_orders_final[['same month pay', 'next month pay']].sum(axis=1)

    # Get Status from Payments and backfill from Orders file
    payment_status = df_all_payments[['Sub Order No', 'Live Order Status']].drop_duplicates('Sub Order No', keep='last')
    df_orders_final = pd.merge(df_orders_final, payment_status, on='Sub Order No', how='left')
    df_orders_final = pd.merge(df_orders_final, df_orders_status_orig, on='Sub Order No', how='left')

    # Reconcile Status (Force Uppercase for matching)
    df_orders_final['status'] = df_orders_final['Live Order Status'].fillna(df_orders_final['orig_status']).fillna('UNKNOWN')
    df_orders_final['status'] = df_orders_final['status'].astype(str).str.upper().str.strip()

    # Define logic sets based on your CSV status strings
    DELIVERED_SET = ['DELIVERED', 'EXCHANGE', 'DOOR_STEP_EXCHANGED']
    PACKAGING_SET = ['DELIVERED', 'EXCHANGE', 'DOOR_STEP_EXCHANGED', 'RETURN', 'RTO', 'RTO_COMPLETE', 'RTO_LOCKED']

    # --- COST LOGIC ---
    cost_lookup = df_cost.iloc[:, :2].copy()
    cost_lookup.columns = ['SKU_Lookup', 'Cost_Value']
    cost_lookup['SKU_Lookup'] = cost_lookup['SKU_Lookup'].astype(str).str.strip()
    
    df_orders_final = pd.merge(df_orders_final, cost_lookup, left_on='SKU', right_on='SKU_Lookup', how='left')
    missing_cost_mask = df_orders_final['Cost_Value'].isna()
    df_orders_final['Cost_Value'] = df_orders_final['Cost_Value'].fillna(0)

    # 1. Product Cost
    df_orders_final['cost'] = np.where(df_orders_final['status'].isin(DELIVERED_SET), df_orders_final['Cost_Value'], 0)
    df_orders_final['actual cost'] = df_orders_final['cost'] * df_orders_final['Quantity']

    # 2. Packaging Cost
    df_orders_final['packaging cost'] = np.where(df_orders_final['status'].isin(PACKAGING_SET), packaging_cost_value, 0)

    # --- Stats ---
    total_payment_sum = df_orders_final['total'].sum()
    total_actual_cost_sum = df_orders_final['actual cost'].sum()
    total_packaging_sum = df_orders_final['packaging cost'].sum()
    
    # Using abs() for ads as Meesho exports them as negative
    profit_loss_value = total_payment_sum - total_actual_cost_sum - total_packaging_sum - abs(same_ads_sum) - misc_cost_value

    stats = {
        "Total Payments": total_payment_sum,
        "Total Actual Cost": total_actual_cost_sum,
        "Total Packaging Cost": total_packaging_sum,
        "Same Month Ads Cost": same_ads_sum,
        "Profit / Loss": profit_loss_value,
        "count_total": len(df_orders_final),
        "count_delivered": len(df_orders_final[df_orders_final['status'].isin(DELIVERED_SET)]),
        "count_return": len(df_orders_final[df_orders_final['status'] == 'RETURN']),
    }

    # Prep Download
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_orders_final.to_excel(writer, sheet_name='Orders_Detail', index=False)
        pd.DataFrame(list(stats.items())).to_excel(writer, sheet_name='Summary', index=False)
    output.seek(0)

    return output, stats, df_orders_final[missing_cost_mask]

# --- UI ---
if check_password():
    st.title("📊 Meesho Profit/Loss Processor")
    
    col1, col2 = st.columns(2)
    with col1:
        orders_file = st.file_uploader("Upload Orders CSV", type=['csv'])
        cost_file = st.file_uploader("Upload Cost Sheet", type=['csv', 'xlsx'])
    with col2:
        same_month = st.file_uploader("Same Month Payment (Excel)", type=['xlsx'])
        next_month = st.file_uploader("Next Month Payment (Excel)", type=['xlsx'])

    p_cost = st.number_input("Packaging Cost", value=5.0)
    m_cost = st.number_input("Misc Cost", value=0.0)

    if all([orders_file, cost_file, same_month, next_month]):
        if st.button("Generate Report", type="primary"):
            out, stats, missing = process_data(orders_file, same_month, next_month, cost_file, p_cost, m_cost)
            if out:
                st.metric("Profit / Loss", f"₹{stats['Profit / Loss']:,.2f}")
                st.download_button("Download Excel Report", out, "Meesho_Report.xlsx")
                if not missing.empty:
                    st.warning(f"{len(missing)} SKUs were missing from cost sheet.")
                    st.dataframe(missing[['Sub Order No', 'SKU']])
