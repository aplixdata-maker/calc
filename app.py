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
        # Capture status from first column of Orders CSV as backup
        df_orders_backup_status = df_orders.iloc[:, [0, 1]].copy()
        df_orders_backup_status.columns = ['backup_status', 'Sub Order No']
        df_orders_backup_status['Sub Order No'] = df_orders_backup_status['Sub Order No'].astype(str).str.strip()

        # --- B. Read Order Payments ---
        excel_cols = ["Sub Order No", "Live Order Status", "Final Settlement Amount"]
        # Standard Meesho indices: A=Sub Order, H=Status, N=Settlement
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

    # --- Data Cleaning & Merging ---
    df_orders_raw = df_orders[["Sub Order No", "SKU", "Quantity"]].copy()
    df_orders_raw['Sub Order No'] = df_orders_raw['Sub Order No'].astype(str).str.strip()
    df_orders_raw['SKU'] = df_orders_raw['SKU'].astype(str).str.strip()
    df_orders_raw['Quantity'] = pd.to_numeric(df_orders_raw['Quantity'], errors='coerce').fillna(0)

    # Prepare Payments
    df_all_pay = pd.concat([df_same, df_next], ignore_index=True)
    df_all_pay['Sub Order No'] = df_all_pay['Sub Order No'].astype(str).str.strip()
    
    df_pivot_same = df_same.groupby('Sub Order No')['Final Settlement Amount'].sum().reset_index()
    df_pivot_same.rename(columns={'Final Settlement Amount': 'same month pay'}, inplace=True)
    
    df_pivot_next = df_next.groupby('Sub Order No')['Final Settlement Amount'].sum().reset_index()
    df_pivot_next.rename(columns={'Final Settlement Amount': 'next month pay'}, inplace=True)

    # Merge Logic
    df_orders_final = pd.merge(df_orders_raw, df_pivot_same, on='Sub Order No', how='left')
    df_orders_final = pd.merge(df_orders_final, df_pivot_next, on='Sub Order No', how='left')
    
    # Fix TypeError by forcing numeric before sum
    df_orders_final['same month pay'] = pd.to_numeric(df_orders_final['same month pay'], errors='coerce').fillna(0)
    df_orders_final['next month pay'] = pd.to_numeric(df_orders_final['next month pay'], errors='coerce').fillna(0)
    df_orders_final['total'] = df_orders_final['same month pay'] + df_orders_final['next month pay']

    # Reconcile Status (Force UPPERCASE)
    status_lookup = df_all_pay[['Sub Order No', 'Live Order Status']].drop_duplicates('Sub Order No', keep='last')
    df_orders_final = pd.merge(df_orders_final, status_lookup, on='Sub Order No', how='left')
    df_orders_final = pd.merge(df_orders_final, df_orders_backup_status, on='Sub Order No', how='left')
    
    # Priority: Payment Status -> Orders File Status -> UNKNOWN
    df_orders_final['status'] = df_orders_final['Live Order Status'].fillna(df_orders_final['backup_status']).fillna('UNKNOWN')
    df_orders_final['status'] = df_orders_final['status'].astype(str).str.upper().str.strip()

    # --- Cost Calculation ---
    cost_lookup = df_cost.iloc[:, :2].copy()
    cost_lookup.columns = ['SKU_Lookup', 'Cost_Value']
    cost_lookup['SKU_Lookup'] = cost_lookup['SKU_Lookup'].astype(str).str.strip()
    
    df_orders_final = pd.merge(df_orders_final, cost_lookup, left_on='SKU', right_on='SKU_Lookup', how='left')
    missing_cost_mask = df_orders_final['Cost_Value'].isna()
    df_orders_final['Cost_Value'] = df_orders_final['Cost_Value'].fillna(0)

    # Define logic sets based on observed status in files
    DELIVERED_SET = ['DELIVERED', 'EXCHANGE', 'DOOR_STEP_EXCHANGED']
    PACKAGING_SET = DELIVERED_SET + ['RETURN', 'RTO', 'RTO_COMPLETE', 'RTO_LOCKED']

    df_orders_final['cost'] = np.where(df_orders_final['status'].isin(DELIVERED_SET), df_orders_final['Cost_Value'], 0)
    df_orders_final['actual cost'] = df_orders_final['cost'] * df_orders_final['Quantity']
    df_orders_final['packaging cost'] = np.where(df_orders_final['status'].isin(PACKAGING_SET), packaging_cost_value, 0)

    # --- Stats ---
    total_payment_sum = df_orders_final['total'].sum()
    total_actual_cost_sum = df_orders_final['actual cost'].sum()
    total_packaging_sum = df_orders_final['packaging cost'].sum()
    
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
        "count_rto": len(df_orders_final[df_orders_final['status'].str.contains('RTO')]),
    }

    # Final File Prep
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_orders_final.to_excel(writer, sheet_name='Orders_Summary', index=False)
        pd.DataFrame(list(stats.items())).to_excel(writer, sheet_name='Final_Summary', index=False)
    output.seek(0)

    return output, stats, df_orders_final[missing_cost_mask]

# --- UI ---
if check_password():
    st.title("📊 Dashboard Data Processor")
    
    col_left, col_right = st.columns(2)
    with col_left:
        orders_file = st.file_uploader("1. Upload orders file", type=['csv'])
        cost_file = st.file_uploader("2. Upload cost file", type=['csv', 'xlsx'])
    with col_right:
        same_month_file = st.file_uploader("3. Upload same month file", type=['xlsx'])
        next_month_file = st.file_uploader("4. Upload next month file", type=['xlsx'])

    p_cost = st.number_input("Packaging Cost", value=5.0)
    m_cost = st.number_input("Misc Cost", value=0.0)

    if all([orders_file, same_month_file, next_month_file, cost_file]):
        if st.button("🚀 Process Data", type="primary"):
            with st.spinner("Processing..."):
                excel_data, stats, missing = process_data(orders_file, same_month_file, next_month_file, cost_file, p_cost, m_cost)
                if excel_data:
                    st.success("Complete!")
                    st.metric("PROFIT / LOSS", f"₹{stats['Profit / Loss']:,.2f}")
                    st.download_button("⬇️ Download Excel Report", data=excel_data, file_name="Final_Report.xlsx", type="primary")
                    if not missing.empty:
                        st.warning(f"{len(missing)} SKUs missing from cost sheet.")
                        st.dataframe(missing[['Sub Order No', 'SKU']])
                    st.balloons()
