import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# --- Configuration ---
st.set_page_config(
    page_title="Meesho Profit/loss calculator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SECURITY: Password Authentication ---
def check_password():
    """Returns `True` if the user had a correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
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

    if submit_button:
        if not st.session_state["password_correct"]:
            st.error("😕 User not known or password incorrect")
            return False
        else:
            st.rerun() 
            
    return False

# --- Main Processing Logic ---
def process_data(orders_file, same_month_file, next_month_file, cost_file, packaging_cost_value, misc_cost_value):
    try:
        # --- A. Read Orders ---
        df_orders = pd.read_csv(orders_file)

        # --- B. Read Order Payments ---
        excel_cols = ["Sub Order No", "Live Order Status", "Final Settlement Amount"]
        
        # Note: headers are 1 (row 2) based on your logic
        df_same = pd.read_excel(same_month_file, sheet_name='Order Payments', header=1, usecols='A,F,L')
        df_next = pd.read_excel(next_month_file, sheet_name='Order Payments', header=1, usecols='A,F,L')

        # --- C. Read Cost File ---
        if cost_file.name.endswith('.csv'):
            df_cost = pd.read_csv(cost_file)
        else:
            df_cost = pd.read_excel(cost_file)

        # --- D. Read Ads Cost ---
        same_month_file.seek(0)
        next_month_file.seek(0)

        try:
            df_same_ads = pd.read_excel(same_month_file, sheet_name='Ads Cost', usecols="H")
            same_ads_sum = pd.to_numeric(df_same_ads.iloc[:, 0], errors='coerce').sum()
        except Exception:
            same_ads_sum = 0
            
        try:
            df_next_ads = pd.read_excel(next_month_file, sheet_name='Ads Cost', usecols="H")
            next_ads_sum = pd.to_numeric(df_next_ads.iloc[:, 0], errors='coerce').sum()
        except Exception:
            next_ads_sum = 0

    except Exception as e:
        st.error(f"Error reading one or more files: {e}")
        return None, None

    # --- Data Processing ---
    df_same.columns = excel_cols
    df_next.columns = excel_cols
    df_orders_raw = df_orders[["Sub Order No", "SKU", "Quantity"]].copy()
    df_orders_raw['Quantity'] = pd.to_numeric(df_orders_raw['Quantity'], errors='coerce').fillna(0)

    df_same_sheet = df_same[excel_cols].copy()
    df_next_sheet = df_next[excel_cols].copy()
    df_order_status = pd.concat([df_same_sheet, df_next_sheet], ignore_index=True)
    
    def prepare_for_pivot(df):
        df['Final Settlement Amount'] = pd.to_numeric(df['Final Settlement Amount'], errors='coerce').fillna(0)
        return df
        
    df_same_pivot_data = prepare_for_pivot(df_same_sheet.copy())
    df_next_pivot_data = prepare_for_pivot(df_next_sheet.copy())

    df_pivot_same = pd.pivot_table(df_same_pivot_data, values='Final Settlement Amount', index=['Sub Order No'], aggfunc='sum').reset_index()
    df_pivot_same.rename(columns={'Final Settlement Amount': 'same month pay'}, inplace=True)
    
    df_pivot_next = pd.pivot_table(df_next_pivot_data, values='Final Settlement Amount', index=['Sub Order No'], aggfunc='sum').reset_index()
    df_pivot_next.rename(columns={'Final Settlement Amount': 'next month pay'}, inplace=True)

    # --- Merging Data ---
    df_orders_final = df_orders_raw.copy()
    df_orders_final = pd.merge(df_orders_final, df_pivot_same[['Sub Order No', 'same month pay']], on='Sub Order No', how='left')
    df_orders_final = pd.merge(df_orders_final, df_pivot_next[['Sub Order No', 'next month pay']], on='Sub Order No', how='left')
    df_orders_final['total'] = df_orders_final[['same month pay', 'next month pay']].sum(axis=1, skipna=True)
    
    status_lookup = df_order_status[['Sub Order No', 'Live Order Status']].drop_duplicates(subset=['Sub Order No'], keep='last')
    df_orders_final = pd.merge(df_orders_final, status_lookup, on='Sub Order No', how='left')
    df_orders_final.rename(columns={'Live Order Status': 'status'}, inplace=True)

    # --- Status Counting Logic ---
    status_series = df_orders_final['status'].fillna('Unknown').str.strip()
    
    # --- COST LOGIC ---
    cost_lookup = df_cost.iloc[:, :2].copy()
    cost_lookup.columns = ['SKU_Lookup', 'Cost_Value'] 
    df_orders_final['SKU'] = df_orders_final['SKU'].astype(str)
    cost_lookup['SKU_Lookup'] = cost_lookup['SKU_Lookup'].astype(str)
    df_orders_final = pd.merge(df_orders_final, cost_lookup, left_on='SKU', right_on='SKU_Lookup', how='left')

    # 1. Product Cost Calculation (Only for Delivered and Exchange)
    condition_product = df_orders_final['status'].str.strip().isin(['Delivered', 'Exchange'])
    df_orders_final['cost'] = np.where(condition_product, df_orders_final['Cost_Value'], 0)
    df_orders_final['cost'] = df_orders_final['cost'].fillna(0)
    df_orders_final['actual cost'] = df_orders_final['cost'] * df_orders_final['Quantity']

    # 2. Packaging Cost Calculation (UPDATED: Delivered, Exchange & Return only)
    condition_packaging = df_orders_final['status'].str.strip().isin(['Delivered', 'Exchange', 'Return'])
    df_orders_final['packaging cost'] = np.where(condition_packaging, packaging_cost_value, 0)
    
    df_orders_final.drop(columns=['SKU_Lookup', 'Cost_Value'], inplace=True)

    # --- Calculate Final Stats ---
    total_payment_sum = df_orders_final['total'].sum(skipna=True)
    total_cost_sum = df_orders_final['cost'].sum(skipna=True)
    total_actual_cost_sum = df_orders_final['actual cost'].sum(skipna=True)
    total_packaging_sum = df_orders_final['packaging cost'].sum(skipna=True)
    profit_loss_value = total_payment_sum - total_actual_cost_sum - total_packaging_sum - abs(same_ads_sum) - misc_cost_value

    stats = {
        "Total Payments": total_payment_sum,
        "Total Cost": total_cost_sum,
        "Total Actual Cost": total_actual_cost_sum,
        "Total Packaging Cost": total_packaging_sum,
        "Same Month Ads Cost": same_ads_sum,
        "Next Month Ads Cost": next_ads_sum,
        "Miscellaneous Cost": misc_cost_value,
        "Profit / Loss": profit_loss_value,
        "count_total": len(df_orders_final),
        "count_delivered": len(df_orders_final[status_series == 'Delivered']),
        "count_return": len(df_orders_final[status_series == 'Return']),
        "count_rto": len(df_orders_final[status_series == 'RTO']),
        "count_Exchange": len(df_orders_final[status_series == 'Exchange']),
        "count_cancelled": len(df_orders_final[status_series == 'Cancelled']),
        "count_Shipped": len(df_orders_final[status_series == 'Shipped']),
        "count_ready_to_ship": len(df_orders_final[status_series == 'Ready_to_ship'])
    }

    # --- Write to Excel ---
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_orders_final.to_excel(writer, sheet_name='orders.csv', index=False)
        summary_df = pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])
        summary_df.to_excel(writer, sheet_name='final sheet', index=False)
        
        # ---------------------------------------------------------------------
        # Create total cost Sheet for Delivered, Return & Exchange
        # ---------------------------------------------------------------------
        pkg_filter = df_orders_final['status'].str.strip().isin(['Delivered', 'Return', 'Exchange'])
        df_pkg = df_orders_final[pkg_filter][['Sub Order No', 'SKU', 'status', 'actual cost']].copy()
        pkg_sum = df_pkg['actual cost'].sum()
        
        total_row_data = {
            'Sub Order No': '',
            'SKU': '',
            'status': 'GRAND TOTAL',
            'actual cost': pkg_sum
        }
        total_row_df = pd.DataFrame([total_row_data])
        df_pkg_final = pd.concat([df_pkg, total_row_df], ignore_index=True)
        df_pkg_final.to_excel(writer, sheet_name='Cost (Del, Ret, Exc)', index=False)
        # ---------------------------------------------------------------------

        df_same_sheet.to_excel(writer, sheet_name='same month', index=False)
        df_next_sheet.to_excel(writer, sheet_name='next month', index=False)

    output.seek(0)
    return output, stats

# --- Streamlit App Interface (GATED) ---
if check_password():
    st.title("📊 Dashboard Data Processor")
    results_container = st.container()

    st.markdown("### 1. Upload & Settings")
    col_left, col_right = st.columns(2)
    with col_left:
        orders_file = st.file_uploader("1. Upload orders file ", type=['csv'])
        cost_file = st.file_uploader("2. Upload cost file", type=['csv', 'xlsx'])
    with col_right:
        same_month_file = st.file_uploader("3. Upload same month payment file ", type=['xlsx'])
        next_month_file = st.file_uploader("4. Upload Next month payment file ", type=['xlsx'])

    col_set1, col_set2 = st.columns(2)
    with col_set1:
        pack_cost = st.number_input("Packaging Cost (per record)", value=5.0, step=0.5)
    with col_set2:
        misc_cost = st.number_input("Miscellaneous Cost", value=0.0, step=100.0)

    if orders_file and same_month_file and next_month_file and cost_file:
        if st.button("🚀 Process Data and Generate Report", type="primary"):
            with st.spinner("Processing data..."):
                excel_data, stats = process_data(orders_file, same_month_file, next_month_file, cost_file, pack_cost, misc_cost)
                
                if excel_data and stats:
                    with results_container:
                        st.success("✅ Processing Complete!")
                        
                        st.markdown("### 📈 Financial Summary")
                        pl_val = stats['Profit / Loss']
                        st.metric("PROFIT / LOSS", f"₹{pl_val:,.2f}")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total Payments", f"₹{stats['Total Payments']:,.2f}")
                        col2.metric("Actual Cost", f"₹{stats['Total Actual Cost']:,.2f}")
                        col3.metric("Packaging", f"₹{stats['Total Packaging Cost']:,.2f}")
                        col4.metric("Ads (Same Month)", f"₹{stats['Same Month Ads Cost']:,.2f}")
                        
                        st.divider()

                        st.markdown("### 📦 Order Status Breakdown")
                        c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
                        c1.metric("Total Orders", stats['count_total'])
                        c2.metric("Delivered", stats['count_delivered'])
                        c3.metric("Return", stats['count_return'])
                        c4.metric("RTO", stats['count_rto'])
                        c5.metric("Exchange", stats['count_Exchange'])
                        c6.metric("Cancelled", stats['count_cancelled'])
                        c7.metric("Shipped", stats['count_Shipped'])
                        c8.metric("Ready_to_ship", stats['count_ready_to_ship'])
                        
                        st.divider()
                        
                        st.download_button("⬇️ Download Excel Report", data=excel_data, file_name="Final_Report.xlsx", use_container_width=True, type="primary")
                    st.balloons()
