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
        # UPDATED: Standard Meesho Payment indices: A=Sub Order No, H=Status, N=Settlement
        excel_cols = ["Sub Order No", "Live Order Status", "Final Settlement Amount"]
        
        df_same = pd.read_excel(same_month_file, sheet_name='Order Payments', header=1, usecols='A,H,N')
        df_next = pd.read_excel(next_month_file, sheet_name='Order Payments', header=1, usecols='A,H,N')
        
        # Explicitly rename columns to ensure consistency
        df_same.columns = excel_cols
        df_next.columns = excel_cols

        # --- C. Read Cost File ---
        if cost_file.name.endswith('.csv'):
            df_cost = pd.read_csv(cost_file)
        else:
            df_cost = pd.read_excel(cost_file)

        # --- D. Read Ads Cost ---
        same_month_file.seek(0)
        next_month_file.seek(0)

        try:
            # H is the standard column for "Total Ads Cost"
            df_same_ads = pd.read_excel(same_month_file, sheet_name='Ads Cost', header=1, usecols="H")
            same_ads_sum = pd.to_numeric(df_same_ads.iloc[:, 0], errors='coerce').sum()
        except Exception:
            same_ads_sum = 0
            
        try:
            df_next_ads = pd.read_excel(next_month_file, sheet_name='Ads Cost', header=1, usecols="H")
            next_ads_sum = pd.to_numeric(df_next_ads.iloc[:, 0], errors='coerce').sum()
        except Exception:
            next_ads_sum = 0

    except Exception as e:
        st.error(f"Error reading one or more files: {e}")
        return None, None, None

    # --- Data Cleaning ---
    df_orders_raw = df_orders[["Sub Order No", "SKU", "Quantity"]].copy()
    df_orders_raw['Sub Order No'] = df_orders_raw['Sub Order No'].astype(str).str.strip()
    df_orders_raw['SKU'] = df_orders_raw['SKU'].astype(str).str.strip()
    df_orders_raw['Quantity'] = pd.to_numeric(df_orders_raw['Quantity'], errors='coerce').fillna(0)

    # Prepare payment sheets for merge
    df_order_status = pd.concat([df_same, df_next], ignore_index=True)
    df_order_status['Sub Order No'] = df_order_status['Sub Order No'].astype(str).str.strip()
    
    def prepare_for_pivot(df):
        df['Sub Order No'] = df['Sub Order No'].astype(str).str.strip()
        df['Final Settlement Amount'] = pd.to_numeric(df['Final Settlement Amount'], errors='coerce').fillna(0)
        return df
        
    df_same_pivot_data = prepare_for_pivot(df_same.copy())
    df_next_pivot_data = prepare_for_pivot(df_next.copy())

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

    # --- Status Cleanup ---
    df_orders_final['status'] = df_orders_final['status'].fillna('Unknown').str.strip()
    status_series = df_orders_final['status']
    
    # --- COST LOGIC ---
    cost_lookup = df_cost.iloc[:, :2].copy()
    cost_lookup.columns = ['SKU_Lookup', 'Cost_Value'] 
    cost_lookup['SKU_Lookup'] = cost_lookup['SKU_Lookup'].astype(str).str.strip()
    
    # Merge with Cost Lookup
    df_orders_final = pd.merge(df_orders_final, cost_lookup, left_on='SKU', right_on='SKU_Lookup', how='left')

    # IDENTIFY MISSING SKUS
    missing_cost_mask = df_orders_final['Cost_Value'].isna()
    missing_details_df = df_orders_final.loc[missing_cost_mask, ['Sub Order No', 'SKU', 'status', 'Quantity', 'total']].copy()
    missing_details_df.rename(columns={'total': 'Total Payment'}, inplace=True)
    
    # Fill NaN with 0 for Calculation
    df_orders_final['Cost_Value'] = df_orders_final['Cost_Value'].fillna(0)

    # 1. Product Cost Calculation (Only for Delivered and Exchange)
    condition_product = df_orders_final['status'].isin(['Delivered', 'Exchange'])
    df_orders_final['cost'] = np.where(condition_product, df_orders_final['Cost_Value'], 0)
    df_orders_final['actual cost'] = df_orders_final['cost'] * df_orders_final['Quantity']

    # 2. Packaging Cost Calculation
    condition_packaging = df_orders_final['status'].isin(['Delivered', 'Exchange', 'Return'])
    df_orders_final['packaging cost'] = np.where(condition_packaging, packaging_cost_value, 0)
    
    df_orders_final.drop(columns=['SKU_Lookup', 'Cost_Value'], inplace=True)

    # --- Calculate Final Stats ---
    total_payment_sum = df_orders_final['total'].sum(skipna=True)
    total_actual_cost_sum = df_orders_final['actual cost'].sum(skipna=True)
    total_packaging_sum = df_orders_final['packaging cost'].sum(skipna=True)
    
    # Ads are usually negative in Meesho files, using abs() to treat as a positive expense to subtract
    profit_loss_value = total_payment_sum - total_actual_cost_sum - total_packaging_sum - abs(same_ads_sum) - misc_cost_value

    stats = {
        "Total Payments": total_payment_sum,
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

    # EXPORT PREP
    df_orders_final['cost_display'] = df_orders_final['cost'].astype(object)
    condition_display_error = missing_cost_mask & condition_product
    df_orders_final.loc[condition_display_error, 'cost_display'] = "SKU Not Found"

    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_orders_final.to_excel(writer, sheet_name='orders_summary', index=False)
        summary_df = pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])
        summary_df.to_excel(writer, sheet_name='final_summary', index=False)
        
        pkg_filter = df_orders_final['status'].isin(['Delivered', 'Return', 'Exchange'])
        df_pkg = df_orders_final[pkg_filter][['Sub Order No', 'SKU', 'status', 'actual cost']].copy()
        pkg_sum = pd.to_numeric(df_pkg['actual cost'], errors='coerce').sum()
        
        total_row_df = pd.DataFrame([{'Sub Order No': '', 'SKU': '', 'status': 'GRAND TOTAL', 'actual cost': pkg_sum}])
        pd.concat([df_pkg, total_row_df], ignore_index=True).to_excel(writer, sheet_name='Cost_Analysis', index=False)

    output.seek(0)
    return output, stats, missing_details_df

# --- Streamlit App Interface (GATED) ---
if check_password():
    st.title("📊 Dashboard Data Processor")
    results_container = st.container()

    st.markdown("### 1. Upload & Settings")
    col_left, col_right = st.columns(2)
    with col_left:
        orders_file = st.file_uploader("1. Upload orders file", type=['csv'])
        cost_file = st.file_uploader("2. Upload cost file", type=['csv', 'xlsx'])
    with col_right:
        same_month_file = st.file_uploader("3. Upload same month payment file", type=['xlsx'])
        next_month_file = st.file_uploader("4. Upload Next month payment file", type=['xlsx'])

    col_set1, col_set2 = st.columns(2)
    with col_set1:
        pack_cost = st.number_input("Packaging Cost (per record)", value=5.0, step=0.5)
    with col_set2:
        misc_cost = st.number_input("Miscellaneous Cost", value=0.0, step=100.0)

    if orders_file and same_month_file and next_month_file and cost_file:
        if st.button("🚀 Process Data and Generate Report", type="primary"):
            with st.spinner("Processing data..."):
                excel_data, stats, missing_details = process_data(orders_file, same_month_file, next_month_file, cost_file, pack_cost, misc_cost)
                
                if excel_data and stats:
                    with results_container:
                        st.success("✅ Processing Complete!")
                        st.markdown("### 📈 Financial Summary")
                        st.metric("PROFIT / LOSS", f"₹{stats['Profit / Loss']:,.2f}")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total Payments", f"₹{stats['Total Payments']:,.2f}")
                        col2.metric("Actual Cost", f"₹{stats['Total Actual Cost']:,.2f}")
                        col3.metric("Packaging", f"₹{stats['Total Packaging Cost']:,.2f}")
                        col4.metric("Ads (Same Month)", f"₹{stats['Same Month Ads Cost']:,.2f}")
                        
                        if not missing_details.empty:
                            st.markdown("---")
                            st.error(f"⚠️ **{len(missing_details)} Orders Missing SKU Cost**")
                            st.caption("The following orders have SKUs not found in your cost sheet. Calculated as 0 cost.")
                            st.dataframe(missing_details, use_container_width=True, hide_index=True)

                        st.divider()
                        st.markdown("### 📦 Order Status Breakdown")
                        st.json(stats) # Or use metrics columns as before
                        
                        st.download_button("⬇️ Download Excel Report", data=excel_data, file_name="Final_Report.xlsx", use_container_width=True, type="primary")
                    st.balloons()
