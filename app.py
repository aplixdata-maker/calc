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
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    if st.session_state["password_correct"]:
        return True

    st.markdown("### 🔒 Please Login")
    with st.form("credentials_form"):
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        if st.form_submit_button("Log In"):
            if st.session_state["username"] in st.secrets["passwords"] and \
               st.session_state["password"] == st.secrets["passwords"][st.session_state["username"]]:
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("😕 User not known or password incorrect")
    return False

# --- Main Processing Logic ---
def process_data(orders_file, same_month_file, next_month_file, cost_file, packaging_cost_value, misc_cost_value):
    try:
        # --- A. Read Orders ---
        df_orders = pd.read_csv(orders_file)

        # --- B. Read Order Payments (Fixed Columns) ---
        excel_cols = ["Sub Order No", "Live Order Status", "Final Settlement Amount"]
        # usecols based on your requirement: A (Sub Order), F (Status), L (Settlement)
        df_same_raw = pd.read_excel(same_month_file, sheet_name='Order Payments', header=1, usecols='A,F,L')
        df_next_raw = pd.read_excel(next_month_file, sheet_name='Order Payments', header=1, usecols='A,F,L')
        
        df_same_raw.columns = excel_cols
        df_next_raw.columns = excel_cols

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
        except: same_ads_sum = 0
            
        try:
            df_next_ads = pd.read_excel(next_month_file, sheet_name='Ads Cost', usecols="H")
            next_ads_sum = pd.to_numeric(df_next_ads.iloc[:, 0], errors='coerce').sum()
        except: next_ads_sum = 0

    except Exception as e:
        st.error(f"Error reading files: {e}")
        return None, None

    # --- Data Processing ---
    df_orders_raw = df_orders[["Sub Order No", "SKU", "Quantity"]].copy()
    df_orders_raw['Quantity'] = pd.to_numeric(df_orders_raw['Quantity'], errors='coerce').fillna(0)

    # Prepare status pool
    df_order_status = pd.concat([df_same_raw, df_next_raw], ignore_index=True)
    
    # Pivot Payments
    def get_pivot(df, col_name):
        df['Final Settlement Amount'] = pd.to_numeric(df['Final Settlement Amount'], errors='coerce').fillna(0)
        pivot = pd.pivot_table(df, values='Final Settlement Amount', index=['Sub Order No'], aggfunc='sum').reset_index()
        pivot.rename(columns={'Final Settlement Amount': col_name}, inplace=True)
        return pivot

    df_pivot_same = get_pivot(df_same_raw.copy(), 'same month pay')
    df_pivot_next = get_pivot(df_next_raw.copy(), 'next month pay')

    # --- Merging Data ---
    df_orders_final = pd.merge(df_orders_raw, df_pivot_same, on='Sub Order No', how='left')
    df_orders_final = pd.merge(df_orders_final, df_pivot_next, on='Sub Order No', how='left')
    df_orders_final['total'] = df_orders_final[['same month pay', 'next month pay']].sum(axis=1)
    
    status_lookup = df_order_status[['Sub Order No', 'Live Order Status']].drop_duplicates(subset=['Sub Order No'], keep='first')
    df_orders_final = pd.merge(df_orders_final, status_lookup, on='Sub Order No', how='left')
    df_orders_final.rename(columns={'Live Order Status': 'status'}, inplace=True)

    # --- COST LOGIC ---
    cost_lookup = df_cost.iloc[:, :2].copy()
    cost_lookup.columns = ['SKU_Lookup', 'Cost_Value']
    df_orders_final['SKU'] = df_orders_final['SKU'].astype(str)
    cost_lookup['SKU_Lookup'] = cost_lookup['SKU_Lookup'].astype(str)
    df_orders_final = pd.merge(df_orders_final, cost_lookup, left_on='SKU', right_on='SKU_Lookup', how='left')

    clean_status = df_orders_final['status'].fillna('Unknown').str.strip()
    cond_delivered = clean_status.isin(['Delivered', 'Exchange'])
    
    df_orders_final['cost'] = np.where(cond_delivered, df_orders_final['Cost_Value'], 0)
    df_orders_final['cost'] = df_orders_final['cost'].fillna(0)
    df_orders_final['actual cost'] = df_orders_final['cost'] * df_orders_final['Quantity']
    df_orders_final['packaging cost'] = packaging_cost_value

    # --- Final Stats ---
    total_payment_sum = df_orders_final['total'].sum()
    total_actual_cost_sum = df_orders_final['actual cost'].sum()
    total_packaging_sum = len(df_orders_final) * packaging_cost_value # As per your logic
    
    stats = {
        "Total Payments": total_payment_sum,
        "Total Actual Cost": total_actual_cost_sum,
        "Total Packaging Cost": total_packaging_sum,
        "Same Month Ads Cost": same_ads_sum,
        "Next Month Ads Cost": next_ads_sum,
        "Miscellaneous Cost": misc_cost_value,
        "Profit / Loss": total_payment_sum - total_actual_cost_sum - total_packaging_sum - abs(same_ads_sum) - misc_cost_value,
        "count_total": len(df_orders_final),
        "count_delivered": len(df_orders_final[clean_status == 'Delivered']),
        "count_return": len(df_orders_final[clean_status == 'Return']),
        "count_rto": len(df_orders_final[clean_status == 'RTO']),
        "count_Exchange": len(df_orders_final[clean_status == 'Exchange']),
        "count_cancelled": len(df_orders_final[clean_status == 'Cancelled']),
        "count_Shipped": len(df_orders_final[clean_status == 'Shipped']),
        "count_ready_to_ship": len(df_orders_final[clean_status == 'Ready_to_ship'])
    }

    # --- Write to Excel (Multiple Sheets) ---
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Sheet 1: orders.csv
        df_orders_final.drop(columns=['SKU_Lookup', 'Cost_Value'], errors='ignore').to_excel(writer, sheet_name='orders.csv', index=False)
        
        # Sheet 2: final sheet (Summary)
        pd.DataFrame(list(stats.items()), columns=['Metric', 'Value']).to_excel(writer, sheet_name='final sheet', index=False)
        
        # Sheet 3: Packaging Details
        pkg_filter = clean_status.isin(['Delivered', 'Return', 'Exchange'])
        df_pkg = df_orders_final[pkg_filter][['Sub Order No', 'SKU', 'status', 'packaging cost']].copy()
        pkg_sum = df_pkg['packaging cost'].sum()
        total_row = pd.DataFrame([['', '', 'packaging cost(Deliver, Return & Exchange)', pkg_sum]], columns=df_pkg.columns)
        pd.concat([df_pkg, total_row], ignore_index=True).to_excel(writer, sheet_name='Packaging Details', index=False)

        # Sheet 4 & 5: Payment Data
        df_same_raw.to_excel(writer, sheet_name='same month', index=False)
        df_next_raw.to_excel(writer, sheet_name='next month', index=False)

    output.seek(0)
    return output, stats

# --- App Interface ---
if check_password():
    st.title("📊 Dashboard Data Processor")

    if "results" not in st.session_state:
        st.session_state.results = None

    st.markdown("### 1. Upload & Settings")
    col_left, col_right = st.columns(2)
    with col_left:
        orders_file = st.file_uploader("1. Upload orders.csv", type=['csv'])
        cost_file = st.file_uploader("2. Upload cost file", type=['csv', 'xlsx'])
    with col_right:
        same_month_file = st.file_uploader("3. Upload same month payment", type=['xlsx'])
        next_month_file = st.file_uploader("4. Upload next month payment", type=['xlsx'])

    c_set1, c_set2 = st.columns(2)
    pack_cost = c_set1.number_input("Packaging Cost", value=5.0)
    misc_cost = c_set2.number_input("Misc Cost", value=0.0)

    if orders_file and cost_file and same_month_file and next_month_file:
        if st.button("🚀 Process Data and Generate Report", type="primary"):
            with st.spinner("Processing data..."):
                excel_data, stats = process_data(orders_file, same_month_file, next_month_file, cost_file, pack_cost, misc_cost)
                st.session_state.results = {"data": excel_data, "stats": stats}

    if st.session_state.results:
        res = st.session_state.results
        stats = res["stats"]

        st.success("✅ Processing Complete!")
        st.markdown("### 📈 Financial Summary")
        st.metric("PROFIT / LOSS", f"₹{stats['Profit / Loss']:,.2f}")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Payments", f"₹{stats['Total Payments']:,.2f}")
        col2.metric("Actual Cost", f"₹{stats['Total Actual Cost']:,.2f}")
        col3.metric("Packaging", f"₹{stats['Total Packaging Cost']:,.2f}")
        col4.metric("Ads (Same Month)", f"₹{stats['Same Month Ads Cost']:,.2f}")
        
        st.divider()
        st.markdown("### 📦 Order Status Breakdown")
        c = st.columns(8)
        c[0].metric("Total", stats['count_total'])
        c[1].metric("Delivered", stats['count_delivered'])
        c[2].metric("Return", stats['count_return'])
        c[3].metric("RTO", stats['count_rto'])
        c[4].metric("Exchange", stats['count_Exchange'])
        c[5].metric("Cancelled", stats['count_cancelled'])
        c[6].metric("Shipped", stats['count_Shipped'])
        c[7].metric("Ready", stats['count_ready_to_ship'])

        st.download_button("⬇️ Download Excel Report", data=res["data"], file_name="Final_Report.xlsx", type="primary", use_container_width=True)
