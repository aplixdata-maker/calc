import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# --- Configuration ---
st.set_page_config(
    page_title="Meesho Profit/loss calculator",
    layout="wide",import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# --- Configuration ---
st.set_page_config(
    page_title="Meesho Profit/loss calculator",
    layout="wide",
    initial_sidebar_state="expanded"
)

def process_data(orders_file, same_month_file, next_month_file, cost_file, packaging_cost_value, misc_cost_value):
    try:
        # --- A. Read Orders ---
        df_orders = pd.read_csv(orders_file)

        # --- B. Read Order Payments ---
        excel_cols = ["Sub Order No", "Live Order Status", "Final Settlement Amount"]
        
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
    
    status_lookup = df_order_status[['Sub Order No', 'Live Order Status']].drop_duplicates(subset=['Sub Order No'], keep='first')
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

    condition = df_orders_final['status'].str.strip().isin(['Delivered', 'Exchange'])
    df_orders_final['cost'] = np.where(condition, df_orders_final['Cost_Value'], 0)
    df_orders_final['cost'] = df_orders_final['cost'].fillna(0)
    df_orders_final['actual cost'] = df_orders_final['cost'] * df_orders_final['Quantity']
    df_orders_final['packaging cost'] = packaging_cost_value
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
        
        # --- NEW: Create Packaging Details Sheet ---
        # Filter for specific statuses
        pkg_filter = df_orders_final['status'].str.strip().isin(['Delivered', 'Return', 'Exchange'])
        df_pkg = df_orders_final[pkg_filter][['Sub Order No', 'SKU', 'status', 'packaging cost']].copy()
        
        # Calculate total for the footer
        pkg_sum = df_pkg['packaging cost'].sum()
        
        # Append a total row
        total_row = pd.DataFrame([['', '', 'packaging cost(Deliver, Return & Exchange)', pkg_sum]], 
                                 columns=['Sub Order No', 'SKU', 'status', 'packaging cost'])
        df_pkg_final = pd.concat([df_pkg, total_row], ignore_index=True)
        
        # Write the new sheet
        df_pkg_final.to_excel(writer, sheet_name='Packaging Details', index=False)
        # --------------------------------------------

        df_same_sheet.to_excel(writer, sheet_name='same month', index=False)
        df_next_sheet.to_excel(writer, sheet_name='next month', index=False)

    output.seek(0)
    return output, stats

# --- Streamlit App Interface ---
st.title("📊 Dashboard Data Processor")
results_container = st.container()

st.markdown("### 1. Upload & Settings")
col_left, col_right = st.columns(2)
with col_left:
    orders_file = st.file_uploader("1. Upload orders file `orders.csv`", type=['csv'])
    cost_file = st.file_uploader("2. Upload `cost` file", type=['csv', 'xlsx'])
with col_right:
    same_month_file = st.file_uploader("3. Upload same month payment file `same.xlsx`", type=['xlsx'])
    next_month_file = st.file_uploader("4. Upload Next month payment file `next.xlsx`", type=['xlsx'])

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
