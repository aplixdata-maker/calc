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

    if submit_button and not st.session_state["password_correct"]:
        st.error("😕 User not known or password incorrect")
    return False

# --- Main Processing Logic ---
def process_data(orders_file, same_month_file, next_month_file, cost_file, packaging_cost_value, misc_cost_value):
    try:
        # 1. Load Files
        df_orders = pd.read_csv(orders_file)
        
        # Helper to read sheets with header offset (usually row 2 in Meesho excels)
        def read_meesho_sheet(file, sheet_name, cols=None):
            try:
                file.seek(0)
                return pd.read_excel(file, sheet_name=sheet_name, header=1, usecols=cols)
            except:
                return pd.DataFrame()

        df_same = read_meesho_sheet(same_month_file, 'Order Payments', 'A,F,L')
        df_next = read_meesho_sheet(next_month_file, 'Order Payments', 'A,F,L')
        
        # Load Ads Cost
        df_same_ads = read_meesho_sheet(same_month_file, 'Ads Cost', 'H')
        df_next_ads = read_meesho_sheet(next_month_file, 'Ads Cost', 'H')
        
        # Load Recovery/Compensation (New)
        df_same_rec = read_meesho_sheet(same_month_file, 'Compensation and Recovery', 'D')
        
        # Load Cost Master
        if cost_file.name.endswith('.csv'):
            df_cost = pd.read_csv(cost_file)
        else:
            df_cost = pd.read_excel(cost_file)

    except Exception as e:
        st.error(f"Error reading files: {e}")
        return None, None, None

    # 2. Cleanup & Standardize
    excel_cols = ["Sub Order No", "Live Order Status", "Final Settlement Amount"]
    df_same.columns = excel_cols
    df_next.columns = excel_cols
    
    # Merge Settlement Data
    df_settlements = pd.concat([df_same, df_next], ignore_index=True)
    df_settlements['Final Settlement Amount'] = pd.to_numeric(df_settlements['Final Settlement Amount'], errors='coerce').fillna(0)
    
    # Pivot to handle multiple payment entries for one Sub Order
    df_pivot = df_settlements.groupby('Sub Order No').agg({
        'Final Settlement Amount': 'sum',
        'Live Order Status': 'last'
    }).reset_index()

    # 3. Join Orders with Settlements
    df_orders_raw = df_orders[["Sub Order No", "SKU", "Quantity"]].copy()
    df_orders_raw['SKU'] = df_orders_raw['SKU'].astype(str).str.strip().str.upper()
    
    df_final = pd.merge(df_orders_raw, df_pivot, on='Sub Order No', how='left')
    df_final.rename(columns={'Live Order Status': 'status', 'Final Settlement Amount': 'total_payment'}, inplace=True)
    df_final['total_payment'] = df_final['total_payment'].fillna(0)
    df_final['status'] = df_final['status'].fillna('Unknown').str.strip()

    # 4. Join with Cost Data
    cost_lookup = df_cost.iloc[:, :2].copy()
    cost_lookup.columns = ['SKU_Lookup', 'Cost_Value']
    cost_lookup['SKU_Lookup'] = cost_lookup['SKU_Lookup'].astype(str).str.strip().str.upper()
    
    df_final = pd.merge(df_final, cost_lookup, left_on='SKU', right_on='SKU_Lookup', how='left')
    
    # Identify Missing SKUs
    missing_mask = df_final['Cost_Value'].isna()
    missing_details = df_final.loc[missing_mask, ['Sub Order No', 'SKU', 'status', 'Quantity', 'total_payment']].copy()
    
    # 5. Financial Calculations
    # Product Cost only for Delivered/Exchange
    df_final['product_cost'] = np.where(
        df_final['status'].isin(['Delivered', 'Exchange']), 
        df_final['Cost_Value'].fillna(0) * df_final['Quantity'], 
        0
    )
    
    # Packaging for all shipped items (Delivered, Return, Exchange)
    df_final['pkg_cost'] = np.where(
        df_final['status'].isin(['Delivered', 'Return', 'Exchange']), 
        packaging_cost_value, 
        0
    )

    # Sums
    total_payment_sum = df_final['total_payment'].sum()
    total_cogs = df_final['product_cost'].sum()
    total_pkg = df_final['pkg_cost'].sum()
    
    same_ads_sum = pd.to_numeric(df_same_ads.iloc[:, 0], errors='coerce').sum() if not df_same_ads.empty else 0
    next_ads_sum = pd.to_numeric(df_next_ads.iloc[:, 0], errors='coerce').sum() if not df_next_ads.empty else 0
    
    recovery_sum = pd.to_numeric(df_same_rec.iloc[:, 0], errors='coerce').sum() if not df_same_rec.empty else 0
    
    # Profit Calculation: Settlement - COGS - Packaging - Ads - Recoveries - Misc
    # (Note: recovery_sum is usually negative in the sheet, so we add it)
    profit_loss = total_payment_sum - total_cogs - total_pkg - abs(same_ads_sum) + recovery_sum - misc_cost_value

    stats = {
        "Total Payments": total_payment_sum,
        "Total Product Cost": total_cogs,
        "Total Packaging Cost": total_pkg,
        "Same Month Ads": same_ads_sum,
        "Next Month Ads": next_ads_sum,
        "Recoveries/Charges": recovery_sum,
        "Profit / Loss": profit_loss,
        "count_total": len(df_final),
        "count_delivered": len(df_final[df_final['status'] == 'Delivered']),
        "count_return": len(df_final[df_final['status'] == 'Return']),
        "count_rto": len(df_final[df_final['status'] == 'RTO']),
        "count_Exchange": len(df_final[df_final['status'] == 'Exchange']),
        "count_cancelled": len(df_final[df_final['status'] == 'Cancelled']),
    }

    # 6. Export Preparation
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_final.to_excel(writer, sheet_name='Detailed Orders', index=False)
        pd.DataFrame(list(stats.items()), columns=['Metric', 'Value']).to_excel(writer, sheet_name='Summary')
        if not missing_details.empty:
            missing_details.to_excel(writer, sheet_name='Missing SKUs', index=False)

    output.seek(0)
    return output, stats, missing_details

# --- Streamlit App UI ---
if check_password():
    st.title("📊 Meesho Dashboard Processor")
    
    st.markdown("### 1. Upload Files")
    c1, c2 = st.columns(2)
    with c1:
        orders_file = st.file_uploader("Upload Orders CSV", type=['csv'])
        cost_file = st.file_uploader("Upload Cost File", type=['csv', 'xlsx'])
    with c2:
        same_month_file = st.file_uploader("Upload Payment File (Current Month)", type=['xlsx'])
        next_month_file = st.file_uploader("Upload Payment File (Next Month)", type=['xlsx'])

    st.markdown("### 2. Settings")
    s1, s2 = st.columns(2)
    with s1:
        pack_cost = st.number_input("Packaging Cost (per order)", value=5.0)
    with s2:
        misc_cost = st.number_input("Misc Monthly Expenses", value=0.0)

    if orders_file and same_month_file and next_month_file and cost_file:
        if st.button("🚀 Process Data", type="primary"):
            excel_data, stats, missing = process_data(orders_file, same_month_file, next_month_file, cost_file, pack_cost, misc_cost)
            
            if stats:
                st.divider()
                st.metric("NET PROFIT / LOSS", f"₹{stats['Profit / Loss']:,.2f}")
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Settlements", f"₹{stats['Total Payments']:,.2f}")
                m2.metric("Product Cost", f"₹{stats['Total Product Cost']:,.2f}")
                m3.metric("Ads Cost", f"₹{stats['Same Month Ads']:,.2f}")
                m4.metric("Recoveries", f"₹{stats['Recoveries/Charges']:,.2f}")

                if not missing.empty:
                    st.warning(f"⚠️ {len(missing)} orders have SKUs missing from the cost sheet.")
                    with st.expander("View Missing SKUs"):
                        st.dataframe(missing)

                st.download_button("⬇️ Download Detailed Report", data=excel_data, file_name="Meesho_Final_Report.xlsx", type="primary")
                st.balloons()
