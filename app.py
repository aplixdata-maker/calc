import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import traceback

# --- Smart Parser Utils ---
def find_header_row(df_raw, target_keywords):
    """Finds the row index that likely contains headers based on the most keyword matches."""
    best_idx = None
    max_matches = 0
    
    for i, row in df_raw.head(10).iterrows():
        row_str = " ".join(row.astype(str).str.lower())
        matches = sum(1 for kw in target_keywords if kw.lower() in row_str)
        if matches > max_matches:
            max_matches = matches
            best_idx = i
            
    return best_idx

def smart_read_file(file, target_keywords=None, sheet_name=0):
    """Reads Excel/CSV and tries to find the correct header row."""
    if hasattr(file, 'seek'):
        file.seek(0)
        
    if file.name.lower().endswith('.csv'):
        df_raw = pd.read_csv(file, nrows=20, header=None)
        h_idx = find_header_row(df_raw, target_keywords) if target_keywords else 0
        file.seek(0)
        return pd.read_csv(file, header=h_idx if h_idx is not None else 0)
    else:
        xl = pd.ExcelFile(file)
        actual_sheet = sheet_name
        if isinstance(sheet_name, str) and sheet_name not in xl.sheet_names:
            matches = [s for s in xl.sheet_names if sheet_name.lower() in s.lower()]
            actual_sheet = matches[0] if matches else xl.sheet_names[0]
        
        df_sample = xl.parse(sheet_name=actual_sheet, nrows=20, header=None)
        h_idx = find_header_row(df_sample, target_keywords) if target_keywords else None
        
        if h_idx is None and not target_keywords:
             row0 = df_sample.iloc[0]
             if any(isinstance(x, (int, float)) for x in row0):
                 return xl.parse(sheet_name=actual_sheet, header=None)
             return xl.parse(sheet_name=actual_sheet, header=0)

        return xl.parse(sheet_name=actual_sheet, header=h_idx if h_idx is not None else 0)

def fuzzy_get_cols(df, mapping):
    """Maps fuzzy names to standard columns."""
    result = {}
    cols = [str(c).strip() for c in df.columns]
    for standard_name, aliases in mapping.items():
        found_col = None
        for alias in aliases:
            matches = [c for c in cols if alias.lower() in c.lower()]
            if matches:
                found_col = matches[0]
                break
        result[standard_name] = found_col
    return result

def get_ads_from_sheet(file):
    """Standalone logic for ads extraction to ensure robustness."""
    try:
        xl = pd.ExcelFile(file)
        # Find sheet name
        actual_sheet = "Ads Cost"
        if actual_sheet not in xl.sheet_names:
            matches = [s for s in xl.sheet_names if "ads" in s.lower()]
            actual_sheet = matches[0] if matches else xl.sheet_names[0]
            
        df_ads = smart_read_file(file, ["Ads Cost", "Total Ads Cost", "Campaign"], sheet_name=actual_sheet)
        # Find column that has ads and cost
        possible_cols = [c for c in df_ads.columns if 'ads' in str(c).lower() and 'cost' in str(c).lower()]
        if not possible_cols:
            possible_cols = [c for c in df_ads.columns if 'ad' in str(c).lower() and 'cost' in str(c).lower()]
        
        if possible_cols:
            # Prefer 'total' if multiple matches
            total_cols = [c for c in possible_cols if 'total' in str(c).lower()]
            ads_col = total_cols[0] if total_cols else possible_cols[0]
            
            vals = pd.to_numeric(df_ads[ads_col], errors='coerce').fillna(0)
            return vals.sum()
        return 0
    except:
        return 0
    finally:
        if hasattr(file, 'seek'): file.seek(0)

# --- Core Logic ---
def run_profit_calculation(orders_file, same_month_file, next_month_file, cost_file, pack_cost, misc_cost):
    # 1. Load Orders
    df_orders_raw = smart_read_file(orders_file, ["Sub Order", "SKU", "Quantity"])
    order_map = fuzzy_get_cols(df_orders_raw, {
        "so": ["Sub Order No", "Order Number"],
        "sku": ["SKU", "Supplier SKU", "Seller SKU"],
        "qty": ["Quantity", "Qty"]
    })
    
    for key in ["so", "sku", "qty"]:
        if not order_map[key]:
            raise ValueError(f"Orders file: mandatory column '{key}' not found.")

    # 2. Load Payments
    def get_payment_data(file, label):
        df = smart_read_file(file, ["Sub Order", "Settlement", "Status"], sheet_name="Order Payments")
        m = fuzzy_get_cols(df, {
            "so": ["Sub Order No", "Order Number"],
            "st": ["Live Order Status", "Status"],
            "am": ["Final Settlement Amount", "Settlement"]
        })
        for key in ["so", "st", "am"]:
            if not m[key]:
                raise ValueError(f"{label} Payment file: mandatory column '{key}' not found.")
        return df, m

    df_same, map_same = get_payment_data(same_month_file, "Current Month")
    df_next, map_next = get_payment_data(next_month_file, "Next Month")

    # 3. Load Costs
    df_cost_raw = smart_read_file(cost_file)
    if isinstance(df_cost_raw.columns[0], int) or "Unnamed" in str(df_cost_raw.columns[0]):
        df_cost_raw = df_cost_raw.iloc[:, :2]
        df_cost_raw.columns = ["sku", "price"]
    else:
        c_map = fuzzy_get_cols(df_cost_raw, {"sku": ["SKU", "Product"], "price": ["Cost", "Price", "Value"]})
        if not c_map["sku"] or not c_map["price"]:
             df_cost_raw = df_cost_raw.iloc[:, :2]
             df_cost_raw.columns = ["sku", "price"]
        else:
             df_cost_raw = df_cost_raw[[c_map["sku"], c_map["price"]]].rename(columns={c_map["sku"]: "sku", c_map["price"]: "price"})

    # 4. Ads
    total_ads = get_ads_from_sheet(same_month_file) + get_ads_from_sheet(next_month_file)

    # Merge
    for df, m in [(df_orders_raw, order_map), (df_same, map_same), (df_next, map_next)]:
        df[m["so"]] = df[m["so"]].astype(str).str.strip()

    df_pay_pool = pd.concat([
        df_same[[map_same["so"], map_same["st"], map_same["am"]]].rename(columns={map_same["so"]: "so", map_same["st"]: "st", map_same["am"]: "am"}),
        df_next[[map_next["so"], map_next["st"], map_next["am"]]].rename(columns={map_next["so"]: "so", map_next["st"]: "st", map_next["am"]: "am"})
    ]).drop_duplicates(subset=["so"], keep="last")

    def get_so_sum(df, m):
        temp = df[[m["so"], m["am"]]].copy()
        temp.columns = ["so", "am"]
        temp["am"] = pd.to_numeric(temp["am"], errors="coerce").fillna(0)
        return temp.groupby("so")["am"].sum().reset_index()

    df_sum_same = get_so_sum(df_same, map_same).rename(columns={"am": "pay_same"})
    df_sum_next = get_so_sum(df_next, map_next).rename(columns={"am": "pay_next"})

    df_final = df_orders_raw[[order_map["so"], order_map["sku"], order_map["qty"]]].copy()
    df_final.columns = ["so", "sku", "qty"]
    df_final["qty"] = pd.to_numeric(df_final["qty"], errors="coerce").fillna(0)

    df_final = df_final.merge(df_sum_same, on="so", how="left")
    df_final = df_final.merge(df_sum_next, on="so", how="left")
    df_final["total_pay"] = df_final[["pay_same", "pay_next"]].sum(axis=1)
    df_final = df_final.merge(df_pay_pool[["so", "st"]], on="so", how="left")
    
    df_cost_raw["sku"] = df_cost_raw["sku"].astype(str).str.strip()
    df_final["sku"] = df_final["sku"].astype(str).str.strip()
    df_final = df_final.merge(df_cost_raw[["sku", "price"]], on="sku", how="left")
    
    missing_df = df_final[df_final["price"].isna()][["so", "sku", "st"]].drop_duplicates()
    df_final["price"] = df_final["price"].fillna(0)

    cl_st = df_final["st"].fillna("Unknown").str.strip()
    delivered = cl_st.isin(['Delivered', 'Exchange'])
    shipped = cl_st.isin(['Delivered', 'Exchange', 'Return', 'RTO', 'Shipped', 'Ready_to_ship'])
    
    df_final["prod_cost"] = np.where(delivered, df_final["price"] * df_final["qty"], 0)
    df_final["pkg_cost"] = np.where(shipped, pack_cost, 0)
    
    stats = {
        "Total Payments": df_final["total_pay"].sum(),
        "Total Product Cost": df_final["prod_cost"].sum(),
        "Total Packaging Cost": df_final["pkg_cost"].sum(),
        "Ads Cost": abs(total_ads),
        "Misc Cost": misc_cost,
        "Delivered Count": len(df_final[cl_st == 'Delivered']),
        "Return Count": len(df_final[cl_st == 'Return']),
        "RTO Count": len(df_final[cl_st == 'RTO'])
    }
    stats["Net Profit"] = stats["Total Payments"] - stats["Total Product Cost"] - stats["Total Packaging Cost"] - stats["Ads Cost"] - stats["Misc Cost"]

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        df_final.to_excel(writer, sheet_name='Orders', index=False)
        pd.DataFrame(list(stats.items()), columns=["Metric", "Value"]).to_excel(writer, sheet_name='Summary', index=False)
    buf.seek(0)

    return buf, stats, missing_df

# --- Streamlit UI Components ---
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if st.session_state["password_correct"]:
        return True

    st.title("🔒 Please Login")
    with st.form("credentials_form"):
        user = st.text_input("Username")
        pw = st.text_input("Password", type="password")
        if st.form_submit_button("Log In"):
            secrets_ok = False
            try:
                if "passwords" in st.secrets and user in st.secrets["passwords"]:
                    if pw == st.secrets["passwords"][user]:
                        secrets_ok = True
            except: pass
            
            if secrets_ok or (user == "admin" and pw == "meesho123"):
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("😕 User not known or password incorrect")
    return False

def show_main_app():
    st.title("📊 Meesho Profit/Loss Calculator")
    
    with st.expander("Step 1: Upload Files", expanded=True):
        c1, c2 = st.columns(2)
        orf = c1.file_uploader("Orders Master File (CSV/XLSX)", type=['csv', 'xlsx'])
        csf = c1.file_uploader("Item Cost File (CSV/XLSX)", type=['csv', 'xlsx'])
        smf = c2.file_uploader("Current Month Payment (XLSX)", type=['xlsx'])
        nmf = c2.file_uploader("Next Month Payment (XLSX)", type=['xlsx'])

    with st.expander("Step 2: Settings"):
        sc1, sc2 = st.columns(2)
        p_c = sc1.number_input("Packaging Cost per Shipped Order", value=5.0)
        m_c = sc2.number_input("Miscellaneous Monthly Cost", value=0.0)

    if orf and csf and smf and nmf:
        if st.button("Calculate Profit", type="primary"):
            try:
                xl_buf, st_res, miss = run_profit_calculation(orf, smf, nmf, csf, p_c, m_c)
                if st_res:
                    st.divider()
                    st.metric("Total Net Profit", f"₹{st_res['Net Profit']:,.2f}")
                    
                    mc = st.columns(4)
                    mc[0].metric("Payments", f"₹{st_res['Total Payments']:,.2f}")
                    mc[1].metric("Product Cost", f"₹{st_res['Total Product Cost']:,.2f}")
                    mc[2].metric("Pkg Cost", f"₹{st_res['Total Packaging Cost']:,.2f}")
                    mc[3].metric("Ads Spent", f"₹{st_res['Ads Cost']:,.2f}")
                    
                    if not miss.empty:
                        st.warning(f"Found {len(miss)} missing SKUs in cost sheet.")
                        st.dataframe(miss)
                    
                    st.download_button("📥 Download Detailed Report", xl_buf, "Meesho_Profit_Report.xlsx")
            except Exception as e:
                st.error(f"Analysis Error: {str(e)}")
                st.code(traceback.format_exc())

# --- Launcher ---
if __name__ == "__main__":
    if "st" in globals() and hasattr(st, 'set_page_config'):
        # We are running via streamlit
        if "page_config_done" not in st.session_state:
            st.set_page_config(page_title="Meesho Calculator", layout="wide", initial_sidebar_state="collapsed")
            st.session_state["page_config_done"] = True
            
        if check_password():
            show_main_app()
    else:
        print("This script is designed to be run via: streamlit run app.py")
