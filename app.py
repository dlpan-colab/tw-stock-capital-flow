import sys
import os
import io
import time
import datetime
import zoneinfo
import sqlite3
import requests
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Tuple, List
from IPython.display import display, HTML

# ================= 0. 版本識別與參數設定 =================
tw_tz = zoneinfo.ZoneInfo("Asia/Taipei")
VERSION_TAG = f"v{datetime.datetime.now(tw_tz).strftime('%Y%m%d_%H%M%S')}"
RESET_STOCK_DATA = False
TARGET_DAYS = 240

EXCLUDE_PARENT_INDUSTRIES = ['合計', '總計', '電子', '電子類', '化學生技醫療']

# ================= 1. 環境與終端裝置偵測 =================
def detect_runtime_and_device() -> Tuple[str, str, str]:
    env = '本地/筆電獨立環境'
    data_dir = './data'
    device_type = "筆電/桌機 (Desktop)"

    if 'google.colab' in sys.modules:
        env = 'Colab 雲端環境'
        from google.colab import drive, output
        drive_mount = '/content/drive'
        if not os.path.exists(os.path.join(drive_mount, 'MyDrive')):
            drive.mount(drive_mount)
        data_dir = os.path.join(drive_mount, 'MyDrive', 'stock_data')
        
        try:
            ua = output.eval_js('navigator.userAgent').lower()
            if any(k in ua for k in ['mobile', 'android', 'iphone', 'ipad', 'ipod']):
                device_type = "手機 (Mobile)"
            else:
                device_type = "筆電/桌機 (Desktop)"
        except Exception:
            device_type = "筆電/桌機 (Desktop)"

    elif 'streamlit' in sys.modules:
        env = 'Streamlit 部署環境'
        data_dir = './data'
        user_agent = os.environ.get('HTTP_USER_AGENT', '').lower()
        if any(k in user_agent for k in ['mobile', 'android', 'iphone', 'ipad', 'ipod']):
            device_type = "手機 (Mobile)"
        else:
            device_type = "筆電/桌機 (Desktop)"

    os.makedirs(data_dir, exist_ok=True)
    return env, device_type, data_dir

# ================= 2. 資料庫初始化 =================
def init_database(data_dir: str) -> sqlite3.Connection:
    db_path = os.path.join(data_dir, "tw_market_data.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    
    if RESET_STOCK_DATA:
        cursor.execute("DROP TABLE IF EXISTS daily_stock_turnover")
        cursor.execute("DROP TABLE IF EXISTS daily_industry_ratio")
        cursor.execute("DROP TABLE IF EXISTS daily_industry_chips")
        cursor.execute("DROP TABLE IF EXISTS daily_top_gainers_ratio")
        conn.commit()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_industry_map (
            code TEXT PRIMARY KEY,
            name TEXT,
            market TEXT,
            industry TEXT,
            clean_ind TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_industry_ratio (
            date TEXT,
            market TEXT,
            industry TEXT,
            clean_ind TEXT,
            trade_value REAL,
            value_ratio REAL,
            PRIMARY KEY (date, market, industry)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_stock_turnover (
            date TEXT,
            market TEXT,
            code TEXT,
            name TEXT,
            close_price REAL,
            change_rate REAL,
            trade_value REAL,
            foreign_buy_net REAL,
            trust_buy_net REAL,
            inst_total_net REAL,
            PRIMARY KEY (date, market, code)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_industry_chips (
            date TEXT,
            market TEXT,
            clean_ind TEXT,
            foreign_buy_net REAL,
            trust_buy_net REAL,
            inst_total_net REAL,
            PRIMARY KEY (date, market, clean_ind)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_top_gainers_ratio (
            date TEXT,
            market TEXT,
            industry TEXT,
            clean_ind TEXT,
            trade_value REAL,
            value_ratio REAL,
            PRIMARY KEY (date, market, clean_ind)
        )
    """)
    conn.commit()

    cursor.execute("PRAGMA table_info(daily_stock_turnover)")
    columns = [col[1] for col in cursor.fetchall()]
    if 'change_rate' not in columns:
        cursor.execute("ALTER TABLE daily_stock_turnover ADD COLUMN change_rate REAL DEFAULT 0.0")
    if 'close_price' not in columns:
        cursor.execute("ALTER TABLE daily_stock_turnover ADD COLUMN close_price REAL DEFAULT 0.0")
    conn.commit()

    return conn

def normalize_industry_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return name.replace('類指數', '').replace('指數', '').replace('類', '').replace('業', '').strip()

def to_roc_date_slash(d_str: str) -> str:
    y, m, d = d_str.split('-')
    return f"{int(y) - 1911}/{m}/{d}"

def format_date_str(d_str: str) -> str:
    clean_d = str(d_str).replace('-', '').replace('/', '').strip()
    return f"{clean_d[:4]}-{clean_d[4:6]}-{clean_d[6:]}" if len(clean_d) == 8 else d_str

# ================= 3. 產業對照表建立 =================
def sync_stock_industry_map(conn: sqlite3.Connection) -> pd.DataFrame:
    df_existing = pd.read_sql("SELECT * FROM stock_industry_map", conn)
    if not df_existing.empty:
        return df_existing

    print("自官方 ISIN 建置個股產業資料庫...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    records = []

    for str_mode, market_name in [(2, '上市'), (4, '上櫃')]:
        url = f"https://isin.twse.com.tw/isin/C_public.jsp?strMode={str_mode}"
        try:
            res = requests.get(url, headers=headers, timeout=12)
            res.encoding = 'big5'
            tables = pd.read_html(io.StringIO(res.text))
            df_table = tables[0]
            df_table.columns = df_table.iloc[0]
            df_table = df_table.iloc[1:].copy()

            for _, row in df_table.iterrows():
                raw_code_name = str(row.get('有價證券代號及名稱', ''))
                industry = str(row.get('產業別', '')).strip()
                parts = raw_code_name.split('\u3000')
                if len(parts) >= 2:
                    code, name = parts[0].strip(), parts[1].strip()
                    if len(code) == 4 and code.isdigit() and industry and industry != 'nan':
                        records.append({
                            'code': code,
                            'name': name,
                            'market': market_name,
                            'industry': industry,
                            'clean_ind': normalize_industry_name(industry)
                        })
        except Exception:
            pass

    if records:
        df_map = pd.DataFrame(records).drop_duplicates(subset=['code'])
        df_map.to_sql('stock_industry_map', conn, if_exists='replace', index=False)
        return df_map
    return pd.DataFrame()

# ================= 4. 連線與數據抓取 =================
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/html, */*',
    'Accept-Language': 'zh-TW,zh;q=0.9',
})

def fetch_twse_day_data(date_str: str, df_map: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    norm_date = format_date_str(date_str)
    raw_date = date_str.replace('-', '')
    
    url_stk = f"https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX?date={raw_date}&type=ALLBUT0999&response=json"
    stock_dict = {}
    try:
        res_s = session.get(url_stk, timeout=12)
        if res_s.status_code == 200:
            data_json = res_s.json()
            tables = data_json.get('tables', [])
            stock_rows = []
            if 'data9' in data_json:
                stock_rows = data_json['data9']
            else:
                for t in tables:
                    if "每日收盤行情" in t.get('title', ''):
                        stock_rows = t.get('data', [])
                        break
            
            for r in stock_rows:
                if len(r) >= 11:
                    code, name = str(r[0]).strip(), str(r[1]).strip()
                    if len(code) == 4 and code.isdigit():
                        try:
                            val = float(str(r[4]).replace(',', '').strip() or 0)
                            close_p = float(str(r[8]).replace(',', '').strip() or 0)
                            sign = str(r[9]).strip()
                            diff = float(str(r[10]).replace(',', '').strip() or 0)
                            if '-' in sign:
                                diff = -diff
                            elif '+' in sign:
                                diff = abs(diff)

                            prev_p = close_p - diff
                            change_rate = (diff / prev_p * 100.0) if prev_p > 0 else 0.0

                            if val > 0:
                                stock_dict[code] = {
                                    'code': code,
                                    'name': name,
                                    'close_price': close_p,
                                    'change_rate': change_rate,
                                    'trade_value': val,
                                    'foreign_buy_net': 0.0,
                                    'trust_buy_net': 0.0,
                                    'inst_total_net': 0.0,
                                    'date': norm_date,
                                    'market': '上市'
                                }
                        except ValueError:
                            pass
    except Exception:
        pass

    if not stock_dict:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    url_t86 = f"https://www.twse.com.tw/rwd/zh/fund/T86?date={raw_date}&selectType=ALLBUT0999&response=json"
    try:
        res_f = session.get(url_t86, timeout=12)
        if res_f.status_code == 200:
            d_t86 = res_f.json()
            if d_t86.get('stat') in ['OK', '查詢成功'] and 'data' in d_t86:
                for r in d_t86['data']:
                    code = str(r[0]).strip()
                    if code in stock_dict:
                        try:
                            f_net = float(str(r[4]).replace(',', '').strip() or 0)
                            t_net = float(str(r[10]).replace(',', '').strip() or 0)
                            tot_net = float(str(r[18]).replace(',', '').strip() or 0)
                            stock_dict[code]['foreign_buy_net'] = f_net
                            stock_dict[code]['trust_buy_net'] = t_net
                            stock_dict[code]['inst_total_net'] = tot_net
                        except (ValueError, IndexError):
                            pass
    except Exception:
        pass

    df_stock = pd.DataFrame(list(stock_dict.values()))

    df_ind = pd.DataFrame(columns=['industry', 'clean_ind', 'trade_value', 'value_ratio', 'date', 'market'])
    if not df_stock.empty and not df_map.empty:
        df_m = pd.merge(df_stock, df_map[['code', 'industry', 'clean_ind']], on='code', how='inner')
        df_m = df_m[~df_m['industry'].isin(EXCLUDE_PARENT_INDUSTRIES)]
        if not df_m.empty:
            df_grp = df_m.groupby(['industry', 'clean_ind'])['trade_value'].sum().reset_index()
            tot = df_grp['trade_value'].sum()
            if tot > 0:
                df_grp['value_ratio'] = (df_grp['trade_value'] / tot) * 100.0
                df_grp['date'] = norm_date
                df_grp['market'] = '上市'
                df_ind = df_grp[['industry', 'clean_ind', 'trade_value', 'value_ratio', 'date', 'market']]

    df_chips = pd.DataFrame(columns=['date', 'market', 'clean_ind', 'foreign_buy_net', 'trust_buy_net', 'inst_total_net'])
    if not df_stock.empty and not df_map.empty:
        df_merged = pd.merge(df_stock, df_map[['code', 'clean_ind']], on='code', how='inner')
        if not df_merged.empty:
            df_chips = df_merged.groupby('clean_ind')[['foreign_buy_net', 'trust_buy_net', 'inst_total_net']].sum().reset_index()
            df_chips['date'] = norm_date
            df_chips['market'] = '上市'

    return df_ind, df_stock, df_chips

def fetch_tpex_day_data(date_str: str, df_map: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    norm_date = format_date_str(date_str)
    roc_d = to_roc_date_slash(norm_date)
    
    url_stk = f"https://www.tpex.org.tw/web/stock/aftertrading/otc_quotes_no1430/stk_wn1430_result.php?l=zh-tw&d={roc_d}&se=EW&response=json"
    stock_dict = {}
    try:
        res = session.get(url_stk, timeout=12)
        if res.status_code == 200:
            d_json = res.json()
            raw_rows = d_json.get('aaData', []) or d_json.get('tables', [{}])[0].get('data', [])
            for r in raw_rows:
                if len(r) >= 9:
                    code, name = str(r[0]).strip(), str(r[1]).strip()
                    if len(code) == 4 and code.isdigit():
                        try:
                            amt = float(str(r[8]).replace(',', '').strip() or 0)
                            close_str = str(r[2]).replace(',', '').strip()
                            close_p = float(close_str) if close_str and close_str != '---' else 0.0

                            diff_raw = str(r[3]).strip()
                            diff_val = float(diff_raw.replace('+', '').replace('-', '').replace(',', '').replace('<font color=red>', '').replace('<font color=green>', '').replace('</font>', '').strip() or 0)
                            if '-' in diff_raw or 'green' in diff_raw:
                                diff = -diff_val
                            elif '+' in diff_raw or 'red' in diff_raw:
                                diff = diff_val
                            else:
                                diff = 0.0

                            prev_p = close_p - diff
                            change_rate = (diff / prev_p * 100.0) if prev_p > 0 else 0.0

                            if amt > 0:
                                stock_dict[code] = {
                                    'code': code,
                                    'name': name,
                                    'close_price': close_p,
                                    'change_rate': change_rate,
                                    'trade_value': amt,
                                    'foreign_buy_net': 0.0,
                                    'trust_buy_net': 0.0,
                                    'inst_total_net': 0.0,
                                    'date': norm_date,
                                    'market': '上櫃'
                                }
                        except ValueError:
                            pass
    except Exception:
        pass

    if not stock_dict:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    url_tpex_inst = f"https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_result.php?l=zh-tw&d={roc_d}&response=json"
    try:
        res_i = session.get(url_tpex_inst, timeout=12)
        if res_i.status_code == 200:
            d_i_json = res_i.json()
            tables = d_i_json.get('tables', [])
            raw_i_rows = tables[0].get('data', []) if tables else []
            for r in raw_i_rows:
                if len(r) >= 24:
                    code = str(r[0]).strip()
                    if code in stock_dict:
                        try:
                            f_net = float(str(r[4]).replace(',', '').strip() or 0)
                            t_net = float(str(r[10]).replace(',', '').strip() or 0)
                            tot_net = float(str(r[23]).replace(',', '').strip() or 0)
                            stock_dict[code]['foreign_buy_net'] = f_net
                            stock_dict[code]['trust_buy_net'] = t_net
                            stock_dict[code]['inst_total_net'] = tot_net
                        except (ValueError, IndexError):
                            pass
    except Exception:
        pass

    df_stock = pd.DataFrame(list(stock_dict.values()))
    
    df_ind = pd.DataFrame(columns=['industry', 'clean_ind', 'trade_value', 'value_ratio', 'date', 'market'])
    df_chips = pd.DataFrame(columns=['date', 'market', 'clean_ind', 'foreign_buy_net', 'trust_buy_net', 'inst_total_net'])
    
    if not df_stock.empty and not df_map.empty:
        df_merged = pd.merge(df_stock, df_map[['code', 'clean_ind']], on='code', how='inner')
        if not df_merged.empty:
            df_agg = df_merged.groupby('clean_ind')['trade_value'].sum().reset_index()
            tot_val = df_agg['trade_value'].sum()
            if tot_val > 0:
                df_agg['value_ratio'] = (df_agg['trade_value'] / tot_val) * 100.0
                df_agg['date'] = norm_date
                df_agg['market'] = '上櫃'
                df_agg['industry'] = df_agg['clean_ind']
                df_ind = df_agg[['industry', 'clean_ind', 'trade_value', 'value_ratio', 'date', 'market']]

            df_chips = df_merged.groupby('clean_ind')[['foreign_buy_net', 'trust_buy_net', 'inst_total_net']].sum().reset_index()
            df_chips['date'] = norm_date
            df_chips['market'] = '上櫃'

    return df_ind, df_stock, df_chips

# ================= 5. 資料同步排程（智能對齊、增量寫入與分批回補） =================
def load_or_sync_data(conn: sqlite3.Connection, target_valid_days: int = 240) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cursor = conn.cursor()
    cursor.execute("DELETE FROM daily_industry_ratio WHERE date = '2026-10-09' OR trade_value <= 0")
    cursor.execute("DELETE FROM daily_stock_turnover WHERE date = '2026-10-09' OR trade_value <= 0")
    cursor.execute("DELETE FROM daily_industry_chips WHERE date = '2026-10-09'")
    cursor.execute("DELETE FROM daily_top_gainers_ratio WHERE date = '2026-10-09'")
    conn.commit()

    df_map = sync_stock_industry_map(conn)
    df_hist_ind = pd.read_sql("SELECT * FROM daily_industry_ratio WHERE trade_value > 0", conn)
    df_hist_stock = pd.read_sql("SELECT * FROM daily_stock_turnover WHERE trade_value > 0", conn)
    df_hist_chips = pd.read_sql("SELECT * FROM daily_industry_chips", conn)

    # 檢查有真實漲幅 (change_rate > 0) 的有效日
    tw_valid_days = set(df_hist_stock[(df_hist_stock['market'] == '上市') & (df_hist_stock['change_rate'] > 0)]['date'].dropna().unique())
    tp_valid_days = set(df_hist_stock[(df_hist_stock['market'] == '上櫃') & (df_hist_stock['change_rate'] > 0)]['date'].dropna().unique())

    tw_days = set(df_hist_ind[df_hist_ind['market'] == '上市']['date'].dropna().unique()) if not df_hist_ind.empty else set()
    tp_days = set(df_hist_ind[df_hist_ind['market'] == '上櫃']['date'].dropna().unique()) if not df_hist_ind.empty else set()

    now_dt = datetime.datetime.now(tw_tz)
    check_ptr = now_dt if now_dt.hour >= 15 else now_dt - datetime.timedelta(days=1)
    
    print(f"資料庫狀態 -> 上市已有 {len(tw_days)} 天（最新: {max(tw_days) if tw_days else '無'}），上櫃已有 {len(tp_days)} 天（最新: {max(tp_days) if tp_days else '無'}）")

    latest_tw = max(tw_days) if tw_days else "2024-01-01"
    latest_tp = max(tp_days) if tp_days else "2024-01-01"

    missing_dates = []

    # 1. 遞增模式：僅掃描最新交易日
    ptr = check_ptr
    while ptr.strftime('%Y-%m-%d') >= min(latest_tw, latest_tp):
        d_str = ptr.strftime('%Y%m%d')
        d_f = format_date_str(d_str)
        if ptr.weekday() < 5 and d_f != '2026-10-09':
            if (d_f not in tw_valid_days) or (d_f not in tp_valid_days):
                if d_str not in missing_dates:
                    missing_dates.append(d_str)
        ptr -= datetime.timedelta(days=1)

    # 2. 歷史回溯：若過去歷史紀錄缺少有效漲幅，自動排入補齊名單（上限 30 天分批回補避免超時）
    if len(tw_valid_days) < target_valid_days or len(tp_valid_days) < target_valid_days:
        all_hist_dates = sorted(list(tw_days.union(tp_days)), reverse=True)
        for d_f in all_hist_dates:
            d_str = d_f.replace('-', '')
            if (d_f not in tw_valid_days) or (d_f not in tp_valid_days):
                if d_str not in missing_dates:
                    missing_dates.append(d_str)
            if len(missing_dates) >= 30:
                break

    if missing_dates:
        print(f"共有 {len(missing_dates)} 個交易日需補齊/校正，開始【擷取一筆、立即寫入一筆】...")
        for d in missing_dates:
            d_formatted = format_date_str(d)
            
            df_tw_i, df_tw_s, df_tw_c = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            df_tp_i, df_tp_s, df_tp_c = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

            if d_formatted not in tw_valid_days:
                df_tw_i, df_tw_s, df_tw_c = fetch_twse_day_data(d, df_map)
                time.sleep(0.8)

            if d_formatted not in tp_valid_days:
                df_tp_i, df_tp_s, df_tp_c = fetch_tpex_day_data(d, df_map)
                time.sleep(0.8)

            day_inds = [df for df in [df_tw_i, df_tp_i] if not df.empty and df['trade_value'].sum() > 0]
            day_stocks = [df for df in [df_tw_s, df_tp_s] if not df.empty and df['trade_value'].sum() > 0]
            day_chips = [df for df in [df_tw_c, df_tp_c] if not df.empty]

            has_new_write = False
            if day_inds:
                df_d_ind = pd.concat(day_inds, ignore_index=True).drop_duplicates(subset=['date', 'market', 'industry'])
                for _, r in df_d_ind.iterrows():
                    cursor.execute("""
                        INSERT OR REPLACE INTO daily_industry_ratio (date, market, industry, clean_ind, trade_value, value_ratio)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (r['date'], r['market'], r['industry'], r['clean_ind'], r['trade_value'], r['value_ratio']))
                has_new_write = True

            if day_stocks:
                df_d_stock = pd.concat(day_stocks, ignore_index=True).drop_duplicates(subset=['date', 'market', 'code'])
                for _, r in df_d_stock.iterrows():
                    cursor.execute("""
                        INSERT OR REPLACE INTO daily_stock_turnover (date, market, code, name, close_price, change_rate, trade_value, foreign_buy_net, trust_buy_net, inst_total_net)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (r['date'], r['market'], r['code'], r['name'], r.get('close_price', 0.0), r.get('change_rate', 0.0), r['trade_value'], r['foreign_buy_net'], r['trust_buy_net'], r['inst_total_net']))
                has_new_write = True

            if day_chips:
                df_d_chips = pd.concat(day_chips, ignore_index=True).drop_duplicates(subset=['date', 'market', 'clean_ind'])
                for _, r in df_d_chips.iterrows():
                    cursor.execute("""
                        INSERT OR REPLACE INTO daily_industry_chips (date, market, clean_ind, foreign_buy_net, trust_buy_net, inst_total_net)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (r['date'], r['market'], r['clean_ind'], r['foreign_buy_net'], r['trust_buy_net'], r['inst_total_net']))
                has_new_write = True

            if has_new_write:
                conn.commit()
                print(f"成功寫入並同步存檔交易日: {d_formatted}")
    else:
        print("資料庫已是最新狀態，無需同步。")

    df_hist_ind = pd.read_sql("SELECT * FROM daily_industry_ratio WHERE trade_value > 0", conn)
    df_hist_stock = pd.read_sql("SELECT * FROM daily_stock_turnover WHERE trade_value > 0", conn)
    df_hist_chips = pd.read_sql("SELECT * FROM daily_industry_chips", conn)
    
    return df_hist_ind, df_hist_stock, df_hist_chips, df_map

# ================= 6. 漲幅前 100 名產業佔比（持久化＋增量計算） =================
def sync_top_gainers_ratio(conn: sqlite3.Connection, df_stk: pd.DataFrame, df_map: pd.DataFrame) -> pd.DataFrame:
    cursor = conn.cursor()
    df_exist = pd.read_sql("SELECT * FROM daily_top_gainers_ratio", conn)
    exist_pairs = set(zip(df_exist['date'], df_exist['market'])) if not df_exist.empty else set()

    if df_stk.empty or df_map.empty:
        return df_exist

    df_m = pd.merge(df_stk, df_map[['code', 'clean_ind']], on='code', how='inner')
    df_m = df_m[~df_m['clean_ind'].isin(EXCLUDE_PARENT_INDUSTRIES)]
    df_m['change_rate'] = pd.to_numeric(df_m['change_rate'], errors='coerce').fillna(0.0)

    all_pairs = set(zip(df_m['date'], df_m['market']))
    need_calc_pairs = all_pairs - exist_pairs

    if need_calc_pairs:
        print(f"增量計算新交易日之漲幅前 100 名產業佔比（共 {len(need_calc_pairs)} 個市場交易日）...")
        new_records = []
        for d, market in need_calc_pairs:
            sub_day = df_m[(df_m['date'] == d) & (df_m['market'] == market)]
            valid_gainers = sub_day[sub_day['change_rate'] > 0]
            if valid_gainers.empty:
                continue
            
            top100 = valid_gainers.sort_values(by='change_rate', ascending=False).head(100)
            tot_val = top100['trade_value'].sum()
            if tot_val > 0:
                for ind, sub_ind in top100.groupby('clean_ind'):
                    ind_val = sub_ind['trade_value'].sum()
                    new_records.append({
                        'date': d,
                        'market': market,
                        'industry': ind,
                        'clean_ind': ind,
                        'trade_value': ind_val,
                        'value_ratio': (ind_val / tot_val) * 100.0
                    })
        
        if new_records:
            df_new = pd.DataFrame(new_records)
            for _, r in df_new.iterrows():
                cursor.execute("""
                    INSERT OR REPLACE INTO daily_top_gainers_ratio (date, market, industry, clean_ind, trade_value, value_ratio)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (r['date'], r['market'], r['industry'], r['clean_ind'], r['trade_value'], r['value_ratio']))
            conn.commit()
            print("新交易日漲幅佔比已成功持久化寫入資料庫！")

    df_final = pd.read_sql("SELECT * FROM daily_top_gainers_ratio", conn)
    return df_final

# ================= 7. 多模式 Plotly 圖表生成 =================
def render_plotly_with_infobar(
    df_ind: pd.DataFrame, 
    df_stk: pd.DataFrame, 
    df_chips: pd.DataFrame, 
    df_gainers: pd.DataFrame,
    df_map: pd.DataFrame, 
    market_name: str, 
    device_type: str, 
    mode_option: str,
    chart_id: str
) -> None:
    
    mode_configs = {
        '1': {'col': 'value_ratio', 'title': '全市場類股成交比率 (%)', 'suffix': '%', 'scale': 1.0},
        '2': {'col': 'foreign_buy_net', 'title': '外資買賣超金額 (億元)', 'suffix': '億', 'scale': 1e8},
        '3': {'col': 'inst_total_net', 'title': '三大法人合計買賣超金額 (億元)', 'suffix': '億', 'scale': 1e8},
        '4': {'col': 'inst_total_net', 'title': '主力籌碼指標 (億元)', 'suffix': '億', 'scale': 1e8},
        '5': {'col': 'value_ratio', 'title': '漲幅前100名個股產業佔比 (%)', 'suffix': '%', 'scale': 1.0},
    }
    cfg = mode_configs.get(mode_option, mode_configs['1'])
    
    if mode_option == '1':
        df_target = df_ind[
            (df_ind['market'] == market_name) & 
            (~df_ind['industry'].isin(EXCLUDE_PARENT_INDUSTRIES)) &
            (df_ind['trade_value'] > 0)
        ].copy()
        if 'clean_ind' not in df_target.columns:
            df_target['clean_ind'] = df_target['industry'].apply(normalize_industry_name)
    elif mode_option == '5':
        df_target = df_gainers[df_gainers['market'] == market_name].copy() if not df_gainers.empty else pd.DataFrame()
    else:
        df_target = df_chips[
            (df_chips['market'] == market_name) &
            (~df_chips['clean_ind'].isin(EXCLUDE_PARENT_INDUSTRIES))
        ].copy() if not df_chips.empty else pd.DataFrame()
        if not df_target.empty:
            df_target['industry'] = df_target['clean_ind']

    if df_target.empty or 'date' not in df_target.columns:
        print(f"{market_name} 查無資料")
        return

    today_str = datetime.datetime.now(tw_tz).strftime('%Y-%m-%d')
    df_target['date'] = df_target['date'].astype(str).str.strip()
    df_target = df_target[df_target['date'] <= today_str]

    val_col = cfg['col']
    df_target['metric_val'] = (pd.to_numeric(df_target[val_col], errors='coerce').fillna(0)) / cfg['scale']

    dates = sorted(df_target['date'].dropna().unique().tolist())
    n_days = len(dates)
    if n_days == 0:
        print(f"{market_name} 查無有效交易日")
        return

    tick_texts = []
    prev_m = None
    for d in dates:
        try:
            curr_m = int(d.split('-')[1])
            if curr_m != prev_m:
                tick_texts.append(f"{curr_m}月")
                prev_m = curr_m
            else:
                tick_texts.append("")
        except Exception:
            tick_texts.append("")

    is_mobile = "手機" in device_type
    default_days_count = 20 if is_mobile else 60

    daily_summary_dict = {}
    for d, sub_df in df_target.groupby('date'):
        sub_sorted = sub_df.sort_values(by='metric_val', ascending=False)
        items = [
            f"{r['clean_ind'] if pd.notnull(r['clean_ind']) and str(r['clean_ind']).strip() not in ['', 'None'] else r['industry']}:{r['metric_val']:+.2f}{cfg['suffix']}"
            for _, r in sub_sorted.iterrows()
        ]
        daily_summary_dict[d] = f"<b>【{market_name} 關注交易日：{d}】</b> " + " | ".join(items)

    latest_date = max(dates)
    init_info_text = daily_summary_dict.get(latest_date, "")

    recent_dates = dates[-default_days_count:] if n_days > default_days_count else dates
    df_recent = df_target[df_target['date'].isin(recent_dates)]
    if not df_recent.empty:
        y_min = float(df_recent['metric_val'].min())
        y_max = float(df_recent['metric_val'].max())
        y_span = y_max - y_min
        pad = y_span * 0.08 if y_span > 0 else 0.5
        default_y_range = [y_min - pad, y_max + pad]
    else:
        default_y_range = None

    if not df_map.empty and not df_stk.empty and 'code' in df_stk.columns:
        if 'clean_ind' not in df_map.columns:
            df_map['clean_ind'] = df_map['industry'].apply(normalize_industry_name)
        df_stk_merged = pd.merge(df_stk[df_stk['market'] == market_name], df_map[['code', 'clean_ind']], on='code', how='inner')
    else:
        df_stk_merged = pd.DataFrame()

    if 'change_rate' not in df_stk_merged.columns:
        df_stk_merged['change_rate'] = 0.0

    stock_top3_dict = {}
    if not df_stk_merged.empty:
        for (d, c_ind), sub_df in df_stk_merged.groupby(['date', 'clean_ind']):
            if mode_option == '5':
                day_all = df_stk_merged[df_stk_merged['date'] == d]
                top100_day = day_all[day_all['change_rate'] > 0].sort_values(by='change_rate', ascending=False).head(100)
                sub_top100 = top100_day[top100_day['clean_ind'] == c_ind]
                top3 = sub_top100.sort_values(by='change_rate', ascending=False).head(3)
                details = "<br>".join([f"  {r['code']}{r['name']}: {r['change_rate']:+.2f}% ({r['trade_value']/1e8:.2f}億)" for _, r in top3.iterrows()])
                stock_top3_dict[(d, c_ind)] = f"前三大漲幅強勢股:<br>{details}" if details else "  (無進榜個股)"
            elif mode_option == '1':
                top3 = sub_df.sort_values(by='trade_value', ascending=False).head(3)
                details = "<br>".join([f"  {r['code']}{r['name']}: {r['trade_value']/1e8:.2f}億" for _, r in top3.iterrows()])
                stock_top3_dict[(d, c_ind)] = f"前三大成交個股:<br>{details}"
            else:
                rank_col = 'foreign_buy_net' if mode_option == '2' else 'inst_total_net'
                sub_target = df_target[(df_target['date'] == d) & (df_target['clean_ind'] == c_ind)]
                metric_sum = sub_target['metric_val'].iloc[0] if not sub_target.empty else 0.0
                if metric_sum < 0:
                    top3_sell = sub_df.sort_values(by=rank_col, ascending=True).head(3)
                    details = "<br>".join([f"  {r['code']}{r['name']}: {r[rank_col]/1e8:.2f}億" for _, r in top3_sell.iterrows()])
                    stock_top3_dict[(d, c_ind)] = f"前三大賣超個股:<br>{details}"
                else:
                    top3_buy = sub_df.sort_values(by=rank_col, ascending=False).head(3)
                    details = "<br>".join([f"  {r['code']}{r['name']}: {r[rank_col]/1e8:.2f}億" for _, r in top3_buy.iterrows()])
                    stock_top3_dict[(d, c_ind)] = f"前三大買超個股:<br>{details}"

    fig = go.Figure()
    for ind in sorted(df_target['industry'].unique()):
        sub_line = df_target[df_target['industry'] == ind].sort_values(by='date')
        c_name = normalize_industry_name(ind)
        customdata = [stock_top3_dict.get((d, c_name), "  (無資料)") for d in sub_line['date']]
        
        fig.add_trace(go.Scatter(
            x=sub_line['date'],
            y=sub_line['metric_val'],
            mode='lines+markers',
            name=ind,
            customdata=customdata,
            hovertemplate=f"<b>%{{x}} 【%{{data.name}}】</b><br>{cfg['title']}: %{{y:.2f}}{cfg['suffix']}<br>%{{customdata}}<extra></extra>"
        ))

    latest_x_idx = dates.index(latest_date)
    start_x_idx = max(-0.5, latest_x_idx - default_days_count + 0.5)
    end_x_idx = latest_x_idx + 0.5
    default_x_range = [start_x_idx, end_x_idx]

    c_width = 720 if is_mobile else 1050
    c_height = 680 if is_mobile else 620

    fig.update_layout(
        title=dict(
            text=f"{market_name} {cfg['title']} 走勢圖 (檢視期程: {default_days_count} 交易日)", 
            x=0.02, 
            y=0.98,
            font=dict(size=14)
        ),
        width=c_width,
        height=c_height,
        autosize=False,
        dragmode="pan",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(size=9 if is_mobile else 10)
        ),
        xaxis=dict(
            title="交易月份",
            type="category",
            categoryorder="array",
            categoryarray=dates,
            tickmode="array",
            tickvals=dates,
            ticktext=tick_texts,
            tickangle=0,
            range=default_x_range,
            minallowed=-0.5,
            maxallowed=end_x_idx,
            fixedrange=False,
            rangeslider=dict(visible=False),
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikethickness=1,
            spikedash="dash",
            spikecolor="#888888"
        ),
        yaxis=dict(
            title=cfg['title'],
            ticksuffix=cfg['suffix'],
            range=default_y_range,
            autorange=False,
            fixedrange=False,
            tickformat=".1f"
        ),
        hovermode="closest",
        template="plotly_white",
        margin=dict(l=55, r=20, t=60, b=140)
    )

    chart_config = {
        'scrollZoom': True,
        'displayModeBar': True,
        'modeBarButtonsToRemove': ['zoomIn2d', 'zoomOut2d', 'resetScale2d'],
        'responsive': True
    }
    plotly_div_html = fig.to_html(
        include_plotlyjs='cdn',
        full_html=False,
        div_id=f"plotly_div_{chart_id}",
        config=chart_config
    )

    dict_json = json.dumps(daily_summary_dict, ensure_ascii=False)
    
    wrapper_html = f"""
    <div style="width: {c_width}px; margin-bottom: 25px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        <div id="info_{chart_id}" style="width: 100%; min-height: 26px; padding: 6px 12px; box-sizing: border-box; background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 6px; font-size: 11px; font-family: monospace; color: #24292f; white-space: nowrap; overflow-x: auto; margin-bottom: 6px;">
            {init_info_text}
        </div>
        {plotly_div_html}
    </div>
    <script>
    (function() {{
        var dMap = {dict_json};
        var bar = document.getElementById("info_{chart_id}");
        var gd = document.getElementById("plotly_div_{chart_id}");
        if (gd && bar) {{
            function updateBar(data) {{
                if (data && data.points && data.points.length > 0) {{
                    var curX = data.points[0].x;
                    if (dMap[curX]) {{
                        bar.innerHTML = dMap[curX];
                    }}
                }}
            }}
            gd.on('plotly_hover', updateBar);
            gd.on('plotly_click', updateBar);
        }}
    }})();
    </script>
    """
    display(HTML(wrapper_html))

# ================= 8. 匯出 Excel（檔名標註功能簡述、保留交易日期、無版本識別） =================
def export_excel(df_ind: pd.DataFrame, df_stk: pd.DataFrame, df_map: pd.DataFrame, data_dir: str, mode_option: str) -> str:
    now = datetime.datetime.now(tw_tz)

    mode_name_map = {
        '1': '全市場成交比率',
        '2': '外資買賣超',
        '3': '三大法人合計',
        '4': '主力籌碼',
        '5': '漲幅前百大產業'
    }
    func_name = mode_name_map.get(mode_option, '全市場成交比率')

    dates = sorted(df_ind['date'].dropna().unique(), reverse=True) if 'date' in df_ind.columns else []
    latest_data_date = dates[0].replace('-', '') if dates else '無資料'

    filename = f"{func_name}_{latest_data_date}_{now.strftime('%Y%m%d%H%M')}.xlsx"
    export_path = os.path.join(data_dir, filename)

    if not df_map.empty and not df_stk.empty:
        if 'clean_ind' not in df_map.columns:
            df_map['clean_ind'] = df_map['industry'].apply(normalize_industry_name)
        df_stk_m = pd.merge(df_stk, df_map[['code', 'clean_ind']], on='code', how='left')
    else:
        df_stk_m = df_stk.copy()
        df_stk_m['clean_ind'] = '未分類'

    if 'change_rate' not in df_stk_m.columns:
        df_stk_m['change_rate'] = 0.0

    # 工作表 1: 類股成交資金流動
    records_turnover = []
    for d in dates:
        for market in ['上市', '上櫃']:
            sub_i = df_ind[
                (df_ind['date'] == d) & 
                (df_ind['market'] == market) & 
                (~df_ind['industry'].isin(EXCLUDE_PARENT_INDUSTRIES))
            ].sort_values(by='value_ratio', ascending=False)
            top4 = sub_i.head(4)

            for rank, (_, row) in enumerate(top4.iterrows(), 1):
                i_name = row['industry']
                c_name = normalize_industry_name(i_name)
                ratio = row['value_ratio']
                
                top5_stocks = ["-", "-", "-", "-", "-"]
                if not df_stk_m.empty and 'trade_value' in df_stk_m.columns:
                    sub_s = df_stk_m[(df_stk_m['date'] == d) & (df_stk_m['market'] == market) & (df_stk_m['clean_ind'] == c_name)]
                    top5_df = sub_s.sort_values(by='trade_value', ascending=False).head(5)
                    for idx, (_, s_row) in enumerate(top5_df.iterrows()):
                        top5_stocks[idx] = f"{s_row['code']}{s_row['name']}({s_row['trade_value']/1e8:.2f}億)"

                records_turnover.append({
                    '交易日期': d,
                    '市場別': market,
                    '比率排名': f"第 {rank} 名",
                    '產業類別': i_name,
                    '成交金額比率(%)': ratio,
                    '成交第1名個股': top5_stocks[0],
                    '成交第2名個股': top5_stocks[1],
                    '成交第3名個股': top5_stocks[2],
                    '成交第4名個股': top5_stocks[3],
                    '成交第5名個股': top5_stocks[4],
                })

    df_out_turnover = pd.DataFrame(records_turnover)

    # 工作表 2: 當日漲幅前100名個股產業資金比率（嚴格排除 0% 與負數，每檔個股獨立單獨欄位）
    records_gainers = []
    for d in dates:
        for market in ['上市', '上櫃']:
            sub_market_stk = df_stk_m[(df_stk_m['date'] == d) & (df_stk_m['market'] == market)]
            if sub_market_stk.empty:
                continue
            
            valid_gainers = sub_market_stk[sub_market_stk['change_rate'] > 0]
            if valid_gainers.empty:
                continue

            top100_day = valid_gainers.sort_values(by='change_rate', ascending=False).head(100)
            tot_val = top100_day['trade_value'].sum()
            if tot_val <= 0:
                continue

            ind_agg = top100_day.groupby('clean_ind')['trade_value'].sum().reset_index()
            ind_agg['value_ratio'] = (ind_agg['trade_value'] / tot_val) * 100.0
            ind_agg = ind_agg.sort_values(by='value_ratio', ascending=False)
            top5_inds = ind_agg.head(5)

            for rank, (_, i_row) in enumerate(top5_inds.iterrows(), 1):
                c_name = i_row['clean_ind']
                ratio = i_row['value_ratio']
                
                sub_ind_top = top100_day[top100_day['clean_ind'] == c_name].sort_values(by='change_rate', ascending=False)
                
                row_dict = {
                    '交易日期': d,
                    '市場別': market,
                    '佔比排名': f"第 {rank} 名",
                    '產業類別': c_name,
                    '前100名成交市值佔比(%)': ratio,
                    '進榜檔數': len(sub_ind_top),
                }
                
                for idx, (_, s_row) in enumerate(sub_ind_top.iterrows(), 1):
                    row_dict[f'第{idx}檔個股'] = f"{s_row['code']}{s_row['name']}({s_row['change_rate']:+.2f}%, {s_row['trade_value']/1e8:.2f}億)"
                
                records_gainers.append(row_dict)

    df_out_gainers = pd.DataFrame(records_gainers)

    with pd.ExcelWriter(export_path, engine='openpyxl') as writer:
        df_out_turnover.to_excel(writer, sheet_name='類股成交資金流動', index=False)
        if not df_out_gainers.empty:
            df_out_gainers.to_excel(writer, sheet_name='漲幅前百大產業分佈', index=False)

    return export_path

# ================= 9. 執行進入點 =================
if __name__ == "__main__":
    env, device_type, data_dir = detect_runtime_and_device()
    print("=" * 60)
    print(f"【系統版本識別】: {VERSION_TAG}")
    print(f"【當前執行環境】: {env}")
    print(f"【終端裝置類型】: {device_type}")
    print(f"【資料儲存目錄】: {data_dir}")
    print("=" * 60)

    conn = init_database(data_dir)
    df_ind, df_stk, df_chips, df_map = load_or_sync_data(conn, target_valid_days=TARGET_DAYS)
    df_gainers = sync_top_gainers_ratio(conn, df_stk, df_map)

    print("\n" + "=" * 40)
    print("【請選擇欲檢視的類股日趨勢指標】:")
    print("  [1] 全市場類股成交金額比率走勢圖 (%)")
    print("  [2] 各類股外資買賣超金額走勢圖 (億元)")
    print("  [3] 各類股三大法人合計買賣超金額走勢圖 (億元)")
    print("  [4] 各類股主力籌碼指標走勢圖 (億元)")
    print("  [5] 每日漲幅前100名個股產業佔比走勢圖 (%)")
    print("=" * 40)

    user_choice = input("請輸入選項編號 (1 / 2 / 3 / 4 / 5，直接 Enter 預設為 5): ").strip()
    if user_choice not in ['1', '2', '3', '4', '5']:
        user_choice = '5'

    render_plotly_with_infobar(df_ind, df_stk, df_chips, df_gainers, df_map, '上市', device_type, user_choice, chart_id='tw')
    render_plotly_with_infobar(df_ind, df_stk, df_chips, df_gainers, df_map, '上櫃', device_type, user_choice, chart_id='tp')

    excel_path = export_excel(df_ind, df_stk, df_map, data_dir, user_choice)
    print(f"\nExcel 報表產製完成，路徑: {excel_path}")
    conn.close()
