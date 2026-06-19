import os, re
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text

st.set_page_config(page_title="Human Capital Analytics", page_icon="🔌")
st.title("🔌 Human Capital Analytics — Chatbot Text-to-SQL")
st.caption("Tanya data SDM dengan bahasa biasa → SQL → jawaban + grafik")

# ---- DB: pakai PostgreSQL miniproject yang sudah dimuat ----
DB_URL = os.environ.get("DB_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/miniproject")
engine = create_engine(DB_URL)

SCHEMA_STR = """employees(nip, nama, divisi, jabatan, join_date)
trainings(training_id, nama_diklat, tanggal)
enrollments(nip, training_id, status, nilai)"""

# ---- LLM ----
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""
USE_MOCK = not bool(GEMINI_API_KEY)
if GEMINI_API_KEY:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")

def build_prompt(q):
    return (f"Anda ahli SQL PostgreSQL. Gunakan HANYA skema berikut.\n"
            f"Buat SATU query SELECT, balas HANYA SQL tanpa penjelasan.\n\nSkema:\n{SCHEMA_STR}\n\nPertanyaan: {q}\nSQL:")

def _mock_sql(q):
    q = q.lower()
    if "data engineering" in q or "belum" in q:
        return ("SELECT e.nip, e.nama, e.divisi FROM employees e WHERE e.nip NOT IN "
                "(SELECT en.nip FROM enrollments en JOIN trainings t ON t.training_id=en.training_id "
                "WHERE t.nama_diklat='Data Engineering')")
    if "rata" in q and "nilai" in q:
        return ("SELECT e.divisi, ROUND(AVG(en.nilai),1) AS rata_nilai FROM enrollments en "
                "JOIN employees e ON e.nip=en.nip WHERE en.nilai IS NOT NULL GROUP BY e.divisi ORDER BY rata_nilai DESC")
    return "SELECT divisi, COUNT(*) AS jumlah FROM employees GROUP BY divisi ORDER BY jumlah DESC"

def generate_sql(q):
    if USE_MOCK:
        return _mock_sql(q)
    teks = (model.generate_content(build_prompt(q)).text or "").strip()
    m = re.search(r"```(?:sql)?\s*(.+?)```", teks, re.S)
    if m: teks = m.group(1).strip()
    m = re.search(r"(select\b.+)", teks, re.I | re.S)
    if m: teks = m.group(1)
    return teks.rstrip(";").strip()

FORBIDDEN = ["drop","delete","update","insert","alter","truncate","create","grant"]
def validate_sql(sql):
    if not sql or not sql.strip(): return False
    s = sql.strip().rstrip(";").strip(); low = s.lower()
    if not low.startswith("select"): return False
    if ";" in s: return False
    return not any(re.search(rf"\b{k}\b", low) for k in FORBIDDEN)

def run_sql(sql):
    with engine.connect() as c:
        return pd.read_sql(text(sql), c)

def chart(df):
    if df.shape[1] < 2 or len(df) == 0 or not pd.api.types.is_numeric_dtype(df[df.columns[-1]]):
        return None
    x, y = df.columns[0], df.columns[-1]
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(df[x].astype(str), df[y], color="#0E8388")
    ax.set_ylabel(str(y)); plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    return fig

if "messages" not in st.session_state:
    st.session_state.messages = []
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.write(m["content"])

q = st.chat_input("Tanya tentang data SDM…")
if q:
    st.session_state.messages.append({"role":"user","content":q})
    with st.chat_message("user"): st.write(q)
    with st.chat_message("assistant"):
        sql = generate_sql(q)
        if not validate_sql(sql):
            sql = generate_sql(q + " (hanya SELECT valid)")
        if not validate_sql(sql):
            st.error("Query valid tidak dapat disusun. Perjelas pertanyaan.")
        else:
            try:
                df = run_sql(sql)
                with st.expander("🔎 SQL"): st.code(sql, language="sql")
                st.dataframe(df, use_container_width=True)
                fig = chart(df)
                if fig: st.pyplot(fig)
                st.session_state.messages.append({"role":"assistant","content":f"{len(df)} baris."})
            except Exception as e:
                st.error(f"Gagal eksekusi: {e}")
