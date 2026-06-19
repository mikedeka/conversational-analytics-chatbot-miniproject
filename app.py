import os, re
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text

st.set_page_config(page_title="Human Capital Analytics")
st.title("Human Capital Analytics — Chatbot Text-to-SQL")
st.caption("Tanya data SDM dengan bahasa biasa → SQL → jawaban + grafik")

# ---- DB: PostgreSQL miniproject yang sudah dimuat di notebook ----
# Catatan: di Colab, PostgreSQL kadang perlu di-start ulang sebelum app dijalankan:
#   !service postgresql start
DB_URL = os.environ.get("DB_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/miniproject")

@st.cache_resource
def get_engine():
    eng = create_engine(DB_URL, pool_pre_ping=True)
    with eng.connect() as c:           # tes koneksi sekali di awal
        c.execute(text("SELECT 1"))
    return eng

try:
    engine = get_engine()
    DB_OK = True
except Exception as e:
    engine = None
    DB_OK = False
    DB_ERR = str(e)

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

# ---- deteksi sapaan / basa-basi (tidak menyentuh DB) ----
SAPAAN = ("halo", "hai", "hi", "hello", "pagi", "siang", "sore", "malam",
          "terima kasih", "makasih", "thanks", "siapa kamu", "apa kabar", "test", "tes")
def is_sapaan(teks):
    t = teks.lower().strip()
    return (len(t.split()) <= 3) and any(s in t for s in SAPAAN)

SAPAAN_BALAS = ("Halo! 👋 Saya asisten analitik data SDM. "
                "Coba tanya, misalnya: *Berapa jumlah pegawai per divisi?* atau "
                "*Berapa rata-rata nilai diklat per divisi?*")

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

with st.sidebar:
    st.subheader("Status")
    st.write("DB:", "🟢 terhubung" if DB_OK else "🔴 gagal")
    st.write("LLM:", "DEMO (mock)" if USE_MOCK else "Gemini")
    if not DB_OK:
        st.caption("Jalankan `!service postgresql start` di notebook, lalu restart app.")
    if st.button("Reset percakapan"):
        st.session_state.messages = []; st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.write(m["content"])

q = st.chat_input("Tanya tentang data SDM…")
if q:
    st.session_state.messages.append({"role":"user","content":q})
    with st.chat_message("user"): st.write(q)
    with st.chat_message("assistant"):
        # 1) sapaan -> balas ramah, jangan sentuh DB
        if is_sapaan(q):
            st.write(SAPAAN_BALAS)
            st.session_state.messages.append({"role":"assistant","content":SAPAAN_BALAS})
        elif not DB_OK:
            msg = f"Database belum terhubung sehingga query tidak bisa dijalankan.\n\n`{DB_ERR}`"
            st.error(msg)
            st.session_state.messages.append({"role":"assistant","content":msg})
        else:
            sql = generate_sql(q)
            if not validate_sql(sql):
                sql = generate_sql(q + " (hanya SELECT valid)")
            if not validate_sql(sql):
                st.error("Query valid tidak dapat disusun. Coba perjelas pertanyaan.")
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
