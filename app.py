import streamlit as st
import os, re, glob, zipfile
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
import google.generativeai as genai
# gdown is not needed in app.py as dataset loading is handled during notebook setup

# --- Configuration ---
# For Streamlit deployment, GEMINI_API_KEY should be passed as an environment variable or Streamlit secret.
# Using st.secrets is recommended for cloud deployments like Streamlit Community Cloud.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", st.secrets.get("GEMINI_API_KEY"))
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found. Please set it as an environment variable or Streamlit secret.")
    st.stop()

MODEL_NAME = "gemini-2.5-flash" # Use the model that was successfully configured earlier

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

# --- Database setup ---
# In a real deployment, the DB would be hosted externally. For this Colab context,
# we assume the PostgreSQL setup is handled outside this app.py and is accessible.
try:
    # Attempt to connect to the local PostgreSQL instance started by Colab
    engine = create_engine("postgresql+psycopg2://postgres:postgres@localhost:5432/miniproject")
    with engine.connect() as conn:
        conn.execute(text("SELECT 1")) # Test connection
    st.success("Successfully connected to PostgreSQL database.")
except Exception as e:
    st.error(f"Error connecting to database: {e}. Please ensure PostgreSQL is running and accessible.")
    engine = None # Set engine to None to prevent further DB operations

# --- Schema Definition ---
SCHEMA_STR = """employees(nip, nama, divisi, jabatan, join_date)
trainings(training_id, nama_diklat, tanggal)
enrollments(nip, training_id, status, nilai)

Relasi:
- enrollments.nip      -> employees.nip
- enrollments.training_id -> trainings.training_id
Catatan: enrollments.nilai bisa kosong (NULL) jika status = 'berjalan'."""

# --- Helper Functions (adapted for Streamlit) ---

# TODO 2: build_prompt
def build_prompt(question: str) -> str:
    prompt = f"""
Anda adalah seorang analis data yang membantu pengguna untuk mendapatkan informasi dari database PostgreSQL. Tugas Anda adalah mengubah pertanyaan dalam bahasa natural menjadi query SQL yang valid. Anda harus SELALU mengembalikan HANYA query SQL, TANPA penjelasan, tanpa markdown code block, tanpa teks tambahan lainnya. Jika Anda tidak bisa membuat query SQL yang relevan, kembalikan string kosong.

Berikut adalah skema tabel database:
{SCHEMA_STR}

Pertanyaan: {question}
SQL:"""
    return prompt

# TODO 3: generate_sql
def generate_sql(question: str) -> str:
    try:
        prompt = build_prompt(question)
        resp = model.generate_content(prompt)
        # Check if the response contains any text content
        if resp and resp.candidates and len(resp.candidates) > 0 and resp.candidates[0].content and resp.candidates[0].content.parts and len(resp.candidates[0].content.parts) > 0:
            sql = resp.candidates[0].content.parts[0].text.strip()
            # Clean up markdown code block if present
            if sql.startswith("```sql") and sql.endswith("```"):
                sql = sql[len("```sql"):-len("```")].strip()
            elif sql.startswith("```") and sql.endswith("```"):
                sql = sql[len("```"):-len("```")].strip()
            return sql
        else:
            # Log the reason for no content, e.g., safety block
            if resp and resp.prompt_feedback and resp.prompt_feedback.block_reason:
                st.warning(f"Model response blocked for question: '{question}' due to: {resp.prompt_feedback.block_reason}. Returning empty SQL.")
            else:
                st.warning(f"Model did not return any text content for question: '{question}'. Returning empty SQL.")
            return ""
    except Exception as e:
        st.error(f"Error generating SQL: {e}. Returning empty SQL.")
        return ""

# TODO 4: validate_sql
FORBIDDEN = ["drop", "delete", "update", "insert", "alter", "truncate", "create", "grant"]
def validate_sql(sql: str) -> bool:
    sql_lower = sql.strip().lower()
    if not sql_lower:
        return False
    if not sql_lower.startswith("select"):
        return False
    for keyword in FORBIDDEN:
        if keyword in sql_lower:
            return False
    if ";" in sql_lower[:-1]:
        return False
    return True

# run_sql
def run_sql(sql: str) -> pd.DataFrame:
    if engine is None:
        raise ConnectionError("Database engine not initialized or connected.")
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)

# TODO 5: visualize (modified to return plot figure)
def visualize(df: pd.DataFrame):
    if df.empty:
        return None # No data, no plot

    fig = None
    # Detect if there are 2 columns and one is numeric for a bar chart
    if len(df.columns) == 2:
        col1, col2 = df.columns
        if pd.api.types.is_numeric_dtype(df[col2]) and not pd.api.types.is_numeric_dtype(df[col1]):
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(df[col1], df[col2])
            ax.set_xlabel(col1.replace('_', ' ').title())
            ax.set_ylabel(col2.replace('_', ' ').title())
            ax.set_title(f'Bar Chart of {col2.replace("_", " ").title()} by {col1.replace("_", " ").title()}')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            return fig
        elif pd.api.types.is_numeric_dtype(df[col1]) and not pd.api.types.is_numeric_dtype(df[col2]):
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(df[col2], df[col1])
            ax.set_xlabel(col2.replace('_', ' ').title())
            ax.set_ylabel(col1.replace('_', ' ').title())
            ax.set_title(f'Bar Chart of {col1.replace("_", " ").title()} by {col2.replace("_", " ").title()}')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            return fig
    return None # If no suitable plot, return None

# TODO 6: ask (modified to return structured output for Streamlit)
def ask(question: str):
    response_parts = []
    df_result = None
    plot_fig = None
    sql_query = ""

    response_parts.append(f"Pertanyaan: {question}")

    # First attempt
    sql = generate_sql(question)
    sql_query = sql
    response_parts.append(f"SQL (percobaan 1): {sql}")

    if not validate_sql(sql):
        response_parts.append("SQL tidak valid, mencoba lagi...")
        # Second attempt if the first fails
        sql = generate_sql(question)
        sql_query = sql
        response_parts.append(f"SQL (percobaan 2): {sql}")
        if not validate_sql(sql):
            response_parts.append("Gagal menghasilkan SQL yang valid setelah 2 percobaan. Mohon periksa pertanyaan atau skema.")
            return {"messages": response_parts, "df": None, "plot": None, "sql": sql_query}

    try:
        df_result = run_sql(sql)
        response_parts.append("SQL berhasil dieksekusi.")
        plot_fig = visualize(df_result)
    except ConnectionError as ce:
        response_parts.append(f"Error koneksi database: {ce}")
        response_parts.append("Pastikan database aktif dan terhubung dengan benar.")
    except Exception as e:
        response_parts.append(f"Error saat menjalankan SQL: {e}")
        response_parts.append("Gagal mengambil data. Mohon periksa pertanyaan atau skema database.")

    return {"messages": response_parts, "df": df_result, "plot": plot_fig, "sql": sql}


# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("Human Capital Analytics Bot")
st.write("Ajukan pertanyaan tentang data karyawan, pelatihan, dan pendaftaran dalam bahasa natural.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["type"] == "text":
            st.markdown(message["content"])
        elif message["type"] == "dataframe":
            st.subheader("Query Result:")
            st.dataframe(message["content"])
        elif message["type"] == "plot":
            st.subheader("Visualization:")
            st.pyplot(message["content"])
            plt.close(message["content"]) # Close the plot after displaying to save memory
        elif message["type"] == "sql":
            st.subheader("Generated SQL Query:")
            st.code(message["content"], language="sql")


if prompt := st.chat_input("Tanyakan sesuatu..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "type": "text", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Mencari jawaban..."):
        # Get assistant response
        assistant_output = ask(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            for msg_text in assistant_output["messages"]:
                st.write(msg_text) # Display text messages

            if assistant_output["sql"]:
                # Avoid duplicating SQL in session state if it's already part of the initial messages
                # This check ensures it's only added once, typically after all text messages
                if not any(m["type"] == "sql" and m["content"] == assistant_output["sql"] for m in st.session_state.messages):
                    st.subheader("Generated SQL Query:")
                    st.code(assistant_output["sql"], language="sql")
                    st.session_state.messages.append({"role": "assistant", "type": "sql", "content": assistant_output["sql"]})

            if assistant_output["df"] is not None and not assistant_output["df"].empty:
                # Only show dataframe if no plot was generated, or if the plot didn't cover all info
                if assistant_output["plot"] is None:
                    st.subheader("Query Result:")
                    st.dataframe(assistant_output["df"])
                    st.session_state.messages.append({"role": "assistant", "type": "dataframe", "content": assistant_output["df"]})

            if assistant_output["plot"] is not None:
                st.subheader("Visualization:")
                st.pyplot(assistant_output["plot"])
                st.session_state.messages.append({"role": "assistant", "type": "plot", "content": assistant_output["plot"]})
                plt.close(assistant_output["plot"]) # Close the plot to free up memory


# Instructions to run the Streamlit app from Colab
st.markdown("---")
st.markdown("**To run this Streamlit app in Colab:**")
st.code("""
!pip -q install streamlit pyngrok
!streamlit run app.py &>/dev/null &
from pyngrok import ngrok
public_url = ngrok.connect(8501)
print(public_url)
""", language="bash")

print("TODO 8 (opsional): kerjakan jika waktu masih cukup.")
