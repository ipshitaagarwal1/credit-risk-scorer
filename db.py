"""
Database layer - SQLite-backed user accounts and assessment history.

NOTE: passwords are hashed with SHA-256 for this portfolio project.
For production, use bcrypt/argon2 with per-user salts instead.
"""

import sqlite3
import hashlib
import os
import pandas as pd

DB_PATH = "outputs/app_data.db"


def get_connection():
    os.makedirs("outputs", exist_ok=True)
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            age INTEGER,
            annual_income REAL,
            credit_score INTEGER,
            loan_amount REAL,
            employment_years REAL,
            debt_to_income REAL,
            num_open_accounts INTEGER,
            num_late_payments_2y INTEGER,
            home_ownership TEXT,
            loan_purpose TEXT,
            loan_term_months INTEGER,
            risk_probability REAL,
            decision TEXT
        )
    """)
    conn.commit()
    conn.close()


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def create_user(username: str, password: str):
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, hash_password(password))
        )
        conn.commit()
        return True, "Account created successfully. Please log in."
    except sqlite3.IntegrityError:
        return False, "That username is already taken."
    finally:
        conn.close()


def verify_user(username: str, password: str) -> bool:
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()
    return bool(row) and row[0] == hash_password(password)


def save_assessment(username: str, inputs: dict, prob: float, decision: str):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO assessments
            (username, age, annual_income, credit_score, loan_amount, employment_years,
             debt_to_income, num_open_accounts, num_late_payments_2y, home_ownership,
             loan_purpose, loan_term_months, risk_probability, decision)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        username, inputs["age"], inputs["annual_income"], inputs["credit_score"],
        inputs["loan_amount"], inputs["employment_years"], inputs["debt_to_income"],
        inputs["num_open_accounts"], inputs["num_late_payments_2y"],
        inputs["home_ownership"], inputs["loan_purpose"], inputs["loan_term_months"],
        prob, decision
    ))
    conn.commit()
    conn.close()


def get_user_history(username: str) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM assessments WHERE username = ? ORDER BY timestamp DESC",
        conn, params=(username,)
    )
    conn.close()
    return df
