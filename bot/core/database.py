import sqlite3
from datetime import datetime, timezone


class Database:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT NOT NULL,
                side TEXT NOT NULL,
                price REAL NOT NULL,
                size REAL NOT NULL,
                edge REAL NOT NULL,
                truth_source TEXT NOT NULL,
                truth_probability REAL NOT NULL,
                outcome TEXT,
                payout REAL,
                created_at TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                market_id TEXT PRIMARY KEY,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                size REAL NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_pnl (
                date TEXT PRIMARY KEY,
                pnl REAL NOT NULL,
                trades_count INTEGER NOT NULL,
                wins INTEGER NOT NULL,
                losses INTEGER NOT NULL
            )
        """)
        self.conn.commit()

    def list_tables(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
        return [row["name"] for row in cursor.fetchall()]

    def log_trade(self, market_id, side, price, size, edge, truth_source, truth_probability):
        cursor = self.conn.cursor()
        now = datetime.now(timezone.utc).isoformat()
        cursor.execute(
            """INSERT INTO trades (market_id, side, price, size, edge, truth_source, truth_probability, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (market_id, side, price, size, edge, truth_source, truth_probability, now),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_trades(self, limit=100):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM trades ORDER BY created_at DESC LIMIT ?", (limit,))
        return [dict(row) for row in cursor.fetchall()]

    def settle_trade(self, market_id, outcome, payout):
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE trades SET outcome = ?, payout = ? WHERE market_id = ? AND outcome IS NULL",
            (outcome, payout, market_id),
        )
        self.conn.commit()
        return cursor.rowcount

    def get_daily_pnl(self):
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT COALESCE(SUM(payout - size), 0) as pnl FROM trades WHERE outcome IS NOT NULL"
        )
        return cursor.fetchone()["pnl"]

    def get_open_positions_count(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM trades WHERE outcome IS NULL")
        return cursor.fetchone()["count"]
