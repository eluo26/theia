"""Local demo accounts and revocable, cookie-based sessions."""
import hashlib
import re
import secrets
import sqlite3
import time
from pathlib import Path

from flask import g, jsonify, request
from werkzeug.security import check_password_hash, generate_password_hash

COOKIE = "theia_session"
SESSION_SECONDS = 7 * 24 * 60 * 60
LOCAL_VITE_ORIGINS = {"http://localhost:5173", "http://127.0.0.1:5173"}


def init_auth(app):
    app.config.setdefault("AUTH_DATABASE", Path(__file__).parent / "data" / "auth.sqlite3")

    def connect():
        path = Path(app.config["AUTH_DATABASE"])
        path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(path)
        db.row_factory = sqlite3.Row
        db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('patient', 'staff'))
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id),
                expires_at REAL NOT NULL
            );
        """)
        return db

    def token_hash():
        return hashlib.sha256(request.cookies.get(COOKIE, "").encode()).hexdigest()

    def public_user(row):
        return {key: row[key] for key in ("id", "email", "role")}

    def credentials():
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return None
        email, password, role = (body.get(key) for key in ("email", "password", "role"))
        if not all(isinstance(value, str) for value in (email, password, role)):
            return None
        email = email.strip().lower()
        if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email) or len(email) > 254:
            return None
        if role not in ("patient", "staff") or not 8 <= len(password) <= 128:
            return None
        return email, password, role

    def signed_in(db, user, status=200):
        token = secrets.token_urlsafe(32)
        db.execute("DELETE FROM sessions WHERE token_hash = ? OR expires_at <= ?", (token_hash(), time.time()))
        db.execute("INSERT INTO sessions VALUES (?, ?, ?)",
                   (hashlib.sha256(token.encode()).hexdigest(), user["id"], time.time() + SESSION_SECONDS))
        response = jsonify(user=public_user(user))
        response.status_code = status
        response.set_cookie(COOKIE, token, max_age=SESSION_SECONDS,
                            httponly=True, samesite="Lax", secure=request.is_secure)
        return response

    @app.before_request
    def authorize():
        if not request.path.startswith("/api/"):
            return None
        # Vite forwards the browser's Origin (port 5173) while Flask sees port 5000.
        # Accept only that local development proxy and actual same-origin writes.
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("Origin")
            local_vite = (origin in LOCAL_VITE_ORIGINS and
                          request.remote_addr in ("127.0.0.1", "::1") and
                          request.host in ("127.0.0.1:5000", "localhost:5000"))
            if origin and origin.rstrip("/") != request.host_url.rstrip("/") and not local_vite:
                return jsonify(error="Cross-origin requests are not allowed."), 403
            if request.headers.get("Sec-Fetch-Site") == "cross-site" and not local_vite:
                return jsonify(error="Cross-origin requests are not allowed."), 403
        g.user = None
        if request.cookies.get(COOKIE):
            db = connect()
            try:
                row = db.execute("""SELECT users.* FROM sessions JOIN users ON users.id = sessions.user_id
                                    WHERE token_hash = ? AND expires_at > ?""",
                                 (token_hash(), time.time())).fetchone()
                if row:
                    g.user = public_user(row)
            finally:
                db.close()
        if request.endpoint in ("register", "login", "current_user", "logout"):
            return None
        if g.user is None:
            return jsonify(error="Please log in to continue."), 401
        if (request.path.startswith("/api/stock/") or request.path in
                ("/api/catalog", "/api/camera/preview")) and g.user["role"] != "staff":
            return jsonify(error="This feature is available to staff accounts only."), 403

    @app.after_request
    def private_responses(response):
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.post("/api/auth/register")
    def register():
        values = credentials()
        if values is None:
            return jsonify(error="Enter a valid email, an 8–128 character password, and an account type."), 400
        email, password, role = values
        db = connect()
        try:
            with db:
                try:
                    cursor = db.execute("INSERT INTO users (email, password_hash, role) VALUES (?, ?, ?)",
                                        (email, generate_password_hash(password), role))
                except sqlite3.IntegrityError:
                    return jsonify(error="An account with this email already exists. Log in instead."), 409
                return signed_in(db, {"id": cursor.lastrowid, "email": email, "role": role}, 201)
        finally:
            db.close()

    @app.post("/api/auth/login")
    def login():
        values = credentials()
        if values is None:
            return jsonify(error="Check your email, password, and account type."), 401
        email, password, role = values
        db = connect()
        try:
            with db:
                user = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
                if not user or not check_password_hash(user["password_hash"], password) or user["role"] != role:
                    return jsonify(error="Check your email, password, and account type."), 401
                return signed_in(db, user)
        finally:
            db.close()

    @app.get("/api/auth/me")
    def current_user():
        return jsonify(user=g.user)

    @app.post("/api/auth/logout")
    def logout():
        db = connect()
        try:
            with db:
                db.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash(),))
        finally:
            db.close()
        response = jsonify(user=None)
        response.delete_cookie(COOKIE, httponly=True, samesite="Lax")
        return response
