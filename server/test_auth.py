"""Run with python -m unittest discover -s server -p 'test_*.py'."""
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from app import app
from auth import COOKIE


class PatientAuthTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.database = Path(self.directory.name) / "accounts.sqlite3"
        self.config = patch.dict(app.config, TESTING=True, AUTH_DATABASE=self.database)
        self.config.start()
        self.client = app.test_client()

    def tearDown(self):
        self.config.stop()
        self.directory.cleanup()

    def register(self, email="person@example.com", client=None, **extra):
        return (client or self.client).post("/api/auth/register", json={
            "email": email, "password": "test-password", **extra,
        })

    def test_patient_registration_and_session(self):
        response = self.register(email=" PERSON@example.com ")
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json["user"]["email"], "person@example.com")
        self.assertEqual(response.json["user"]["role"], "patient")
        self.assertIn("HttpOnly", response.headers["Set-Cookie"])
        self.assertEqual(self.client.get("/api/auth/me").json, response.json)
        with closing(sqlite3.connect(self.database)) as db:
            self.assertNotEqual(db.execute("SELECT password_hash FROM users").fetchone()[0], "test-password")
        self.assertEqual(self.register().status_code, 409)

    def test_login_logout_and_expiry(self):
        self.register()
        old_cookie = self.client.get_cookie(COOKIE).value
        self.assertEqual(self.client.post("/api/auth/logout").status_code, 200)
        self.client.set_cookie(COOKIE, old_cookie)
        self.assertIsNone(self.client.get("/api/auth/me").json["user"])
        self.assertEqual(self.client.post("/api/auth/login", json={
            "email": "person@example.com", "password": "wrong-password",
        }).status_code, 401)
        self.assertEqual(self.client.post("/api/auth/login", json={
            "email": "PERSON@example.com", "password": "test-password",
        }).status_code, 200)
        with closing(sqlite3.connect(self.database)) as db:
            db.execute("UPDATE sessions SET expires_at = 0")
            db.commit()
        self.assertIsNone(self.client.get("/api/auth/me").json["user"])

    def test_only_patient_accounts_can_use_main(self):
        with closing(sqlite3.connect(self.database)) as db:
            db.execute("""CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL,
                       password_hash TEXT NOT NULL, role TEXT NOT NULL CHECK(role IN ('patient', 'staff')))""")
            db.commit()
        self.assertEqual(self.register(email="staff@example.com", role="staff").status_code, 400)
        self.register()
        with closing(sqlite3.connect(self.database)) as db:
            # The local SQLite file may contain old staff rows from the warehouse branch.
            db.execute("INSERT INTO users (email, password_hash, role) SELECT ?, password_hash, ? FROM users LIMIT 1",
                       ("old-staff@example.com", "staff"))
            db.commit()
        staff = app.test_client()
        self.assertEqual(staff.post("/api/auth/login", json={
            "email": "old-staff@example.com", "password": "test-password",
        }).status_code, 401)
        for method, path in (("GET", "/api/catalog"), ("POST", "/api/stock/scan"),
                             ("POST", "/api/stock/confirm"), ("GET", "/api/camera/preview")):
            self.assertEqual(self.client.open(path, method=method).status_code, 404)

    def test_search_is_patient_scoped(self):
        self.assertEqual(self.client.post("/api/search", json={"query": "keys"}).status_code, 401)
        self.register()
        with patch("app.threading.Thread"):
            response = self.client.post("/api/search", json={"query": "keys"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.get(f'/api/search/{response.json["id"]}').status_code, 200)
        other = app.test_client()
        self.register(email="other@example.com", client=other)
        self.assertEqual(other.get(f'/api/search/{response.json["id"]}').status_code, 404)

    def test_vite_proxy_and_cross_origin(self):
        response = self.client.post("/api/auth/register", base_url="http://127.0.0.1:5000",
                                    environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
                                    headers={"Origin": "http://localhost:5173"},
                                    json={"email": "vite@example.com", "password": "test-password"})
        self.assertEqual(response.status_code, 201)
        self.assertEqual(self.client.post("/api/auth/logout", headers={"Origin": "https://other.example"}).status_code, 403)
        # The proxy may target another local Flask port (API_PORT).
        self.assertNotEqual(self.client.post("/api/auth/logout", base_url="http://127.0.0.1:5001",
                                             headers={"Origin": "http://localhost:5173"}).status_code, 403)


if __name__ == "__main__":
    unittest.main()
