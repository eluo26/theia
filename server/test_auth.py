"""Run with python -m unittest discover -s server -p 'test_*.py'."""
import sqlite3
import tempfile
import unittest
import io
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import httpx
from openai import BadRequestError

from app import app
from auth import COOKIE


class AuthTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.database = Path(self.directory.name) / "accounts.sqlite3"
        self.config = patch.dict(app.config, TESTING=True, AUTH_DATABASE=self.database)
        self.config.start()
        self.client = app.test_client()

    def tearDown(self):
        self.config.stop()
        self.directory.cleanup()

    def register(self, role="patient", email="person@example.com", client=None):
        return (client or self.client).post("/api/auth/register", json={
            "email": email, "password": "test-password", "role": role,
        })

    def test_register_session_and_password_storage(self):
        response = self.register(email=" PERSON@example.com ")
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json["user"]["email"], "person@example.com")
        self.assertNotIn("password_hash", response.json["user"])
        self.assertIn("HttpOnly", response.headers["Set-Cookie"])
        self.assertIn("SameSite=Lax", response.headers["Set-Cookie"])
        self.assertEqual(self.client.get("/api/auth/me").json, response.json)
        with closing(sqlite3.connect(self.database)) as db:
            password_hash = db.execute("SELECT password_hash FROM users").fetchone()[0]
        self.assertNotEqual(password_hash, "test-password")
        self.assertEqual(self.register().status_code, 409)

    def test_vite_proxy_registration_origin(self):
        for index, origin in enumerate(("http://localhost:5173", "http://127.0.0.1:5173")):
            with self.subTest(origin=origin):
                client = app.test_client()
                response = client.post("/api/auth/register", base_url="http://127.0.0.1:5000",
                                       environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
                                       headers={"Origin": origin, "Sec-Fetch-Site": "same-origin"},
                                       json={"email": f"vite{index}@example.com", "password": "test-password", "role": "staff"})
                self.assertEqual(response.status_code, 201)
        denied = app.test_client().post("/api/auth/register", base_url="http://127.0.0.1:5000",
                                       environ_overrides={"REMOTE_ADDR": "203.0.113.10"},
                                       headers={"Origin": "http://localhost:5173"},
                                       json={"email": "remote@example.com", "password": "test-password", "role": "staff"})
        self.assertEqual(denied.status_code, 403)

    def test_logout_revokes_cookie_and_login_restores_saved_role(self):
        self.register("staff")
        old_cookie = self.client.get_cookie(COOKIE).value
        self.assertEqual(self.client.post("/api/auth/logout").status_code, 200)
        self.client.set_cookie(COOKIE, old_cookie)
        self.assertIsNone(self.client.get("/api/auth/me").json["user"])
        for password, role in (("wrong-password", "staff"), ("test-password", "patient")):
            self.assertEqual(self.client.post("/api/auth/login", json={
                "email": "person@example.com", "password": password, "role": role,
            }).status_code, 401)
        response = self.client.post("/api/auth/login", json={
            "email": "PERSON@example.com", "password": "test-password", "role": "staff",
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["user"]["role"], "staff")

    def test_unauthenticated_and_patient_access(self):
        protected = [("GET", "/api/catalog"), ("GET", "/api/camera/preview"),
                     ("POST", "/api/stock/scan"), ("POST", "/api/stock/confirm"),
                     ("GET", "/api/stock/scans/123.jpg")]
        for method, path in protected + [("POST", "/api/search"), ("GET", "/api/search/1")]:
            self.assertEqual(self.client.open(path, method=method).status_code, 401)
        self.register()
        for method, path in protected:
            self.assertEqual(self.client.open(path, method=method).status_code, 403)
        with patch("app.threading.Thread"):
            response = self.client.post("/api/search", json={"query": "starter pack"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.get(f'/api/search/{response.json["id"]}').status_code, 200)
        other = app.test_client()
        self.register(email="other@example.com", client=other)
        self.assertEqual(other.get(f'/api/search/{response.json["id"]}').status_code, 404)

    def test_staff_access_to_both_scanners(self):
        self.register("staff")
        with patch("app.load_catalog", return_value=[]):
            self.assertEqual(self.client.get("/api/catalog").status_code, 200)
        with patch("app.threading.Thread"):
            self.assertEqual(self.client.post("/api/search", json={"query": "box"}).status_code, 200)
        with patch("app.add_stock", return_value={"count": 1}) as add:
            response = self.client.post("/api/stock/confirm", json={"drug_name": "Demo", "strength": "10 mg"})
            self.assertEqual(response.status_code, 200)
            add.assert_called_once()
        with patch("app.read_label", return_value={"fields": {}, "warnings": []}), \
                patch("app.load_catalog", return_value=[]), \
                patch("app.SCANS_DIR", Path(self.directory.name) / "scans"):
            _, jpeg = cv2.imencode(".jpg", np.zeros((2, 2, 3), dtype=np.uint8))
            response = self.client.post("/api/stock/scan", data={"photo": (io.BytesIO(jpeg.tobytes()), "box.jpg")})
            self.assertEqual(response.status_code, 200)
            with self.client.get(response.json["image_url"]) as image_response:
                self.assertEqual(image_response.status_code, 200)

    def test_upload_format_errors_are_readable(self):
        self.register("staff")
        heic = b"\x00\x00\x00\x28ftypheic" + b"\x00" * 20
        with patch("app.read_label") as read_label:
            response = self.client.post("/api/stock/scan", data={"photo": (io.BytesIO(heic), "box.heic")})
        self.assertEqual(response.status_code, 400)
        self.assertIn("JPEG or PNG", response.json["error"])
        self.assertNotIn("image_url", response.json)
        read_label.assert_not_called()

    def test_png_is_saved_as_jpeg_and_provider_json_is_unwrapped(self):
        self.register("staff")
        _, png = cv2.imencode(".png", np.zeros((2, 2, 3), dtype=np.uint8))
        response = httpx.Response(400, request=httpx.Request("POST", "https://example.invalid"))
        provider_error = BadRequestError("Bad Request", response=response, body={
            "error": {"message": "You uploaded an unsupported image.", "code": "invalid_image_format"}
        })
        with patch("app.read_label", side_effect=provider_error) as read_label, \
                patch("app.load_catalog", return_value=[]), \
                patch("app.SCANS_DIR", Path(self.directory.name) / "scans"):
            result = self.client.post("/api/stock/scan", data={"photo": (io.BytesIO(png.tobytes()), "box.png")})
            self.assertEqual(result.status_code, 502)
            self.assertEqual(result.json["error"], "This photo format isn't supported. Choose a JPEG or PNG image.")
            sent_bytes = read_label.call_args.args[0]
            self.assertTrue(sent_bytes.startswith(b"\xff\xd8\xff"))
            with self.client.get(result.json["image_url"]) as image_response:
                self.assertEqual(image_response.status_code, 200)
                self.assertTrue(image_response.data.startswith(b"\xff\xd8\xff"))

    def test_expired_and_forged_sessions(self):
        self.register()
        with closing(sqlite3.connect(self.database)) as db:
            db.execute("UPDATE sessions SET expires_at = 0")
            db.commit()
        self.assertEqual(self.client.post("/api/search").status_code, 401)
        self.client.set_cookie(COOKIE, "forged")
        self.assertIsNone(self.client.get("/api/auth/me").json["user"])

    def test_invalid_registration_and_cross_origin_writes(self):
        for body in ([], {}, {"email": "bad", "password": "test-password", "role": "staff"},
                     {"email": "a@b.com", "password": "short", "role": "patient"},
                     {"email": "a@b.com", "password": "test-password", "role": "admin"}):
            self.assertEqual(self.client.post("/api/auth/register", json=body).status_code, 400)
        self.assertEqual(self.client.post("/api/auth/register", headers={"Origin": "https://other.example"}).status_code, 403)
        self.assertEqual(self.client.post("/api/auth/logout", headers={"Sec-Fetch-Site": "cross-site"}).status_code, 403)
        response = self.client.post("/api/auth/register", base_url="http://localhost:5173",
                                    headers={"Origin": "http://localhost:5173"},
                                    json={"email": "a@b.com", "password": "test-password", "role": "patient"})
        self.assertEqual(response.status_code, 201)


if __name__ == "__main__":
    unittest.main()
