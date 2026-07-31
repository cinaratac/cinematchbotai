import unittest
import os
from unittest.mock import patch

from flask import Flask

import admin_api


class AdminReportTests(unittest.TestCase):
    def setUp(self):
        app = Flask(__name__)
        app.register_blueprint(admin_api.admin_bp)
        self.client = app.test_client()
        self.key_patch = patch.object(admin_api, "ADMIN_API_KEY", "test-admin-key")
        self.key_patch.start()
        self.headers = {"X-Admin-Key": "test-admin-key"}

    def tearDown(self):
        self.key_patch.stop()

    def test_reports_require_admin_key(self):
        response = self.client.get("/api/admin/reports/overview.csv")
        self.assertEqual(response.status_code, 401)

    def test_monitoring_config_only_returns_valid_http_url(self):
        with patch.dict(os.environ, {
            "GRAFANA_DASHBOARD_URL": "https://grafana.example/d/cinebot",
            "METRICS_BEARER_TOKEN": "not-returned",
        }):
            response = self.client.get(
                "/api/admin/monitoring", headers=self.headers
            )
        self.assertEqual(response.status_code, 200)
        data = response.json["data"]
        self.assertTrue(data["grafana_configured"])
        self.assertEqual(
            data["grafana_dashboard_url"],
            "https://grafana.example/d/cinebot",
        )
        self.assertTrue(data["metrics_auth_enabled"])
        self.assertNotIn("not-returned", response.get_data(as_text=True))

    @patch.object(admin_api.db, "get_admin_overview")
    def test_overview_csv_is_excel_compatible_and_formula_safe(self, overview):
        overview.return_value = {
            "total_sessions": 2,
            "active_sessions": 1,
            "total_messages": 4,
            "total_users": 2,
            "total_tool_calls": 1,
            "successful_tool_calls": 1,
            "failed_tool_calls": 0,
            "avg_rating": 4.5,
            "total_evaluations": 2,
            "classified_messages": 4,
            "success_rate": 75,
            "fallback_rate": 25,
            "technical_error_rate": 0,
            "daily_messages": [{"day": "2026-07-31", "c": 4}],
            "intent_distribution": [{"intent": "film_onerisi_istendi", "count": 4}],
            "outcome_distribution": [{"outcome": "islem_basarili", "count": 3}],
            "rating_distribution": [{"rating": 5, "c": 1}],
            "top_movies": [{
                "movie_name": "=HYPERLINK(\"bad\")",
                "c": 1,
                "avg_duration_ms": 120,
            }],
        }
        response = self.client.get(
            "/api/admin/reports/overview.csv?days=30",
            headers=self.headers,
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.data.startswith(b"\xef\xbb\xbf"))
        self.assertIn("attachment", response.headers["Content-Disposition"])
        text = response.data.decode("utf-8-sig")
        self.assertIn("Toplam oturum", text)
        self.assertIn("'=HYPERLINK", text)
        overview.assert_called_once_with(days=30)

    @patch.object(admin_api.db, "get_performance_metrics_export_admin")
    def test_performance_csv_uses_requested_period(self, export):
        export.return_value = [{
            "id": "metric-1",
            "created_at": "2026-07-31T10:00:00+00:00",
            "channel": "api",
            "status": "success",
            "e2e_ms": 1250,
        }]
        response = self.client.get(
            "/api/admin/reports/performance.csv?days=90&limit=100",
            headers=self.headers,
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("metric-1", response.data.decode("utf-8-sig"))
        export.assert_called_once_with(days=90, limit=100, session_id=None)


if __name__ == "__main__":
    unittest.main()
