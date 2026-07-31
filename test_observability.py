import os
import unittest
from unittest.mock import patch

from flask import Flask

import observability


class ObservabilityTests(unittest.TestCase):
    def make_app(self):
        app = Flask(__name__)

        @app.get("/hello/<name>")
        def hello(name):
            return {"hello": name}

        observability.init_flask_observability(app)
        return app

    def test_metrics_endpoint_and_normalized_http_labels(self):
        client = self.make_app().test_client()
        response = client.get("/hello/cinebot", headers={"X-Request-ID": "test-123"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Request-ID"], "test-123")

        metrics = client.get("/metrics")
        body = metrics.get_data(as_text=True)
        self.assertEqual(metrics.status_code, 200)
        self.assertIn("cinebot_http_requests_total", body)
        self.assertIn('route="/hello/<name>"', body)
        self.assertNotIn('route="/hello/cinebot"', body)

    def test_metrics_bearer_token(self):
        client = self.make_app().test_client()
        with patch.dict(os.environ, {"METRICS_BEARER_TOKEN": "secret"}):
            self.assertEqual(client.get("/metrics").status_code, 401)
            response = client.get(
                "/metrics",
                headers={"Authorization": "Bearer secret"},
            )
            self.assertEqual(response.status_code, 200)

    def test_application_metrics_do_not_expose_user_labels(self):
        observability.observe_performance_metric({
            "channel": "api",
            "status": "success",
            "outcome": "islem_basarili",
            "user_id": "private-user",
            "username": "private-name",
            "e2e_ms": 1250,
        })
        body = observability.generate_latest().decode("utf-8")
        self.assertIn("cinebot_performance_records_total", body)
        self.assertIn("cinebot_pipeline_stage_duration_seconds", body)
        self.assertNotIn("private-user", body)
        self.assertNotIn("private-name", body)


if __name__ == "__main__":
    unittest.main()

