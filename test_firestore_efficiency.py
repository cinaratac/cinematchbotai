import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import ai_service
import database


class FirestoreEfficiencyTests(unittest.TestCase):
    def setUp(self):
        database._past_summary_cache.clear()
        database._pagination_cursor_cache.clear()
        database._performance_docs_cache.clear()
        database._unique_user_cache["count"] = None
        database._unique_user_cache["expires_at"] = 0

    @patch("database._get_db")
    def test_touch_session_does_not_read_document_after_update(self, get_db):
        ref = MagicMock()
        get_db.return_value.collection.return_value.document.return_value = ref

        count = database.touch_session(
            "session-1",
            {"intent": "film_onerisi_istendi", "outcome": "islem_basarili"},
            current_message_count=7,
            summary_text="Güncel özet",
            summary_message_count=8,
        )

        self.assertEqual(count, 8)
        ref.update.assert_called_once()
        updates = ref.update.call_args.args[0]
        self.assertEqual(updates["summary"], "Güncel özet")
        self.assertEqual(updates["summary_message_count"], 8)
        ref.get.assert_not_called()

    @patch("database.get_user_facts")
    @patch("database._get_db")
    def test_profile_update_reuses_already_read_facts(
        self,
        get_db,
        get_user_facts,
    ):
        ref = MagicMock()
        get_db.return_value.collection.return_value.document.return_value = ref

        result = database.update_user_facts(
            "user-1",
            "Ada",
            {"isim": "Ada"},
            existing_facts={"favori_tur": "Bilim kurgu"},
        )

        get_user_facts.assert_not_called()
        self.assertEqual(
            result,
            {"favori_tur": "Bilim kurgu", "isim": "Ada"},
        )
        ref.set.assert_called_once()

    def test_old_session_summary_marker_is_inferred(self):
        doc = SimpleNamespace(id="session-1")
        state = database._session_state(
            doc,
            {
                "message_count": 11,
                "summary": "Önceki konuşmanın özeti.",
            },
        )

        self.assertEqual(state["summary_message_count"], 4)

    def test_sequential_admin_page_uses_cached_cursor(self):
        docs = [SimpleNamespace(id=f"doc-{index}") for index in range(4)]

        class FakeQuery:
            def __init__(self, rows):
                self.rows = rows
                self.start_index = 0
                self.limit_value = len(rows)
                self.start_after_calls = 0

            def start_after(self, snapshot):
                self.start_after_calls += 1
                self.start_index = self.rows.index(snapshot) + 1
                return self

            def limit(self, value):
                self.limit_value = value
                return self

            def stream(self):
                end = self.start_index + self.limit_value
                return iter(self.rows[self.start_index:end])

        first_query = FakeQuery(docs)
        first_page = database._query_page_with_cursor(
            first_query,
            cache_key="test",
            limit=2,
            offset=0,
        )
        second_query = FakeQuery(docs)
        second_page = database._query_page_with_cursor(
            second_query,
            cache_key="test",
            limit=2,
            offset=2,
        )

        self.assertEqual([doc.id for doc in first_page], ["doc-0", "doc-1"])
        self.assertEqual([doc.id for doc in second_page], ["doc-2", "doc-3"])
        self.assertEqual(second_query.start_after_calls, 1)

    @patch("ai_service._call_openrouter")
    @patch("ai_service.get_session_transcript_recent")
    def test_incremental_summary_reads_only_unsummarized_turns(
        self,
        get_recent,
        call_openrouter,
    ):
        get_recent.return_value = [
            {"user_message": "Yeni soru", "bot_response": "Yeni cevap"}
        ]
        call_openrouter.return_value = {
            "choices": [{"message": {"content": "Güncel özet"}}]
        }

        summary = ai_service.summarize_session(
            "session-1",
            previous_summary="Eski özet",
            message_count=8,
            summary_message_count=4,
        )

        get_recent.assert_called_once_with("session-1", 4)
        self.assertEqual(summary, "Güncel özet")

    @patch("ai_service._call_openrouter")
    @patch("ai_service.get_session_transcript_recent")
    def test_incremental_summary_reuses_turns_already_in_memory(
        self,
        get_recent,
        call_openrouter,
    ):
        call_openrouter.return_value = {
            "choices": [{"message": {"content": "Güncel özet"}}]
        }
        turns = [
            {"user_message": f"Soru {index}", "bot_response": f"Cevap {index}"}
            for index in range(1, 5)
        ]

        ai_service.summarize_session(
            "session-1",
            previous_summary="Eski özet",
            message_count=8,
            summary_message_count=4,
            new_turns=turns,
        )

        get_recent.assert_not_called()

    @patch("database.random.random", return_value=0.9)
    @patch("database._get_db")
    def test_success_metrics_are_sampled_but_errors_are_kept(
        self,
        get_db,
        _random,
    ):
        with patch.dict(
            os.environ,
            {"PERFORMANCE_METRIC_SUCCESS_SAMPLE_RATE": "0.25"},
        ):
            skipped = database.log_performance_metric({
                "status": "success",
                "channel": "api",
                "input_type": "text",
            })
            self.assertIsNone(skipped)
            get_db.assert_not_called()

            database.log_performance_metric({
                "status": "error",
                "channel": "api",
                "input_type": "text",
            })
            get_db.assert_called_once()

    @patch("database.storage.bucket")
    @patch("database._get_db")
    def test_successful_voice_recording_uses_one_metadata_write(
        self,
        get_db,
        storage_bucket,
    ):
        ref = MagicMock()
        get_db.return_value.collection.return_value.document.return_value = ref
        storage_bucket.return_value.blob.return_value = MagicMock()

        with patch.dict(
            os.environ,
            {"FIREBASE_STORAGE_BUCKET": "test-bucket"},
        ):
            database.save_voice_recording(
                "session-1",
                "user-1",
                "Ada",
                "recording-1",
                "/tmp/user.wav",
                "/tmp/agent.wav",
                1000,
                900,
            )

        ref.set.assert_called_once()
        ref.update.assert_not_called()
        payload = ref.set.call_args.args[0]
        self.assertEqual(payload["status"], "ready")

    @patch("database.count_performance_metrics_admin", return_value=2)
    @patch("database._get_performance_metric_docs")
    def test_performance_bundle_reuses_one_document_query(
        self,
        get_docs,
        _count,
    ):
        docs = []
        for index in range(2):
            doc = MagicMock()
            doc.id = f"metric-{index}"
            doc.to_dict.return_value = {
                "status": "success",
                "measurement_valid": True,
                "channel": "api",
                "input_type": "text",
                "ai_ms": 100,
                "ttfb_ms": 120,
                "e2e_ms": 130,
                "created_at": database._now(),
            }
            docs.append(doc)
        get_docs.return_value = docs

        bundle = database.get_performance_metrics_bundle(
            limit=1,
            sample_size=1,
        )
        cached_bundle = database.get_performance_metrics_bundle(
            limit=1,
            sample_size=1,
        )

        get_docs.assert_called_once()
        self.assertEqual(len(bundle["data"]), 1)
        self.assertEqual(bundle["averages"]["_sample_count"], 1)
        self.assertEqual(bundle["total"], 2)
        self.assertEqual(cached_bundle["total"], 2)


if __name__ == "__main__":
    unittest.main()
