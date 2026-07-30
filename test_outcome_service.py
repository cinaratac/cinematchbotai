import unittest

from outcome_service import (
    INTENT_ACTOR_MOVIES,
    INTENT_ACTOR_INFO,
    INTENT_CATEGORY_MOVIES,
    INTENT_CINEMATCH_HELP,
    INTENT_DIRECTOR_MOVIES,
    INTENT_IMAGE_ANALYSIS,
    INTENT_MOVIE_RECOMMENDATION,
    INTENT_UNCLEAR,
    OUTCOME_FALLBACK,
    OUTCOME_OUT_OF_SCOPE,
    OUTCOME_PARTIAL_SUCCESS,
    OUTCOME_SUCCESS,
    OUTCOME_TECHNICAL_ERROR,
    categorize_interaction,
)


class CategorizeInteractionTests(unittest.TestCase):
    def test_movie_recommendation_success(self):
        result = categorize_interaction(
            "Bu akşam ne izlesem, bana bir film öner",
            "Arrival iyi bir seçim olur.",
            recommended_movies=["Arrival"],
        )
        self.assertEqual(result["intent"], INTENT_MOVIE_RECOMMENDATION)
        self.assertEqual(result["outcome"], OUTCOME_SUCCESS)

    def test_actor_movie_search(self):
        result = categorize_interaction(
            "Tom Hanks'in filmlerini öner",
            "Cast Away ve Big iyi seçenekler.",
        )
        self.assertEqual(result["intent"], INTENT_ACTOR_MOVIES)

    def test_named_actor_info(self):
        result = categorize_interaction(
            "Tom Hanks kimdir?",
            "Tom Hanks, iki Oscar ödüllü Amerikalı bir oyuncudur.",
        )
        self.assertEqual(result["intent"], INTENT_ACTOR_INFO)

    def test_english_actor_movie_search(self):
        result = categorize_interaction(
            "Recommend movies starring Tom Hanks",
            "Try Cast Away or Big.",
        )
        self.assertEqual(result["intent"], INTENT_ACTOR_MOVIES)

    def test_category_movie_search(self):
        result = categorize_interaction(
            "Bu akşam bilim kurgu filmi öner",
            "Arrival iyi bir seçim olur.",
            recommended_movies=["Arrival"],
        )
        self.assertEqual(result["intent"], INTENT_CATEGORY_MOVIES)

    def test_director_movie_search(self):
        result = categorize_interaction(
            "Christopher Nolan'ın yönettiği filmler hangileri?",
            "Inception ve The Prestige öne çıkıyor.",
        )
        self.assertEqual(result["intent"], INTENT_DIRECTOR_MOVIES)

    def test_app_help(self):
        result = categorize_interaction(
            "CineMatch'te kullanıcı adımı nasıl değiştiririm?",
            "Profil sekmesi üzerinden değiştirebilirsin.",
        )
        self.assertEqual(result["intent"], INTENT_CINEMATCH_HELP)

    def test_image_input_wins(self):
        result = categorize_interaction(
            "[FOTOĞRAF]",
            "Bu bir film afişine benziyor.",
            input_type="photo",
        )
        self.assertEqual(result["intent"], INTENT_IMAGE_ANALYSIS)

    def test_fallback_makes_unknown_request_explicit(self):
        result = categorize_interaction(
            "asdfgh",
            "Ne demek istediğini anlayamadım, tekrar eder misin?",
        )
        self.assertEqual(result["intent"], INTENT_UNCLEAR)
        self.assertEqual(result["outcome"], OUTCOME_FALLBACK)

    def test_error_signal_overrides_response(self):
        result = categorize_interaction(
            "Inception hakkında konuş",
            "Tekrar dener misin?",
            error_stage="openrouter",
            error_type="Timeout",
        )
        self.assertEqual(result["outcome"], OUTCOME_TECHNICAL_ERROR)

    def test_tool_error_is_partial_success(self):
        result = categorize_interaction(
            "Inception gişesi ne?",
            "Sayısal veriye ulaşamadım ama filmi kısaca yorumlayabilirim.",
            tool_calls=[{"status": "error"}],
        )
        self.assertEqual(result["outcome"], OUTCOME_PARTIAL_SUCCESS)

    def test_out_of_scope_is_a_separate_outcome(self):
        result = categorize_interaction(
            "Kek tarifi ver",
            "Üzgünüm, ben bir sinema asistanıyım ve sadece filmler/diziler "
            "ile CineMatch uygulaması hakkında konuşabilirim.",
        )
        self.assertEqual(result["outcome"], OUTCOME_OUT_OF_SCOPE)


if __name__ == "__main__":
    unittest.main()
