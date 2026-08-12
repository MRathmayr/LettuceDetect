"""Unit tests for TaskTypeClassifier."""

import pytest

from lettucedetect.detectors.task_classifier import TaskTypeClassifier, _looks_structured


@pytest.fixture
def classifier():
    return TaskTypeClassifier()


class TestQAClassification:
    """QA is the default when a question is present and no other pattern matches."""

    def test_plain_question(self, classifier):
        assert classifier.classify("What is the capital of France?", None) == "qa"

    def test_factual_query(self, classifier):
        assert classifier.classify("how to cook a frozen pork loin", None) == "qa"

    def test_question_with_context(self, classifier):
        assert classifier.classify("Who wrote Hamlet?", ["Shakespeare was..."]) == "qa"

    def test_ragtruth_qa_style(self, classifier):
        """RAGTruth QA queries are plain search queries with no keywords."""
        assert classifier.classify("best restaurants in new york city", None) == "qa"
        assert classifier.classify("python list comprehension examples", None) == "qa"


class TestSummarizationClassification:
    """Summarization detected via imperative instructions."""

    def test_summarize_keyword(self, classifier):
        assert classifier.classify("Summarize the following document.", None) == "summarization"

    def test_ragtruth_summarization(self, classifier):
        """RAGTruth: 'Summarize the following news within N words:'"""
        assert classifier.classify("Summarize the following news within 86 words:", None) == "summarization"

    def test_summarise_british(self, classifier):
        assert classifier.classify("Summarise this article", None) == "summarization"

    def test_tldr(self, classifier):
        assert classifier.classify("TL;DR this report", None) == "summarization"

    def test_tldr_no_semicolon(self, classifier):
        assert classifier.classify("TLDR of the above", None) == "summarization"

    def test_key_takeaways(self, classifier):
        assert classifier.classify("Give me the key takeaways", None) == "summarization"

    def test_main_points(self, classifier):
        assert classifier.classify("What are the main points?", None) == "summarization"

    def test_condense(self, classifier):
        assert classifier.classify("Condense this into a paragraph", None) == "summarization"

    def test_write_a_summary(self, classifier):
        assert classifier.classify("Write a summary of the document", None) == "summarization"

    def test_briefly_describe(self, classifier):
        assert classifier.classify("Briefly describe the findings", None) == "summarization"

    def test_make_shorter(self, classifier):
        assert classifier.classify("Make this shorter", None) == "summarization"

    def test_sum_up(self, classifier):
        assert classifier.classify("Sum up the key findings", None) == "summarization"


class TestData2txtClassification:
    """Data2txt detected via question keywords or structured context."""

    def test_ragtruth_data2txt_question(self, classifier):
        """RAGTruth: 'Write an objective overview...structured data in JSON'"""
        q = "Write an objective overview of the following structured data in JSON format."
        assert classifier.classify(q, None) == "data2txt"

    def test_convert_to_text(self, classifier):
        assert classifier.classify("Convert this data to text", None) == "data2txt"

    def test_transform_into_prose(self, classifier):
        assert classifier.classify("Transform this table into prose", None) == "data2txt"

    def test_generate_description_from(self, classifier):
        assert classifier.classify("Generate a description from the data", None) == "data2txt"

    def test_data_to_text(self, classifier):
        assert classifier.classify("Perform data-to-text conversion", None) == "data2txt"

    def test_json_context_detection(self, classifier):
        """Structured JSON context should trigger data2txt."""
        context = ['{"name": "Alice", "age": 30, "city": "NYC"}']
        assert classifier.classify("Tell me about this person", [context[0]]) == "data2txt"

    def test_table_context_detection(self, classifier):
        """Pipe-delimited table context should trigger data2txt."""
        context = ["| Name | Age | City |\n| Alice | 30 | NYC |\n| Bob | 25 | LA |\n| Carol | 35 | SF |"]
        assert classifier.classify("Describe these people", context) == "data2txt"

    def test_based_on_following_data(self, classifier):
        assert classifier.classify("Based on the following data, write a report", None) == "data2txt"


class TestDialogueClassification:
    """Dialogue maps to 'unknown' (probe hurts at 0.528 AUROC)."""

    def test_halueval_dialogue_format(self, classifier):
        """HaluEval dialogue: [Human]:...[Assistant]:... format."""
        q = "[Human]: Tell me about dogs [Assistant]: Dogs are domesticated animals."
        assert classifier.classify(q, None) == "unknown"

    def test_user_bot_format(self, classifier):
        q = "[User]: What's the weather? [Bot]: It's sunny today."
        assert classifier.classify(q, None) == "unknown"


class TestUnknownClassification:
    """Unknown when no question is provided."""

    def test_no_question(self, classifier):
        assert classifier.classify(None, ["some context"]) == "unknown"

    def test_no_question_no_context(self, classifier):
        assert classifier.classify(None, None) == "unknown"

    def test_empty_question(self, classifier):
        assert classifier.classify("", None) == "unknown"


class TestLooksStructured:
    """Test the _looks_structured helper."""

    def test_json_object(self):
        assert _looks_structured('{"name": "Alice", "age": 30, "city": "NYC"}') is True

    def test_short_json_not_structured(self):
        """Very short JSON-like strings are below the length threshold."""
        assert _looks_structured('{"k": "v"}') is False

    def test_json_array(self):
        assert _looks_structured('[{"a": 1}, {"b": 2}]') is True

    def test_pipe_table(self):
        text = "| A | B | C |\n| 1 | 2 | 3 |\n| 4 | 5 | 6 |"
        assert _looks_structured(text) is True

    def test_tab_separated(self):
        text = "name\tage\tcity\nAlice\t30\tNYC\nBob\t25\tLA"
        assert _looks_structured(text) is True

    def test_csv_like(self):
        text = "name,age,city\nAlice,30,NYC\nBob,25,LA"
        assert _looks_structured(text) is True

    def test_key_value_pairs(self):
        text = "Name: Alice\nAge: 30\nCity: NYC\nJob: Engineer"
        assert _looks_structured(text) is True

    def test_plain_text_not_structured(self):
        assert _looks_structured("This is a regular paragraph of text.") is False

    def test_short_text_not_structured(self):
        assert _looks_structured("short") is False

    def test_empty_string(self):
        assert _looks_structured("") is False
