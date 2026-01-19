import pytest
from unittest.mock import patch, MagicMock
from agent.tools.community_tools import critical_web_researcher

@pytest.fixture
def mock_llm():
    with patch('agent.tools.community_tools.critical_web_researcher.ask') as mock_ask:
        yield mock_ask

@pytest.fixture
def mock_search():
    with patch('agent.tools.community_tools.critical_web_researcher.search_urls') as mock_search_urls:
        yield mock_search_urls

@pytest.fixture
def mock_read():
    with patch('agent.tools.community_tools.critical_web_researcher.read_url') as mock_read_url:
        yield mock_read_url

def test_run_successful(mock_llm, mock_search, mock_read):
    # Arrange
    mock_llm.side_effect = [
        '{"sub_topics": ["sub_topic_1", "sub_topic_2"]}',  # Sub-topics
        "Summary for URL 1",  # Page summary 1
        "Summary for URL 2",  # Page summary 2
        "Synthesized summary for sub_topic_1",  # Sub-topic summary 1
        "Summary for URL 3",
        "Summary for URL 4",
        "Synthesized summary for sub_topic_2",
        "Final report"  # Final report
    ]
    mock_search.side_effect = [
        {"status": "success", "result": [{"url": "http://example.com/1"}, {"url": "http://example.com/2"}]},
        {"status": "success", "result": [{"url": "http://example.com/3"}, {"url": "http://example.com/4"}]}
    ]
    mock_read.side_effect = [
        "Content for URL 1",
        "Content for URL 2",
        "Content for URL 3",
        "Content for URL 4"
    ]

    # Act
    result = critical_web_researcher.run("test topic")

    # Assert
    assert result["status"] == "success"
    assert result["result"] == "Final report"
    assert len(result["sources"]) == 4

def test_run_sub_topic_generation_fails(mock_llm):
    # Arrange
    mock_llm.side_effect = ["invalid json"] * 3

    # Act
    result = critical_web_researcher.run("test topic")

    # Assert
    assert result["status"] == "error"
    assert "Could not generate sub-topics" in result["message"]

def test_run_no_urls_found(mock_llm, mock_search):
    # Arrange
    mock_llm.side_effect = [
        '{"sub_topics": ["sub_topic_1"]}',
    ]
    mock_search.return_value = {"status": "error", "message": "No results"}

    # Act
    result = critical_web_researcher.run("test topic")

    # Assert
    assert result["status"] == "error"
    assert "No information could be gathered" in result["message"]

def test_run_content_read_fails(mock_llm, mock_search, mock_read):
    # Arrange
    mock_llm.side_effect = [
        '{"sub_topics": ["sub_topic_1"]}',
    ]
    mock_search.return_value = {"status": "success", "result": [{"url": "http://example.com/1"}]}
    mock_read.return_value = "URL read error"

    # Act
    result = critical_web_researcher.run("test topic")

    # Assert
    assert result["status"] == "error"
    assert "No information could be gathered" in result["message"]

def test_run_final_report_fails(mock_llm, mock_search, mock_read):
    # Arrange
    mock_llm.side_effect = [
        '{"sub_topics": ["sub_topic_1"]}',
        "Summary for URL 1",
        "Synthesized summary for sub_topic_1",
        Exception("LLM failed")
    ]
    mock_search.return_value = {"status": "success", "result": [{"url": "http://example.com/1"}]}
    mock_read.return_value = "Content for URL 1"

    # Act
    result = critical_web_researcher.run("test topic")

    # Assert
    assert result["status"] == "error"
    assert "An error occurred while generating the final report" in result["message"]


def test_run_long_context_summarization(mock_llm, mock_search, mock_read):


    # Arrange


    long_summary = "a" * 13000


    mock_llm.side_effect = [


        '{"sub_topics": ["sub_topic_1"]}',


        "Summary for URL 1",


        long_summary,  # Long synthesized summary


        "Intermediate summary", # Intermediate summary


        "Final report"


    ]


    mock_search.return_value = {"status": "success", "result": [{"url": "http://example.com/1"}]}


    mock_read.return_value = "Content for URL 1"





    # Act


    result = critical_web_researcher.run("test topic")





    # Assert


    assert result["status"] == "success"


    assert result["result"] == "Final report"


    # Check that the intermediate summarization was called


    assert "Intermediate summary" in [call[0][0] for call in mock_llm.call_args_list if "Summarize the following research findings" in call[0][0]]