# /mnt/d/my_agent_win/tests/test_tools/test_internet_search.py
import pytest
from unittest.mock import MagicMock, patch
from agent.tools import internet_search

@pytest.fixture
def mock_ddgs(mocker):
    """DDGS kütüphanesini mock'lar."""
    # `ddgs.ddgs.DDGS` sınıfını patch'liyoruz.
    # `__enter__` ve `__exit__` metodlarını da mock'lamamız gerekiyor çünkü `with` bloğu içinde kullanılıyor.
    mock_instance = MagicMock()
    mock_ddgs_class = mocker.patch("ddgs.ddgs.DDGS")
    mock_ddgs_class.return_value.__enter__.return_value = mock_instance
    return mock_instance

def test_internet_search_success(mock_ddgs, mock_llm_ask):
    """
    internet_search aracının başarılı bir arama ve özetleme senaryosunu test eder.
    """
    # 1. Mock DDGS'in ne döndüreceğini ayarla
    mock_search_results = [
        {"title": "Sonuç 1", "href": "http://example.com/1", "body": "Bu ilk sonuçtur."},
        {"title": "Sonuç 2", "href": "http://example.com/2", "body": "Bu da ikinci sonuçtur."},
    ]
    mock_ddgs.text.return_value = mock_search_results

    # 2. Mock LLM'in özetleme için ne döndüreceğini ayarla
    mock_llm_ask.return_value = "Bu, iki sonucun birleştirilmiş özetidir."

    # 3. Aracı çalıştır
    result = internet_search.run("test sorgusu")

    # 4. Sonuçları doğrula
    assert result["status"] == "success"
    assert result["result"] == "Bu, iki sonucun birleştirilmiş özetidir."
    assert len(result["sources"]) == 2
    mock_ddgs.text.assert_called_once_with("test sorgusu", max_results=8)

def test_internet_search_no_results(mock_ddgs):
    """Arama motoru hiç sonuç döndürmediğinde ne olduğunu test eder."""
    mock_ddgs.text.return_value = [] # Boş liste döndür
    result = internet_search.run("kimsenin aramadığı bir şey")
    assert result["status"] == "empty"
    assert "Arama sonucu bulunamadı" in result["message"]