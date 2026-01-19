# /mnt/d/my_agent_win/tests/conftest.py
import pytest

@pytest.fixture
def mock_llm_ask(mocker):
    """
    'agent.models.llm.ask' fonksiyonunu mock'lar (taklit eder).
    Bu fixture'ı kullanan testler, gerçek LLM'e istek atmak yerine
    önceden tanımlanmış bir yanıt alacaklar.
    """
    # `mocker` pytest-mock kütüphanesinden gelir.
    # Hangi modülün içindeki hangi fonksiyonu mock'layacağımızı belirtiyoruz.
    # Bu, tüm araçların import ettiği yolu hedef almalıdır.
    # Örneğin, `code_auditor.py` `from agent.models.llm import ask` yaptığı için bu yol doğrudur.
    mocked_ask = mocker.patch("agent.models.llm.ask")

    # Varsayılan olarak, mock'un ne döndüreceğini belirleyelim.
    # Her test bunu kendi ihtiyacına göre üzerine yazabilir.
    mocked_ask.return_value = '{"status": "success", "result": "Mocked LLM Response"}'

    return mocked_ask