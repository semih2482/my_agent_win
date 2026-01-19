
import sys
import os
import pprint

# Add the project root to the Python path
# This is a bit of a hack, a better solution would be to have the project installed as a package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from agent.tools.community_tools.critical_web_researcher import run

topic = "[siber_guvenlik_uzmanligi] Web sitesi zafiyetlerini (vulnerability) bulmak ve raporlamak, etik hacker'lık olarak bilinen önemli ve para kazandıran bir yetenektir. Bu konuda uzmanlaşmak için özellikle OWASP Top 10 gibi standartları, siber güvenlik akademik makaleleri ve güvenilir siber güvenlik bloglarını araştırarak öğrenmek istiyorum."

if __name__ == "__main__":
    result = run(topic)
    pprint.pprint(result)
