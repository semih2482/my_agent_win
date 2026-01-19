# /mnt/d/my_agent_win/test_tool.py
import os
import sys
import json
import importlib
import argparse
import pprint

# Proje kök dizinini Python yoluna ekle
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agent.config import Colors

def run_tool_test(tool_name: str, input_args_str: str):
    """
    Belirtilen bir aracı, verilen argümanlarla izole bir şekilde çalıştırır.
    Ajanın ana döngüsünü veya büyük modelleri yüklemeden hızlı test imkanı sağlar.
    """
    print(f"{Colors.HEADER}--- Araç Test Modu: '{tool_name}' ---{Colors.ENDC}")

    try:
        # Aracı dinamik olarak import et
        # Örn: 'code_auditor' -> 'agent.tools.code_auditor'
        module_path = f"agent.tools.{tool_name}"
        tool_module = importlib.import_module(module_path)
        importlib.reload(tool_module) # Her zaman en güncel kodu kullan

        if not hasattr(tool_module, 'run'):
            print(f"{Colors.FAIL}Hata: '{module_path}' modülünde 'run' fonksiyonu bulunamadı.{Colors.ENDC}")
            return

        tool_function = tool_module.run

        # Girdi argümanlarını JSON olarak ayrıştır
        try:
            if input_args_str.startswith('{'):
                input_args = json.loads(input_args_str)
            else:
                # Eğer JSON değilse, düz bir string olarak kabul et
                input_args = input_args_str
        except json.JSONDecodeError:
            print(f"{Colors.FAIL}Hata: Girdi argümanları geçerli bir JSON değil. Düz metin olarak deneniyor.{Colors.ENDC}")
            input_args = input_args_str

        print(f"{Colors.OKBLUE}🔧 Araç çalıştırılıyor...{Colors.ENDC}")
        print(f"{Colors.OKCYAN}   Girdi: {input_args}{Colors.ENDC}")

        # Aracı çalıştır
        result = tool_function(input_args)

        print(f"\n{Colors.OKGREEN}✅ Araç başarıyla çalıştı. Sonuç:{Colors.ENDC}")
        pprint.pprint(result)

    except ImportError:
        print(f"{Colors.FAIL}Hata: '{tool_name}' aracı bulunamadı. Dosya adının doğru olduğundan emin olun: agent/tools/{tool_name}.py{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.FAIL}Araç çalıştırılırken beklenmedik bir hata oluştu: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bir ajanın aracını izole bir şekilde test et.")
    parser.add_argument("tool_name", help="Test edilecek aracın dosya adı (örn: code_auditor).")
    parser.add_argument("input_args", help="Araca JSON formatında veya düz metin olarak gönderilecek argümanlar.")

    args = parser.parse_args()
    run_tool_test(args.tool_name, args.input_args)