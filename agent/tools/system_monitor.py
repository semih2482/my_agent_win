import psutil
import platform

# GPU ve sensörler için opsiyonel importlar
try:
    import GPUtil
    _GPUtil_AVAILABLE = True
except ImportError:
    _GPUtil_AVAILABLE = False

def _get_gpu_status():
    if not _GPUtil_AVAILABLE:
        return "  GPU Bilgisi: GPUtil kütüphanesi kurulu değil.\n"
    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            return "  GPU Bilgisi: Uyumlu bir GPU bulunamadı.\n"

        report = "GPU Durumu:\n"
        for gpu in gpus:
            report += (
                f"  - GPU {gpu.id}: {gpu.name}\n"
                f"    Kullanım: %{gpu.load*100:.1f}, Sıcaklık: {gpu.temperature}°C\n"
            )
        return report
    except Exception as e:
        return f"  GPU bilgisi alınamadı: {e}\n"

def _get_storage_status(path='/'):
    try:
        usage = psutil.disk_usage(path)
        return (
            f"Depolama ({path}):\n"
            f"  - Toplam: {usage.total / (1024**3):.2f} GB\n"
            f"  - Kullanılan: {usage.used / (1024**3):.2f} GB (%{usage.percent})\n"
        )
    except Exception as e:
        return f"  Depolama bilgisi alınamadı: {e}\n"

def _get_temperature_status():
    report = "Sıcaklık Durumu:\n"
    has_temps = False
    if hasattr(psutil, "sensors_temperatures"):
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    for entry in entries:
                        report += f"  - {name} ({entry.label or 'core'}): {entry.current}°C\n"
                has_temps = True
        except Exception:
            pass
    if not has_temps:
        return "  Sıcaklık sensörleri bulunamadı veya okunamadı.\n"
    return report

def _get_network_status():
    try:
        net_io = psutil.net_io_counters()
        return (
            "Ağ Bağlantısı:\n"
            f"  - Alınan Veri: {net_io.bytes_recv / (1024**2):.2f} MB\n"
            f"  - Gönderilen Veri: {net_io.bytes_sent / (1024**2):.2f} MB\n"
        )
    except Exception as e:
        return f"  Ağ bilgisi alınamadı: {e}\n"

def get_system_status():
    """
    Sistemin anlık durumunu (CPU, RAM, GPU, Depolama, Sıcaklık, Ağ) döndürür.
    """
    cpu_usage = psutil.cpu_percent(interval=1)
    ram_info = psutil.virtual_memory()
    ram_usage = ram_info.percent

    status_report = (
        "Sistem Durumu:\n"
        f"  İşlemci (CPU) Kullanımı: %{cpu_usage}\n"
        f"  RAM (Bellek) Kullanımı: %{ram_usage}\n"
        f"{_get_gpu_status()}"
        f"{_get_temperature_status()}"
        f"{_get_storage_status('C:/' if platform.system() == 'Windows' else '/')}"
        f"{_get_network_status()}"
    )

    return status_report