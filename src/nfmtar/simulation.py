"""Kapalı döngü ayrık olay simülasyonu: ölç -> karar -> uygula -> ilet -> geri besle (CLAUDE.md §5.10).

İskelet dosyası: henüz uygulanmadı. Her modül kodlanırken bu docstring,
hangi denklemi/kararı uyguladığını belirtecek şekilde güncellenir (CLAUDE.md §1).

Yazılırken uyulacak kararlar (şimdiden bağlayıcı):
- §5.5 zamanlama kuralı: düğüm servis döngüsü, bir paketin atlaması bittikten sonra sıradaki paketi
  `RadioModel.next_packet_ready_s(hop_result)` anında kuyruktan alır (son MAC etkinliği + config
  `radio.processing_delay_s`; başarı/kayıp ayrımı yok). İşlem süresi için ayrı bir sabit TANIMLANMAZ; AIFS
  kalibrasyonu (`experiments/calibration/mac_aifs.py`, `radio.saturated_sender`) aynı fonksiyonla ölçüldüğü için
  kural burada farklı olursa kalibre edilmiş `aifs_slots` simülasyona taşınmaz.
- §3: `send_packet` üreteçlerinin `yield t` protokolü, `radio.run_mac_sessions` ile aynı biçimde kendi heapq
  olay kuyruğunda sürülür (farklı düğümlerin geri çekilme/iletim adımları birbirine karışık işlenir).
"""
