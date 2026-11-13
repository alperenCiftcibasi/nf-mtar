# NF-MTAR Dönem Projesi — Durum ve Yapılacaklar

**Güncelleme:** 12 Kasım 2026
**Ders:** Bulanık Mantık — Dönem Projesi
**Referans makale:** M. S. Parveen ve P. T. V. Bhuvaneswari, "A neuro-fuzzy multi-topology adaptive routing framework for QoS-aware healthcare IoT communications," *Scientific Reports*, vol. 16, art. 3330, 2026. doi: 10.1038/s41598-025-33235-9
**Yeni teslim tarihi:** ______ (ertelendi; buraya yaz)

---

## 0. Özet durum

| Alan | Durum |
|---|---|
| Makale okuması | ✅ Tamamlandı (10/10) |
| Yöntem tasarımı | ✅ Bütün bileşenler tanımlandı (bölümleme, bulanık motor, öğrenme, rank, yol maliyeti, histerezis) |
| CLAUDE.md | ✅ Repoda; her kilometre taşından sonra güncelleniyor (son: M4 yol maliyeti düzeltmesi) |
| Kod | M0–M3 tamam; M4 (`router.py`) yazıldı, yol maliyeti düzeltmesi uygulanıyor; 67 test geçiyor |
| Tekrarlanacak deney | ✅ Seçildi: E1 ana tekrar deneyi; E2, E3, E4, E5, E8 destekleyici (§6) |
| Özgün katkı | Seçilmedi (adaylar ve öneri §5'te) |
| Literatür taraması | ✅ 26 doğrulanmış kaynak, dört alan (Claude Docs belgesi) |
| Rapor | Başlanmadı (plan §7'de) |
| Sunum | Başlanmadı (plan §8'de) |

---

## 1. Puanlama ve yönergeden kritik kurallar

| Bileşen | Puan | En ağır alt kriterler |
|---|---|---|
| Sunum ve sözlü savunma | 20 | Yöntem ve mimari (4), teorik altyapı (3), literatür (3) |
| Uygulama / yeniden gerçekleştirme | 30 | Doğru modelleme (6), fuzzy sistem (6), çalışabilirlik (6), deney tekrarı (5) |
| Özgün katkı | 30 | Teknik gerçekleştirme (8), deney tasarımı (5), adil karşılaştırma (5), analiz (5) |
| Rapor ve GitHub | 20 | Akademik yapı (4), literatür (3), yöntem (3), deneyler (3), tartışma (3) |

**Unutulmaması gerekenler:**
- Git commit geçmişi geliştirme sürecini göstermeli (küçük, düzenli commit).
- Savunmada kodun her bölümü açıklanabilmeli. Claude Code'un yazdığı her modülü commit'lemeden önce oku ve anla.
- Raporda **Generative AI Usage Disclosure** bölümü zorunlu (Claude ile makale analizi ve tasarım, Claude Code ile kod geliştirme).
- Doğrulanmamış veya var olmayan kaynak kullanılmayacak. Makalenin kaynakçasındaki çalışmalar dahil her kaynağı en az özetinden doğrula.
- Dış kaynaklı kod belirtilecek; gizli bilgi repoya girmeyecek.
- Rapor 8–12 sayfa (kaynakça hariç), IEEE kaynak formatı.
- Sunum 15–20 dk + ~5 dk soru-cevap ve canlı demo.
- Makale CC BY-NC-ND 4.0 lisanslı: şekillerini yalnızca değiştirmeden ve atıfla kullan; kendi mimari şeklini sıfırdan, kendi tasarımın olarak çiz.

---

## 2. Makale okuması

| # | Parça | Durum | Ana çıktı |
|---|---|---|---|
| 1 | Giriş, problem, araştırma boşlukları | ✅ | Argüman zinciri; boşluk–çözüm eşlemesi |
| 2 | Motivasyon, Şekil 1, katkılar | ✅ | Gerçekte dinamik olan root; edge router sabit |
| 3 | İlgili çalışmalar | ✅ | Atıf numarası kayması; kendi literatür taraman gerekli |
| 4 | Dört katmanlı mimari (Şekil 2) | ✅ | Root ≈ küme başı; kapalı döngü; modül yapısı |
| 5 | V-SDN denklemleri (1–7), Tablo 1 | ✅ | Tablo 1 parametre kararları; QoS denklemi kodlanmayacak |
| 6 | Bulanık çıkarım motoru (9–13) | ✅ | Zero-order Sugeno; girdi tanımları; 750 kural; MF parametreleri |
| 7 | Nöral öğrenme (17–19) | ✅ | Kâhin etiketleriyle çevrimdışı eğitim; uzmana bağlı düzenlileştirme |
| 8 | Rank, yol, histerezis (18–24) | ✅ | (1−f) düzeltmesi; −log PRR; üç tetikleyici; baseline'lar |
| 9 | Simülasyon ortamı ve sonuçlar (Şekil 4–6) | ✅ | Deney seçimi (E1 ana); politika/yönlendirme etkisinin ayrılması; aritmetik ve fiziksel tutarsızlıklar |
| 10 | İstatistik, Tablo 4, sınırlılıklar, sonuç | ✅ | Eşleştirilmiş istatistik; E5 root arıza deneyi; istatistik tutarsızlıkları |

---

## 3. Alınan tasarım kararları (özet)

Ayrıntıların tamamı CLAUDE.md'de. Bu tablo savunma ve rapor için hızlı başvuru.

| Konu | Karar | Gerekçe | CLAUDE.md |
|---|---|---|---|
| Simülatör | Kendi Python ayrık olay simülatörü | COOJA/MATLAB/ODL entegrasyonu belirsiz, kod yok | §3 |
| Alan / düğüm / menzil | 300×300 m, 100–200 düğüm, 50 m | Menzil makaleden; alan makalede yok | §5.1 |
| Topoloji isimleri | E = yoğun bakım elipsi, L = koridor şeridi, R = rastgele | Makale biçimsel tanım vermiyor | §5.2 |
| Bölümleme | Deterministik, trafik sınıfına göre; ayrık; bağlantısallık kontrolü | Denk. 2'nin olasılıkları tanımsız | §5.3 |
| Root | Edge router'dan ayrı koordinatör (küme başı benzeri); iki aşamalı yol | Şekil 2 ve Denk. 18'deki H terimi | §5.4 |
| Edge router | Topoloji başına sabit | Şekil 2 ve 3 | §5.4 |
| Radyo | 802.15.4, 250 kbit/s, 127 B çerçeve, parçalama, paylaşılan kanal | 6LoWPAN gerçekçiliği | §5.5 |
| Enerji | Birinci dereceden radyo modeli | Standart, basit | §5.5 |
| Tablo 1 | 6 parametre uygulanır, 4'ü metrik eşiği, güvenlik ve yedeklilik kapsam dışı | Simüle edilebilirlik | §5.7 |
| QoS denklemi (Denk. 4) | Kodlanmaz; yerine ölçülen QoS memnuniyet oranı | Normalizasyonsuz toplam ters sonuç veriyor | §5.7 |
| Controller | Merkezi, kontrol maliyeti varsayılan sıfır (opsiyonel) | Makaleyle tutarlı; sınırlılık olarak raporlanacak | §5.8 |
| Bulanık sistem tipi | Zero-order Takagi-Sugeno | Denk. 12 merkez-ortalamaya indirgeniyor | §5.14 |
| Girdiler | x1 komşuluk talebi, x2 kuyruk doluluğu, x3 enerji tükenmesi, x4 PRR, x5 komşu sayısı | Örnek kurallarla yön uyumu; x1–x2 tekrarının önlenmesi | §5.14 |
| MF | Makale formu Gauss, eşit aralıklı, 0,5'te kesişen | Türevlenebilirlik; standart başlangıç | §5.14 |
| Kural tabanı | 750 kuralın tamamı, uzman fonksiyonuyla üretilmiş | Makale kuralları yayımlamamış; kapsama boşluğu riski | §5.14 |
| Öğrenme parametreleri | θ = {c, σ, y_k} (ANFIS tarzı) | "Ağırlık ve bias" bulanık sistemle uyumsuz | §5.16 |
| Öğrenme hedefi | Kâhin etiketleri, çevrimdışı eğitim | y_desired tanımsız; çevrimiçi geri bildirim keşif sorunu yaratır | §5.16 |
| Düzenlileştirme | Uzman değerlerine bağlı L2 | Sıfıra çekmek bulanık parametrelerin anlamını bozar | §5.16 |
| Rank | w1·(1−f) + w2·H + w3·C, normalize, topolojiye özgü ağırlık | Makaledeki argmin işaret hatası | §5.17 |
| Yol maliyeti | λ1·E_hop(d)/E_hop(R) + λ2·(−log PRR)/L_max + λ3·tükenme; λ3 = π6, λ1 = λ2 = (1−λ3)/2; ters Dijkstra | E_hop atlama başına sabit enerji bedelini içerir (d² tek başına kısa atlamalı uzun yolları ödüllendiriyordu); −log çarpımsal güvenilirliği toplamsala çevirir | §5.9 |
| Tetikleyici | Periyodik (π11) + göreli trafik değişimi + root sağlığı | Makaledeki tetikleyici root arızasını görmüyor | §5.18 |
| Histerezis | ε marjı + minimum bekleme; root ölünce marj atlanır | Titremeyi önleme | §5.18 |
| Baseline'lar | B1 hop, B2 ham ETX (MRHOF benzeri) + π6·tükenme, B3 eğitimsiz bulanık | Bileşen katkılarını kademeli izole eder | §5.12 |
| Baseline adilliği | Sabit root ölürse yenisi seçilir | Yapay çöküşü önleme | §5.12 |

---

## 4. Kod yol haritası

### 4.1 CLAUDE.md

- [ ] Birleşik CLAUDE.md'yi (Parça 1–10) repodakinin yerine koy ve commit'le

### 4.2 Kilometre taşları

Her madde en az bir ayrı commit. Durumunu işaretle.

| # | İş | Bağımlılık | Checkpoint | Durum |
|---|---|---|---|---|
| M0 | Repo, CLAUDE.md, README taslağı, LICENSE, .gitignore, requirements.txt | — | — | [x] |
| M1 | `config.py` + `configs/default.yaml`; fiziksel ağ, yerleşim (E/L/R), bölümleme, bağlantısallık kontrolü | M0 | 2 | [x] |
| M2 | Radyo: iletim süresi, parçalama, CSMA, PRR, paylaşılan kanal; enerji modeli | M1 | 2 | [x] (AIFS kalibrasyonu dahil) |
| M3 | Trafik üretimi (üç sınıf, yük çarpanı) | M1 | 2 | [x] |
| M4 | `router.py`: kenar maliyeti (§5.9), ters yönlü Dijkstra, iki aşamalı yol | M2 | 3 | [~] ilk sürüm commit'lendi; maliyet düzeltmesi |
| M5 | `metrics.py` + `simulation.py` (kapalı döngü) | M2–M4 | 3 | [ ] |
| M6 | B1 ve B2 baseline'ları, ilk sonuçlar | M5 | **3** | [ ] |
| M7 | `engine/fuzzy.py` + B3 + testler (monotonluk, kalibrasyon raporu) | M5 | 3 | [ ] |
| M8 | `engine/oracle.py`, eğitim verisi üretimi, `engine/trainer.py`, öğrenme eğrileri | M7 | 4 | [ ] |
| M9 | `engine/rank.py`, controller tetikleyicileri ve histerezis → NF-MTAR tamam | M7–M8 | 4 | [ ] |
| M10 | Deneyler E1, E3, E5, E2, E8, E4 (CLAUDE.md §5.19) | M9 | **5** | [ ] |
| M11 | Özgün katkı deneyleri | M9, §5 kararı | **5** | [ ] |
| M12 | İstatistik testleri, grafikler, tablolar | M10–M11 | 5 | [ ] |
| M13 | README tamamlama, sürüm sabitleme, `references/README.md`, tekrar üretilebilirlik testi (temiz ortamda klonla → kur → çalıştır) | M12 | Final | [ ] |

### 4.3 Simülatör ilk koşularında özellikle bakılacaklar

- Ayrık bölümleme alt çizgeleri koparıyor mu? (bağlantısallık uyarıları — eleştiri ve katkı için veri)
- Kanal doyuyor mu? (PDR çok düşükse yük çarpanını kalibre et)
- Root düğümlerin enerjisi diğerlerinden hızlı mı düşüyor? (sıcak nokta analizi)
- 1024 B paketlerin atlama başına gecikmesi ≥ ~33 ms çıkıyor mu? (makaleyle karşılaştırma için kanıt)

---

## 5. Özgün katkı

### 5.1 Adaylar

| # | Aday | Gerekçe (makalenin hangi boşluğu) | Ölçütler | Efor | Bulanık mantık odağı |
|---|---|---|---|---|---|
| K1 | **Yorumlanabilirliği koruyan eğitim** (uzmana bağlı vs sıfıra çeken L2 vs düzenlileştirmesiz; λ taraması) | Makale yorumlanabilirliği gerekçe yapıyor ama korumuyor; L2 yanlış yöne çekiyor | Pişmanlık, top-1, Spearman, terim sırası bozulması, ağ metrikleri | Orta | Yüksek |
| K2 | **Mamdani vs Sugeno** (aynı kural tabanı; centroid vs merkez-ortalama) | Makale Mamdani dilinde yazıp Sugeno gibi hesaplıyor; yönergede açıkça sayılan katkı türü | Ağ metrikleri, karar süresi, sıralama uyumu | Orta | Yüksek |
| K3 | Girdi ablasyonu (her girdiyi sırayla çıkar) | Makale girdileri üç farklı şekilde sayıyor; hangisinin önemli olduğu belirsiz | Pişmanlık, ağ metrikleri | Düşük | Orta |
| K4 | Tam (750) vs budanmış kural tabanı | Budama + Gauss → sessiz dış değerlendirme | Doğruluk, güven dağılımı, süre | Düşük–orta | Yüksek |
| K5 | Gradyan vs metasezgisel (PSO/GWO) eğitim | En yakın rakip TSFIS-GWO ile yöntemsel fark | Doğruluk, yakınsama, süre | Orta | Orta |
| K6 | Gürültü dayanıklılığı (Type-1 vs Interval Type-2) | Ölçüm belirsizliği; makalenin "belirsizliğe dayanıklılık" iddiası | Gürültü seviyesine göre bozulma | Yüksek | Yüksek |
| K7 | Ayrık vs örtüşen bölümleme | Bağlantısallık riski; klasik MTR örtüşmeye izin verir | PDR, gecikme, kopuk alt çizge oranı | Orta–yüksek | Düşük |
| K8 | Kontrol maliyetinin modellenmesi (π11 taraması) | Merkezi kontrolün bedeli değerlendirilmemiş | Ağ metrikleri vs güncelleme sıklığı | Orta | Düşük |
| K9 | ε / histerezis ablasyonu + güven kapısı | ε tanımsız; güven ölçüsü hiç kullanılmamış | Root değişim sayısı, ağ metrikleri | Düşük | Orta |

### 5.2 Öneri

- **Ana katkı:** K1 (yorumlanabilirliği koruyan eğitim). Makalenin merkezi iddiasını doğrudan sınıyor, dersin konusuyla en güçlü bağa sahip, ölçütleri net ve Parça 7 altyapısı (kâhin, eğitim) zaten gerekli.
  - Zaman daralırsa yedek ana katkı: K2 (Mamdani vs Sugeno) — yönergede birebir sayılan katkı türü, kâhin/eğitim gerektirmez.
- **Destekleyici ablasyonlar (ucuz, tartışmayı zenginleştirir):** K3 (girdi ablasyonu) ve K9 (ε taraması + güven kapısı).
- Karar senin; seçtiğinde CLAUDE.md §6'ya yaz.

---

## 6. Deney planı (kesin — ayrıntılar CLAUDE.md §5.19)

| Deney | Makaledeki karşılığı | Amaç | Seed |
|---|---|---|---|
| **E1** Trafik türüne göre throughput/PDR/gecikme, 3 yük seviyesi | Şekil 4a, 5a, Tablo 4 | **Ana tekrar deneyi** | 30 |
| E3 Atlama sayısına göre gecikme | Şekil 6d | E1 loglarından (ek koşu yok) | — |
| Dağılım, CDF, yüzdelikler | Şekil 4d–e, 5c–e | E1 loglarından | — |
| E2 Ölçeklenebilirlik (yoğunluk sabit) | Şekil 6b | Ölçeklenebilirlik iddiası | 20 |
| E8 Ağ ömrü (FND/HND) + root enerji eğrileri | Tablo 4 | Enerji iddiası, sıcak nokta | 15 |
| E4 Paket boyutu | Şekil 6c | Fiziksel eğim sınırı (0,032 ms/B) | 20 |
| E5 Root arıza enjeksiyonu | — (makalede test edilmemiş) | Makalenin motivasyon senaryosu; canlı demo | 30 |
| 24 saatlik analiz | Şekil 6a | Dışarıda (trafik profili tanımsız) | — |

**Tasarım ilkeleri:**
- B1, B2, B3, NF-MTAR aynı V-SDN politikasını paylaşır; B1-plain politikasız referans → politika etkisi ile yönlendirme etkisi ayrışır
- Bütün yöntemler aynı seed'lerle → eşleştirilmiş istatistik (eşleştirilmiş t, Wilcoxon, Friedman), Holm–Bonferroni, etki büyüklüğü + GA
- 60 s ısınma metriklerden hariç

**Makaleyle karşılaştırma (trend bazlı):** yöntem sıralaması, sınıf sıralaması, göreli iyileşme (ve bunun politika/yönlendirme payları), fiziksel tutarlılık (eğim, atlama başı gecikme). Eşleme: B1-plain ≈ makalenin "geleneksel RPL"i; B3 ≈ "bulanık baseline" ve TSFIS-GWO'ya en yakın.

---

## 7. Rapor planı (8–12 sayfa, IEEE)

| Bölüm | İçerik ve kaynak |
|---|---|
| 1. Introduction | Argüman zinciri (trafik çeşitliliği → sabit root → SDN → zekâ eksikliği); amaç ve kapsam (Parça 1–2) |
| 2. Related Work | **Kendi literatür taraman** (§10); NF-MTAR'ı bulanık küme başı seçimi ve ağ dilimleme çerçevesine konumlandır (Parça 3–4) |
| 3. Theoretical Background | Bulanık küme, MF, dilsel değişken, çıkarım (çarpım t-normu), durulaştırma, Mamdani vs Sugeno, ANFIS, SDN, 6LoWPAN, en büyük üyelik ilkesi (Parça 5–6) |
| 4. Original Method | Makalenin anlattığı haliyle V-SDN ve neuro-fuzzy motor (Denk. 1–24) |
| 5. Reproduction Methodology | §3'teki her karar ve gerekçesi; "makalede tanımsız olduğundan şöyle modellenmiştir" |
| 6. Proposed Extension | Seçilen katkı, gerekçesi, başarı ölçütü |
| 7. Experimental Setup | Simülatör, parametreler, seed'ler, yazılım sürümleri, donanım |
| 8. Results | Tablolar ve grafikler |
| 9. Discussion | Sonuçların nedenleri; makaleyle farkların fiziksel gerekçeleri |
| 10. Limitations | Kontrol maliyeti sıfır, tek trafik türü varsayımı, güvenlik/yedeklilik kapsam dışı, simülasyon-yalnız, makalenin sınırlılıkları |
| 11. Conclusion | Özet ve gelecek çalışma (örtüşen bölümleme, bulanık topoloji ataması, gerçek uyarlanabilir ağırlıklar vb.) |
| Generative AI Usage Disclosure | Claude (analiz, tasarım), Claude Code (kod) — kapsamı ve doğrulama yöntemi |
| References | IEEE formatı, yalnızca doğrulanmış kaynaklar |

---

## 8. Sunum planı (15–20 dk)

| Bölüm | Puan | Süre | İçerik |
|---|---|---|---|
| Problem ve motivasyon | 2 | 2 dk | Trafik çeşitliliği; Şekil 1a/1b (çapraz ok = ayarlanabilir eleman) |
| Literatür | 3 | 2 dk | Karşılaştırmalı tablo (kendi taraman) |
| Teorik altyapı | 3 | 3 dk | Makalenin denklemleri üzerinden bulanık kavramlar; Sugeno indirgemesi; en büyük üyelik ilkesi |
| Yöntem ve mimari | 4 | 3 dk | Kendi düzeltilmiş mimari şeklin (kapalı döngü, kontrol/veri akışı ayrımı) |
| Deney ve veri | 2 | 1,5 dk | Simülatör, parametreler, baseline kademesi |
| Sonuçlar | 2 | 2,5 dk | Tekrarlanan deney + makaleyle karşılaştırma |
| Sınırlılıklar ve geliştirmeler | 2 | 2 dk | En güçlü 4–5 eleştiri (§9'daki ★) + katkı sonuçları |
| Soru-cevap ve demo | 2 | ~5 dk | Aşağıda |

**Canlı demo planı:**
- Bir düğümün beş girdisini göster → bulanık skor, güven ve **en güçlü kural** (insan okunur metin)
- Hocanın verdiği farklı girdi değerleriyle anlık skor hesabı
- Root değişim logu: tetikleyici türü, eski/yeni root, R farkı
- Bir root'u elle "öldür" → root sağlığı tetikleyicisiyle anında yeniden seçim

**Savunmada sorulabilecekler (yönerge §14) ve hazır cevaplar:** kuralların işlevi (uzman fonksiyonu + en güçlü kural), MF seçimi (Gauss: türevlenebilirlik; 0,5 kesişim), parametre etkileri (ε, λ, rank ağırlıkları ablasyonları), algoritmalar (Dijkstra, geri yayılım, ters yönlü tek kaynak), değişikliklerin gerekçeleri (§3), sonuçların nedenleri, kod bölümleri, sınırlılıklar.

---

## 9. Eleştirel değerlendirme envanteri

★ = sunumda kullanılacak en güçlü ve doğrulanabilir maddeler

**Fiziksel gerçekçilik ve sonuçlar**
- ★ 1024 B paket için raporlanan 28–30 ms uçtan uca gecikme, 802.15.4'te tek atlamanın teorik alt sınırının (8192 bit / 250 kbit/s ≈ 32,8 ms) altında
- Şekil 6'daki değerler metinle uyuşmuyor (6a: ~3 ms vs metinde 18–20 ms; 6b: 0–1,5 ms vs 12–28 ms)
- Özetteki %99,8 "güvenilirlik" gövdede yok; en yakın metrik PDR %98,7
- 512 KB tampon, COOJA'da yaygın sensör platformlarının (~10 KB RAM) belleğiyle uyumsuz
- Donanım ve klinik doğrulama yapılmadığı yöntem bölümünde yan cümleyle kabul ediliyor

**Karşılaştırma ve literatür**
- ★ Karşılaştırılan yöntemlerin çoğu farklı ağ türleri için (TSN/kablolu Ethernet, veri merkezi, FANET); biri anket makalesi; 6LoWPAN'a nasıl uyarlandıkları açıklanmamış; RPL'in kaynakçada karşılığı yok
- Metin içi atıf numaraları kaynakçayla sistematik olarak eşleşmiyor (kaynakça iki alfabetik blok halinde numaralandırılmış)
- Vaat edilen literatür kapsamı karşılanmamış; RPL, bulanık yönlendirme, SDN-WSN ve sağlık literatürü taranmamış
- RPL olduğundan katı resmedilmiş (çoklu instance ve amaç fonksiyonları tartışılmamış)

**Kavramsal tutarlılık**
- ★ Merkezi ML tek hata noktası diye eleştirilirken yöntem merkezi bir controller'a dayanıyor
- ★ Yeniden değerlendirme tetikleyicisi yalnızca trafik değişimine duyarlı; makalenin motivasyonu olan root arızasını fark etmiyor
- "Tahmin edici" iddiasının yöntemde karşılığı yok; adaptasyon reaktif
- Şekil 1b'deki dinamik edge router ataması formüle edilmemiş; Şekil 2–3'te sabit
- Root ≈ küme başı, ama veri birleştirme gibi işlevi yok → yol uzaması ve sıcak nokta; bulanık küme başı literatürü anılmıyor
- Motivasyondaki "veri üzerine yazılması" yönlendirmeye nedensel olarak bağlanamaz
- "Ultra düşük gecikme" teriminin 6LoWPAN için kullanımı
- Yorumlanabilirlik iddiası eğitimde korunmuyor
- Katkı listesi şişirilmiş (simülasyon düzeneği katkı olarak sayılmış; "benzersiz model" standart yapılardan oluşuyor)

**Matematiksel model**
- ★ QoS denklemi (Denk. 4) birimleri normalizasyonsuz topluyor; Tablo 1 değerleriyle kritik olmayan topoloji en yüksek skoru alıyor (E ≈ 83, L ≈ 115, R ≈ 653)
- QoS denklemi düğüm durumundan bağımsız → minimum QoS kısıtı işlevsiz
- Amaç fonksiyonunda (Denk. 6) atama değişkenleri x_it yok; problemin nasıl çözüldüğü belirtilmemiş
- Rank formülünde bulanık skor ters işaretle → argmin en uygunsuz düğümü seçer
- Yol maliyetinde link kalitesi ters yönde ekleniyor; toplamsal maliyet çarpımsal güvenilirliği temsil etmiyor
- Yol maliyetindeki doğrusal mesafe terimi atlama başına sabit bedel içermiyor: aynı doğrultudaki bir atlamayı ikiye bölmek maliyeti değiştirmiyor, yani atlama sayısı cezalandırılmıyor (bizim d² sürümümüzde bu, ortalama 7,6–10 atlamalık yollar olarak gözlendi)
- Küme tanımında sembol tekrarı; ayrıklık koşulu üçlü kesişimle yazılmış
- "Maksimum olabilirlik" teriminin yanlış kullanımı; P_t tanımsız
- OpenFlow kuralında "topolojiye ilet" eylemi, öncelik ve kuyruk alanlarının yanlış kullanımı; düğüm tabanlı eşleme
- Ağırlıklar toplamı 1,8; α, β, γ, w, λ, ε, τ değerleri verilmemiş
- Denklem numaralandırması ve metin içi göndermeler tutarsız; "53×2×3" baskı hatası

**Bulanık motor ve öğrenme**
- Girdiler üç yerde üç farklı şekilde sayılıyor; x3 yön çelişkisi; "geçmiş" girdisi denklemlerde yok
- Kural tabanı yayımlanmamış (yalnızca 3 örnek)
- Budanmış kural tabanı + Gauss MF → kapsanmamış bölgelerde sessiz dış değerlendirme
- Mamdani tarzı kural formu, Sugeno tarzı hesaplama; μ_Bk(y_k) terimi gereksiz
- Güven ölçüsü tanımlanmış, hiç kullanılmamış; w sembolü iki anlamda
- Gauss formu MATLAB gaussmf'ten farklı
- ★ Öğrenme hedefi y_desired tanımsız (yeniden üretimin önündeki en büyük engel)
- Sönümlenen öğrenme oranı ile "sürekli öğrenme" iddiası çelişiyor (kararlılık–esneklik)
- L2 parametreleri sıfıra çekiyor → bulanık parametrelerin anlamıyla çelişki
- "Ağırlık ve bias" dili; nöral mimari tanımsız; "hızlı öğrenme" iddiası öğrenme eğrisiz

**İstatistik ve iç tutarlılık (Parça 9–10)**
- ★ Gecikme azalma yüzdelerinin beşten ikisi milisaniye değerleriyle tutmuyor (bulanık baseline: iddia %55,4, hesap %79,2; uyarlamalı: iddia %64,7, hesap %78,0); uyarlamalı yöntemlerin kritik gecikmesi üç farklı değerle veriliyor (38,9 / 52,1 / 83,5 ms)
- ★ Paket boyutu–gecikme eğimi (~0,021 ms/B), 802.15.4'te tek atlamanın fiziksel alt sınırının (0,032 ms/B) altında
- ★ Mann-Whitney U değerleri (2756–3201), grup başına 50 koşuda mümkün olan en büyük değeri (50 × 50 = 2500) aşıyor
- Gecikme ANOVA'sında F = 2312,45 ile η² = 0,853 uyumsuz (η² = 0,853 için F ≈ 313); throughput ANOVA'sında F ve η² tutarlı
- Cohen d değerleri t istatistiği ve n = 50 ile tutarsız (beklenen d = t·√(2/50)); hepsi sistematik olarak ~%22 küçük
- Tablo 2'deki RPL satırı, NF-MTAR'ın genel throughput'unu (94,3) RPL'in yalnızca kritik sınıf değeriyle (71,4) karşılaştırıyor görünüyor
- Tablo 3'teki OptRISQL satırı (39,8 ms) Tablo 4'teki OptRISQL değeriyle (41,2) değil, Graph RL değeriyle tutarlı
- Tablo 4'teki "ortalama gecikme" (18,4 ms) aslında yalnızca kritik sınıfın gecikmesi
- Özetteki %31,2 enerji verimliliği iyileşmesi gövdede hiçbir yerde yok
- ANOVA varsayımları (normallik, eşit varyans) makalenin kendi "iki tepeli dağılım" iddiasıyla çelişiyor; çoklu karşılaştırma düzeltmesi yok
- Ağ ömrü kablolu TSN yöntemine göre göreli yüzdeyle; enerji verimliliği ve uyum hızı ölçümsüz sözel değerler
- ★ Makalenin motivasyonu root arızası, ama sınırlılıklar bölümü aşırı düğüm arızası altındaki performansı gelecek çalışmaya bırakıyor → çözüldüğü iddia edilen senaryo test edilmemiş (tetikleyici de onu fark etmiyor)
- Sınırlılıklar bölümünde yüklemi olmayan yarım bir cümle; veri yalnızca talep üzerine, kod paylaşılmamış

> Not: Sunumda en fazla 4–5 madde. Önerilen beşli: (1) aritmetik ve istatistik tutarsızlıkları (yüzde tablosu + imkânsız U değerleri), (2) fiziksel imkânsızlık (paket boyutu eğimi / 1024 B gecikmesi), (3) QoS denkleminin ters sonucu, (4) karşılaştırılan yöntemlerin farklı ağ türlerine ait olması, (5) root arızası: motivasyon ama test edilmemiş ve tetikleyici fark etmiyor. Geri kalanı raporda ve savunmada.

---

## 10. Literatür taraması (yapılacak)

Dört alan, her birinden doğrulanmış 2–4 güncel çalışma:

1. WBAN / IoMT'de bulanık mantık tabanlı yönlendirme
2. Neuro-fuzzy / ANFIS tabanlı yönlendirme ve bulanık küme başı seçimi
3. RPL ve bulanık RPL amaç fonksiyonları
4. SDN tabanlı IoT ve kablosuz sensör ağları

**Doğrulanarak kullanılabilecek bilinen kaynaklar (her birini kontrol et):**
- Memarian ve ark., TSFIS-GWO, *Applied Soft Computing*, 2024 (makalenin kaynakçası [9])
- Sharma ve ark., OptRISQL, *IEEE TNSM*, 2024 (makalenin kaynakçası [10])
- Demir ve Cevher, TSN için çoklu topoloji yönlendirme, *Real-Time Systems*, 2023 (makalenin kaynakçası [2])
- RFC 6550 (RPL), RFC 4944 ve RFC 6282 (6LoWPAN), RFC 4915 (OSPF çoklu topoloji)
- Bulanık küme başı seçimi üzerine klasik çalışmalar (ör. Gupta ve ark., 2005 — künyeyi doğrula)
- ETX metriği (De Couto ve ark., 2003 — künyeyi doğrula)
- SDN-WISE ve Contiki için μSDN (künyeleri doğrula)

İstersen bu taramayı web üzerinden birlikte yapıp her kaynak için özet ve NF-MTAR ile ilişkisini çıkarabiliriz.

---

## 11. Checkpoint durumu

| Checkpoint | İçerik | Durum |
|---|---|---|
| 1 — Makale ve problem analizi | Problem, yöntem, girdi/çıktı, deney yapısı | İçerik hazır (Parça 1–10); yazılı not haline getirilmeli |
| 2 — Uygulama planı | Ortam, veri, MF, kural tabanı, deney planı | Hazır (§3, §6); yazılı not haline getirilmeli |
| 3 — Baseline sistem | Baseline çalışır + ilk sonuçlar | M6–M7 durumuna bağlı |
| 4 — Katkı önerisi | Ne, neden, nasıl ölçülecek | §5 öneriye hazır; karar bekliyor |
| 5 — Deneysel sonuçlar | Karşılaştırmalı tablolar ve grafikler | M10–M12 |
| Final | Sunum, demo, savunma, rapor, GitHub | — |

---

## 12. Bekleyen kararlar ve açık sorular

- [ ] Yeni teslim ve sunum tarihi
- [ ] Kodun şu anki durumu (§4.2'yi işaretle)
- [ ] Birleşik CLAUDE.md repoya kondu mu
- [ ] MRR hocayla netleşti mi (önerilen kapsam: V-SDN sade controller + bulanık motor + öğrenme + rank/root + en az bir tekrarlanan deney)
- [ ] Özgün katkı seçimi (§5.2)
- [x] Tekrarlanacak deney(ler) — E1 ana; E2, E3, E4, E5, E8

---

## 13. Sıradaki adımlar

1. Birleşik CLAUDE.md'yi repoya koy; §4.2 ve §12'yi işaretle; yeni teslim tarihini yaz.
2. Claude Code ile M6'ya (baseline'lar uçtan uca), ardından M9'a (NF-MTAR tamam) kadar ilerle.
3. Katkıyı seç (§5.2) ve CLAUDE.md §6'ya yaz.
4. Literatür taraması (§10) — birlikte yapabiliriz.
5. Checkpoint 1–2 notlarını yaz; rapor iskeletine (§7) ve sunuma (§8) paralel başla.
