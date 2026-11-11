# CLAUDE.md — Bulanık Mantık Dönem Projesi (NF-MTAR Yeniden Gerçekleştirme)

Bu dosya, projenin bağlamını ve alınmış bütün tasarım kararlarını içerir (Parça 1–10).
Kod yazarken bu kararlara uy; bir karardan sapman gerekirse önce sor.
"BEKLEMEDE" işaretli bölümler henüz netleşmedi; o kısımları uydurma, sor.

Son güncelleme: 8 Kasım 2026

---

## 1. Proje bağlamı

- **Ders:** Bulanık Mantık dönem projesi (İstanbul Üniversitesi-Cerrahpaşa, Bilgisayar Mühendisliği)
- **Referans makale:** M. Shabana Parveen ve P. T. V. Bhuvaneswari, "A neuro-fuzzy multi-topology adaptive routing framework for QoS-aware healthcare IoT communications," *Scientific Reports*, vol. 16, art. 3330, 2026. doi: 10.1038/s41598-025-33235-9
- **Teslim:** ertelendi (yeni tarih öğrenciden alınacak). Classroom'a rapor, sunum, GitHub linki.
- **Puanlama:** Sunum 20, Yeniden gerçekleştirme 30, Özgün katkı 30, Rapor + GitHub 20.
- **Önemli kurallar (yönergeden):**
  - Git commit geçmişi geliştirme sürecini göstermeli; küçük ve anlamlı commit'ler at.
  - Öğrenci teslim edilen her satırı savunmada açıklayabilmeli. Kodu okunabilir yaz; her modülün başında
    hangi denklemi/kararı uyguladığını belirten docstring olsun.
  - Dış kaynaktan alınan kod varsa kaynağı yorumda ve README'de belirt.
  - Gizli bilgi (token, parola) repoya girmemeli.
  - Raporda "Generative AI Usage Disclosure" bölümü olacak; Claude Code kod geliştirmede kullanılıyor.

## 2. Makalenin özü (kısa)

NF-MTAR iki bileşenden oluşur:

1. **V-SDN:** Merkezi bir controller, fiziksel sensör ağını üç sanal topolojiye böler:
   Eliptik (E, kritik trafik), Doğrusal (L, düzenli trafik), Rastgele (R, kritik olmayan trafik).
   Her topolojiye Tablo 1'deki QoS politikası uygulanır.
2. **Neuro-fuzzy arama motoru:** Her topolojide düğümleri 5 girdiyle puanlar, rank hesaplar, en iyi düğümü
   **root** seçer, en düşük maliyetli yolu bulur. Parametreler geri yayılımla güncellenir.

Makalede COOJA + MATLAB + OpenDaylight kullanılmış; kod/veri paylaşılmamış. Kendi Python simülatörümüzü
yazıyoruz. Sayısal sonuçların makaleyle birebir tutması beklenmiyor; trendler ve göreli farklar karşılaştırılacak.
Makalede çok sayıda tanımsız parametre ve işaret/birim hatası var; aşağıdaki kararlar bunları düzeltir.

## 3. Teknoloji yığını

- Python 3.11+
- `numpy`, `networkx`, `matplotlib`, `pandas`, `scipy`, `pyyaml`, `torch` (eğitim için)
- Simülasyon: ayrık olay simülasyonu (`heapq` ile kendi olay kuyruğu veya `simpy`)
- Her koşu bir `seed` ile tekrarlanabilir (numpy Generator, networkx seed, torch seed).
- requirements.txt'te sürümler sabitlensin.

## 4. Repo yapısı (yönerge §9'a göre)

```
fuzzy-logic-project/
├── README.md
├── LICENSE
├── CLAUDE.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── default.yaml              # tüm simülasyon parametreleri
│   └── paper_reference.yaml      # makalenin raporladığı değerler (karşılaştırma için)
├── src/nfmtar/
│   ├── __init__.py
│   ├── config.py                 # YAML -> dataclass
│   ├── network/
│   │   ├── graph.py              # fiziksel ağ (unit disk graph), düğüm durumu
│   │   ├── radio.py              # 802.15.4 iletim, gecikme, kayıp, enerji
│   │   └── partition.py          # V_E, V_L, V_R + bağlantısallık kontrolü
│   ├── traffic.py
│   ├── controller.py             # V-SDN: politika, tetikleyiciler, histerezis, kararları uygulama
│   ├── router.py                 # kenar maliyeti, ters yönlü Dijkstra, iki aşamalı yol
│   ├── engine/
│   │   ├── base.py               # ScoringStrategy arayüzü
│   │   ├── baselines.py          # B1, B2, B3
│   │   ├── fuzzy.py              # bulanık çıkarım motoru (§5.14)
│   │   ├── oracle.py             # kâhin etiketleri (§5.16)
│   │   ├── trainer.py            # eğitim (§5.16)
│   │   └── rank.py               # rank ve root seçimi (§5.17)
│   ├── metrics.py
│   └── simulation.py             # kapalı döngü ana simülasyon
├── experiments/
│   ├── build_training_data.py
│   ├── train.py
│   ├── baseline/                 # baseline koşu scriptleri
│   ├── proposed/                 # önerilen yöntem koşu scriptleri
│   └── analysis/                 # istatistik, grafik, makale karşılaştırması
├── data/                         # üretilmiş eğitim verisi (büyükse .gitignore + üretim talimatı README'de)
├── results/
│   ├── figures/
│   ├── tables/
│   └── models/                   # eğitilmiş bulanık parametreler (json)
├── tests/
├── presentation/presentation.pdf
├── report/report.pdf
└── references/README.md
```

## 5. Tasarım kararları

### 5.1 Fiziksel ağ
- Alan: varsayılan 300 m × 300 m (makale vermiyor; config'den).
- Düğüm sayısı: 100–200 (varsayılan 150). Radyo menzili: 50 m (makaleden).
- Fiziksel topoloji: unit disk graph — mesafesi ≤ 50 m olan düğümler arasında kenar
  (COOJA'nın varsayılan UDGM radyo ortamıyla benzer).
- Yeniden üretim ölçütü (strict modda): her topolojinin indüklenmiş alt çizgesi kendi edge router'ına bağlı
  olmalı; değilse `default_rng([seed, deneme])` ile yeniden üret. Fiziksel ağın tamamının bağlı olması strict
  modda aranmaz (yollar zaten topoloji içinde). shared modda (§5.9) ayrıca fiziksel ağın bağlılığı aranır.
  Deneme sayısı loglanır ve runs.csv'ye yazılır (seçim yanlılığı raporlanabilsin).
  E0 analizinde (`enforce_partition_connectivity: false`) yeniden üretim yapılmaz, yalnızca ölçülür.

### 5.2 Yerleşim ve topoloji isimlerinin anlamı
Makale "eliptik/doğrusal/rastgele" topolojileri biçimsel olarak tanımlamıyor. Yorumumuz (hastane senaryosu):
- **Eliptik (kritik):** düğümler alan içindeki bir elips bölgede (yoğun bakım) sıkı küme halinde.
- **Doğrusal (düzenli):** düğümler bir koridor boyunca (dar dikdörtgen şerit).
- **Rastgele (kritik olmayan):** düğümler Edge-R'ye bitişik dikdörtgen bir bölgede düzgün rastgele
  (`r_placement: region`, varsayılan). Bölge alanı A_R = n_R · π · R² / r_target_degree
  (varsayılan r_target_degree = 10; 45 düğüm için ≈ 35.000 m², ör. ~190 × 185 m). Bölge boyutları E2'de
  yoğunluk sabit ölçeklemeyle birlikte ölçeklenir.
  `r_placement: uniform` (bütün alana düzgün) YALNIZCA E0 analizi için korunur. E0 bulgusu (30 seed, yeniden
  üretimsiz): uniform yerleşimde R sensörlerinin ortalama 32,1/45'i (%71; en az 8, en çok 45) kendi alt çizgesinde
  kopuk; kopukların yalnızca %42'sinin fiziksel yolu var; fiziksel ağ seed'lerin yalnızca %7'sinde bağlı; zorunlu
  yeniden üretimde 100 denemede başarı %23. Region yerleşimde kopuk R ortalama 0,53/45 (%1,2), zorunlu yeniden
  üretim ortalama 0,23 deneme (en çok 2), başarı %100. İlk koşudaki 19,5/45 değeri fiziksel bağlılığa koşullanmış
  bir örneklemdi (seçim yanlılığı örneği; raporda kullanılacak). Makalenin tartışmadığı risk.
  Region modunda R alt çizgesinin etkin ortalama derecesi ~7,7 (r_target_degree = 10, sınır etkisi hariç nominal).
  R bölgesi E elipsiyle fiziksel olarak örtüşebilir: strict modda yolları etkilemez, paylaşılan kanalda sınıflar
  arası çekişmeyi etkiler (bilinçli, gerçekçi; §5.5).
Sınıf oranları config'den (varsayılan E/L/R = 0.3/0.4/0.3).

### 5.3 Bölümleme
- Atama **deterministik**: her sensör ürettiği trafik sınıfına göre bir topolojiye girer
  (makaledeki Denk. 2'nin olasılıkları tanımsız; one-hot kabul ediyoruz).
- **Basitleştirici varsayım:** her düğüm yalnızca tek trafik sınıfı üretir.
- Bölümler **ayrık** (makaledeki gibi). Her topolojinin indüklenmiş alt çizgesinin kendi edge router'ına
  bağlı olduğu kontrol edilir; değilse uyarı + log (kopuk düğüm sayısı metrik olarak kaydedilir).
- Kod, örtüşen bölümlemeye genişletilebilir olsun (node -> set(topology)).

### 5.4 Root ve edge router
- Her topolojinin **sabit** bir edge router'ı var (Edge-E, Edge-L, Edge-R), alan kenarına yakın.
- **Root** = edge router'dan ayrı, topoloji içinden seçilen koordinatör (küme başı benzeri).
  Veri yolu iki aşamalı: sensör -> root -> edge router. Root veri birleştirme YAPMAZ (makalede yok).

### 5.5 Radyo, gecikme, enerji
- IEEE 802.15.4: 250 kbit/s, çerçeve ≤ 127 B, çerçeve başına veri ~90 B (config). Büyük paketler
  parçalanır; bir parça kaybolursa paket kaybolur.
- Atlama başına gecikme = kuyruk bekleme + parça_sayısı × (çerçeve_bit / 250000)
  + CSMA/CA geri çekilme (rastgele, 0–7 × 320 µs, parça başına) + işlem süresi (config).
- Gecikme bileşenleri (kuyruk, iletim, geri çekilme, işlem) paket logunda ayrı tutulsun.
- **Paylaşılan kanal, yerel çekişme:** kanal ağ genelinde tek bir 250 kbit/s havuz DEĞİLDİR (mekânsal yeniden
  kullanım olmazsa ağ anında doyar). Bir düğüm, taşıyıcı algılama menzilindeki (`cs_range`, varsayılan 50 m) başka
  bir düğüm iletim yaparken iletime başlayamaz; CSMA/CA geri çekilmesiyle yeniden dener. Gizli terminal
  çarpışmaları modellenmez; kayıplar PRR ile temsil edilir (COOJA UDGM'ye benzer basitleştirme).
- **Sınıf önceliği MAC'te (π1, π2):** strict modda bir düğüm yalnızca kendi sınıfını taşıdığından sınıflar ancak
  paylaşılan ortamda karşılaşır. Öncelik 802.11e EDCA benzeri iki mekanizmayla uygulanır:
  (a) sınıf bazlı geri çekilme: macMinBE kritik 2, düzenli 3, kritik olmayan 5; macMaxBE 5;
  (b) sınıf bazlı AIFS: kanal boş algılandıktan sonra geri çekilme sayımına başlamadan önce sabit ek bekleme
  (`aifs_slots`, birim geri çekilme periyodu cinsinden; kritik 0, diğerleri kalibrasyonla belirlenir).
  Yalnızca (a) ile doygun yükte ulaşılan pay 0,382 / 0,392 / 0,226 (kalibrasyon CSV'si; hedef 0,80 / 0,15 / 0,05);
  bu yüzden (b) eklendi.
  Geri çekilme aralığını daraltmak tercih edilmedi: çarpışmalar modellenmediği için küçük BE'nin doğal bedeli yok,
  ayarı keyfi olur; AIFS ise güçlü sınıf ayrımı için standarttaki yerleşik mekanizma.
  **Kalibrasyon** (`experiments/calibration/mac_aifs.py`): doygun üç gönderici (sınıf başına bir, hepsi cs menzilinde,
  20 s, 5 seed); düzenli AIFS ∈ {0, 1, 2, 3, 4, 5, 6}, kritik olmayan AIFS ∈ {2, 4, 5, 6, 7, 8, 10}; hedefe L1
  uzaklığı en küçük çift seçilir ve config'e yazılır; tablo `results/tables/mac_aifs_calibration.csv`. Seçilen değer
  için doğruluk kaynağı config'tir. 80/15/5'in tam tutması beklenmez; en yakın değer ve kalan sapma raporlanır.
  (İlk ızgarada en iyi çift 4/7, L1 = 0,175; ızgara dışı 5/7 denemesi L1 = 0,058 verdiği için ızgara inceltildi.)
  **Seçilen:** düzenli 5, kritik olmayan 7 (paylar 0,814 / 0,165 / 0,021; L1 = 0,058). Seçim önceden tanımlı L1
  ölçütüyle yapıldı. 5/8 (L1 = 0,069; doygunlukta kritik olmayan düşmesi 0,14'e karşı 0,23) neredeyse eşdeğer;
  duyarlılık olarak raporlanacak. Üç hedefi aynı anda tutan nokta yok; düzenli payı AIFS 5→6'da 0,16'dan 0,08'e
  düşüyor (ayar ayrık ve hassas). L1 ısı haritası raporda şekil olarak kullanılacak.
  **Doygun gönderici zamanlama kuralı:** bir sonraki paket, önceki iletimin (başarılı ya da kayıp) MAC bitişinden
  işlem süresi (`radio.processing_delay_s`, 1 ms) sonra hazır olur; tek kaynak `RadioModel.next_packet_ready_s`. Kalibrasyon bu kurala bağlıdır: boşluksuz modelde katı öncelik, yüksek AIFS'li sınıfları tamamen aç
  bırakıyor (EDCA'nın bilinen açlık davranışı). `simulation.py`'deki düğüm servis döngüsü AYNI işlem süresi kuralını
  kullanmalıdır; aksi halde kalibrasyon simülasyona taşınmaz.
  **AIFS'nin bedeli:** AIFS boş kanalda da her çerçeveye eklenir (düzenli +1,28 ms, kritik olmayan +2,24 ms;
  1024 B'lik 12 parçalı paket atlama başına ~+27 ms). `HopResult.aifs_s` ayrıca loglanır; raporda önceliğin gecikme
  bedeli olarak gösterilir. Doygunlukta kritik olmayan paketlerin bir kısmı (π10 = 1 nedeniyle) kanal erişim
  hatasıyla düşer; E1'in yüksek yük ayağında yorumlanırken dikkate alınacak.
  **Yorum:** π1 doygun yükte bir airtime öncelik hedefidir; doygun olmayan yükte her sınıf sunduğu yük kadar pay
  alır. Değerlendirmenin asıl ölçütü sınıf bazlı gecikme, PDR ve QoS memnuniyetidir; airtime payı ek metriktir.
  B1-plain: bütün sınıflar BE 3, AIFS 0, tek FIFO kuyruk. Gözlem: öncelik yokken airtime'ın en büyük payını büyük
  paketli kritik olmayan sınıf alıyor (0,234 / 0,340 / 0,426; kalibrasyon CSV'si) — politikanın gerekçesi olarak
  raporda kullanılacak.
- Modellenmeyenler (sınırlılık): ACK çerçeveleri, CCA süresi, boşta dinleme enerjisi. Mutlak ömür değerleri
  bu yüzden iyimserdir; yöntemler arası karşılaştırma aynı varsayımla yapıldığı için adil kalır.
- shared modda aktarıcı düğümde sınıf başına kuyruk, π2'ye göre katı öncelikli servis.
- Link kalitesi: mesafeye bağlı PRR, menzile yaklaştıkça düşen lojistik fonksiyon (parametreler config).
- Enerji: birinci dereceden radyo modeli (E_elec = 50 nJ/bit, ε_amp = 100 pJ/bit/m²);
  başlangıç enerjisi config (varsayılan 2 J). Enerjisi biten düğüm ölür.

### 5.6 Trafik — `traffic.py`

| Sınıf | Paket boyutu | Varış süreci (load_multiplier = 1) | Gerekçe |
|---|---|---|---|
| Kritik (E) | 64 B | Poisson, 1 paket/s + kritik bakım epizotları | Olay güdümlü alarmlar; makale: kritik bakımda 10 paket/s'ye kadar |
| Düzenli (L) | düzgün 128–256 B | Periyodik 1 paket/s, rastgele faz, ±%10 düzgün titreşim | Periyodik yaşamsal bulgu örneklemesi; makale: rutin kayıtlar 1 paket/s |
| Kritik olmayan (R) | düzgün 512–1024 B | Poisson, 0,2 paket/s | Arka plan senkronizasyonu, yedekleme |

- **Kritik bakım epizotları:** her kritik düğüm için epizot başlangıçları Poisson (ortalama 5 dakikada bir), süre
  30 s; epizot boyunca hız 10 paket/s (makaledeki üst sınır). Amaç: (1) makalenin kritik bakım senaryosunu temsil
  etmek, (2) trafik değişimi tetikleyicisini (§5.18, tetikleyici 2) gerçekten sınamak — durağan trafikte o tetikleyici
  hiç çalışmaz. Config: `critical_episodes: {enabled: true, mean_interval_s: 300, duration_s: 30, rate_pps: 10}`.
- `load_multiplier` yalnızca temel hızları ölçekler (epizot hızını değil). Varsayılanlar M6'daki kısa koşularla
  kalibre edilecek: hedef, orta yükte (1,0) baseline'ların ne çöktüğü ne de hep %100 PDR verdiği bir çalışma noktası.
- **Ortak rastgele sayılar (common random numbers):** trafik, yöntemden BAĞIMSIZ olarak seed'den önceden bir varış
  izi (trace) olarak üretilir ve bütün yöntemlere aynen verilir; her düğüm kendi RNG akışını kullanır
  (`numpy.random.SeedSequence(seed).spawn`). Böylece aynı seed'de bütün yöntemler birebir aynı paketleri görür;
  eşleştirilmiş istatistiğin (§5.19) gücünü artırır ve bir sınıfın parametresini değiştirmek diğerlerinin izini kaydırmaz.
- Paket kaydı: id, kaynak, sınıf, boyut, oluşturulma zamanı, son tarih (oluşturulma + π3; QoS memnuniyeti için).

### 5.7 Tablo 1 politika parametreleri

| Parametre | E | L | R | Kullanım |
|---|---|---|---|---|
| π1 Bant genişliği payı (%) | 80 | 15 | 5 | Hedef; MAC'te sınıf bazlı geri çekilmeyle yaklaşık uygulanır, ulaşılan airtime payı ölçülür (§5.5) |
| π2 Öncelik (1–7) | 7 | 4 | 1 | MAC geri çekilme sırası; shared modda aktarıcı kuyruk önceliği (§5.5) |
| π3 Maks. gecikme (ms) | 25 | 100 | 500 | Metrik eşiği |
| π4 Min. throughput (%) | 95 | 85 | 70 | Metrik eşiği |
| π5 Kayıp toleransı (%) | 0.1 | 1 | 5 | Metrik eşiği |
| π6 Enerji verimliliği ağırlığı | 0.3 | 0.5 | 0.8 | Yol maliyetinde λ3 (§5.9) |
| π7 Güvenlik seviyesi | — | — | — | Kapsam dışı |
| π8 Yedeklilik | — | — | — | Kapsam dışı |
| π9 Tampon (paket) | 16 | 8 | 4 | Kuyruk kapasitesi (makaledeki 512/256/128 KB sensör RAM'iyle uyumsuz; 4:2:1 oranı korundu) |
| π10 Yeniden iletim limiti | 5 | 3 | 1 | MAC yeniden deneme |
| π11 Rota güncelleme (s) | 1 | 5 | 30 | Periyodik yeniden değerlendirme (§5.18) |
| π12 QoS izleme (ms) | 100 | 1000 | 10000 | Metrik örnekleme aralığı |

**Makaledeki Denk. 4 (QoS ağırlıklı toplamı) KODLANMAZ** (normalizasyonsuz toplam ters sonuç verir).
Yerine ölçülen **QoS memnuniyet oranı**: her sınıf için π3/π4/π5 eşiklerini sağlayan paket/pencere oranı.

### 5.8 Controller (V-SDN)
- Merkezi, global ağ görünümüne sahip.
- **Karar ile uygulama ayrı:** engine root/yol kararını hesaplar, controller uygular.
- Kontrol mesajı maliyeti varsayılan sıfır (raporda sınırlılık); `control_overhead` seçeneği için yer bırak.

### 5.9 Yol seçimi — `router.py`
Kenar maliyeti (u -> v), tüm terimler [0,1]'e normalize:
  cost(u,v) = λ1 · d(u,v)²/R²  +  λ2 · (−log PRR(u,v)) / L_max  +  λ3 · depletion(u)
- d²: birinci dereceden radyo modelinde iletim enerjisi vekili (R = 50 m)
- −log PRR: toplamı = −log(Π PRR) -> minimizasyon = en yüksek uçtan uca teslim olasılığı
  (L_max = −log(PRR_min), PRR_min config, ör. 0.05)
- depletion(u) = 1 − kalan/başlangıç enerji
- λ3 = π6 (E: 0.3, L: 0.5, R: 0.8); λ1, λ2 config (varsayılan eşit; λ'lar toplamı 1'e normalize)
- B2: −log PRR yerine ETX = 1/PRR (normalize), λ1 = 0 (MRHOF benzeri)
- B1: kenar maliyeti = 1 (atlama sayısı)
- Dijkstra, topoloji alt çizgesinde, TERS yönde tek kaynaklı:
  edge router'dan -> tüm adayların H ve C değerleri; root'tan -> tüm sensörlerin yolları.
- `relay_mode: {strict, shared}`, varsayılan strict (makaleye sadık: yol yalnızca kendi topolojisinin alt
  çizgesinde). shared: yol fiziksel ağda aranır, diğer topolojilerin düğümleri yalnızca aktarıcıdır; root
  seçimi ve politika yine kendi topolojisine aittir; aktarıcı düğümde paket kendi sınıfının π1/π2 kuralıyla
  kuyruğa girer. shared YALNIZCA ablasyon/katkı (ayrık vs örtüşen bölümleme) ve E0 içindir.

### 5.10 Kapalı döngü simülasyon
Her topoloji için, tetikleyici geldiğinde (§5.18):
1. Ölç: canlı adayların özellik vektörleri
2. Karar: engine skor -> rank -> root önerisi
3. Uygula: controller root/yolları günceller (histerezis kuralıyla)
4. İlet: trafik üretimi ve iletim sürekli devam eder; enerji, kuyruk, kayıp güncellenir
5. Geri besle: metrikler toplanır

### 5.11 Engine arayüzü
Bütün yöntemler aynı arayüzü uygular (yalnızca engine değişir; ağ, trafik, metrik aynı):

```python
class ScoringStrategy(Protocol):
    name: str
    def score(self, features: NodeFeatures) -> ScoreOutput: ...  # score: yüksek = root için uygun
    def dynamic_root(self) -> bool: ...
```
`NodeFeatures` tanımı §5.14'te. `ScoreOutput`: score, confidence (opsiyonel), top_rule (opsiyonel).

### 5.12 Yöntemler

| Yöntem | Puanlama | Root | Yol maliyeti | Öğrenme | Histerezis | V-SDN politikası |
|---|---|---|---|---|---|---|
| B1-plain | yok | sabit | atlama sayısı | yok | — | YOK (tek FIFO kuyruk, sınıf önceliği yok) |
| B1 hop | yok | sabit | atlama sayısı | yok | — | var |
| B2 energy | yok | sabit | ETX + depletion | yok | — | var |
| B3 fuzzy_untrained | bulanık, uzman param. | dinamik | §5.9 | yok | var | var |
| NF-MTAR | bulanık, eğitilmiş | dinamik | §5.9 | var | var | var |

- B1-plain = "geleneksel RPL benzeri" referans; V-SDN politikalarının etkisini ayırmak için.
- Sabit root (B1-plain, B1, B2): başlangıçta topolojinin edge router'a en yakın düğümü; root ölürse
  aynı kuralla yenisi seçilir (baseline'ların yapay çöküşünü önleyen adillik kararı).

### 5.13 Metrikler — `metrics.py`
- PDR ve normalize throughput (iletilen/üretilen; sınıf bazında ve toplam) + mutlak throughput (kbit/s)
- Uçtan uca gecikme: ortalama, medyan, 90/95/99. yüzdelik, CDF (sınıf bazında)
- Gecikme bileşenleri (kuyruk / iletim / geri çekilme / işlem)
- Sınıf bazında ulaşılan airtime payı (π1 hedefiyle karşılaştırma)
- Toplam enerji; ağ ömrü FND ve HND
- QoS memnuniyet oranı (§5.7)
- Φ (makale Denk. 6'nın düzeltilmiş hali, α/β/γ config'den) — değerlendirme metriği
- Root değişim sayısı (topoloji başına); root görevi yapmış düğümlerin enerji eğrileri (sıcak nokta analizi)
- Kopuk düğüm sayısı (bölümleme bağlantısallığı)
- Karar turu başına çalışma süresi (ms)

### 5.14 Bulanık çıkarım motoru — `engine/fuzzy.py`

**Sistem tipi:** Sıfırıncı dereceden Takagi-Sugeno (makalenin Denk. 12'si μ_Bk(y_k)=1 olduğu için
merkez-ortalama durulaştırıcıya indirgenir).

**Girdiler (`NodeFeatures`; hepsi sabit normalizasyonla [0,1]):**
| Girdi | Tanım | Terimler | Yön |
|---|---|---|---|
| x1 talep | Aynı topolojideki 2-atlama komşuluğunda üretilen trafik hızı / topoloji toplam trafik hızı | 5 | Yüksek = iyi |
| x2 kullanım | Kuyruk doluluk oranı | 5 | Yüksek = kötü |
| x3 tükenme | 1 − kalan_enerji / başlangıç_enerjisi | 5 | Yüksek = kötü |
| x4 link kalitesi | Topoloji içi komşulara ortalama PRR | 2 | Yüksek = iyi |
| x5 bağlantısallık | Topoloji içi komşu sayısı / 15, 1'de kırp | 3 | Yüksek = iyi |
`NodeFeatures` ayrıca edge router'a atlama mesafesini taşır (rank için). Normalizasyon fuzzy.py içinde.

**Üyelik fonksiyonu (makale Denk. 9; MATLAB gaussmf DEĞİL):** μ(x) = exp(−((x − c)/σ)²)
Eşit aralıklı merkezler, komşu terimler 0.5'te kesişir: σ = (Δ/2)/sqrt(ln 2)
- 5 terim: c = [0, .25, .5, .75, 1], σ ≈ 0.150
- 3 terim: c = [0, .5, 1], σ ≈ 0.300
- 2 terim: c = [0, 1], σ ≈ 0.601

**Kural tabanı:** 5×5×5×2×3 = 750 kuralın TAMAMI. Kural sonucu y_k uzman fonksiyonuyla:
- Terim indeksi -> iyilik g ∈ [0,1] (eşit aralıklı; "yüksek=kötü" girdilerde ters)
- y_k = 0.1 + 0.9 × Σ_i a_i · g_i ; a = {x1: 0.15, x2: 0.20, x3: 0.30, x4: 0.25, x5: 0.10} (config)

**Çıkarım:**
- w_k = Π_i μ_{i, term(k,i)}(x_i)          (çarpım t-normu, Denk. 11)
- f = Σ w_k y_k / Σ w_k                     (Denk. 12, sadeleşmiş)
- f_norm = clip((f − 0.1)/0.9, 0, 1)        (Denk. 13a)
- confidence = max_k w_k / Σ w_k            (Denk. 13b)

**Uygulama:** NumPy vektörize (750×5 kural-terim indeks tablosu), toplu değerlendirme.
`evaluate(features) -> FuzzyOutput(score, confidence, top_rule_index)`.
Kuralı insan okunur metne çeviren fonksiyon ("IF x1 is High AND ... THEN 0.82") — canlı demo için.
Opsiyonel `pruned=True`: bir girdinin tüm terimleri için aynı sonucu veren kuralları birleştir ("don't care").
Eğitim için PyTorch eşdeğeri (aynı matematik) `fuzzy.py` içinde veya ayrı sınıf olarak.

**Testler (tests/test_fuzzy.py):** üyelikler [0,1]; merkezde 1; komşular 0.5'te kesişir; monotonluk
(x3 artınca skor artmaz, x4 artınca azalmaz); kalibrasyon raporu (makalenin Kural 1/47/98 girdileri terim
merkezlerinde -> makalede 0.95/0.78/0.15; fark log'a, test başarısız olmasın); güven ∈ [1/750, 1].

### 5.15 B3 baseline — eğitilmemiş bulanık
§5.14 motoru başlangıç parametreleriyle; rank (§5.17) ve histerezis (§5.18) NF-MTAR ile aynı.

### 5.16 Nöral öğrenme — `engine/oracle.py`, `engine/trainer.py`

**Öğrenilen parametreler:** θ = {c, σ, y_k} (ANFIS tarzı; ayrı sinir ağı YOK).
Parametreleştirme: σ = softplus(s) + 1e-3 ; y_k = 0.1 + 0.9 · sigmoid(u_k)

**Hedef: kâhin etiketleri, çevrimdışı eğitim.** `oracle.py`, bir ağ durumu ve topoloji için her aday root r'ye:
1. Toplama kalitesi: topolojideki her sensörden r'ye §5.9 maliyetiyle Dijkstra; teslim = Π PRR,
   gecikme = Σ atlama gecikmesi tahmini; sensör trafik hızlarıyla ağırlıklı ortalama.
2. Dayanıklılık: r'nin kalan enerjisi / (r root olduğunda tahmini tüketim hızı).
3. Skor = 0.4·teslim + 0.3·(1 − gecikme_norm) + 0.3·dayanıklılık_norm (config);
   aynı durumdaki adaylar arasında min-max ile [0.1, 1.0].
NOT: Kâhin root -> edge router yolunu İÇERMEZ (o kısım rank'taki H ve C'de).

**Veri üretimi** (`experiments/build_training_data.py`): B1 veya B3 ile simülasyon, her K saniyede ağ durumu
kaydı (ömür sonuna kadar); her durum × topoloji × aday için (5 özellik, kâhin skoru); birden çok seed.
Eğitim/doğrulama ayrımı **SEED BAZINDA** (veri sızıntısı önlemi).

**Kayıp:** E = (1/2N) Σ (y_oracle − f)² + (λ/2) Σ (θ − θ0)²   (θ0 = §5.14 uzman başlangıcı)
`reg_mode: {anchored, zero_l2, none}`, λ config (varsayılan 1e-3).

**Optimizasyon:** SGD, momentum 0.9; η(epoch) = 0.01·exp(−epoch/500); erken durdurma;
çevrimiçi ince ayar opsiyonel (η_min = 5e-4). Model -> `results/models/<run_id>.json`.

**Değerlendirme (doğrulama):** MSE, Spearman, top-1 uyumu, pişmanlık (kâhin_en_iyi − kâhin_seçilen),
terim sırası bozulmaları (komşu c'lerin sırası bozulan çift sayısı), öğrenme eğrisi. B3 ve eğitilmiş için.

**Testler:** ∂f/∂y_k = w_k/Σw elle hesap vs autograd; σ > 0, y_k ∈ [0.1,1]; bir düğümün enerjisi düşünce
kâhin skoru artmamalı.

### 5.17 Rank ve root seçimi — `engine/rank.py`
R(v) = w1·(1 − f_norm(v)) + w2·H(v)/H_max + w3·C(v)/C_max ; r* = argmin R
- (1 − f): makaledeki işaret hatası düzeltmesi
- H: v'den edge router'a atlama; C: v'den edge router'a yolun §5.9 maliyeti
- H_max, C_max: o turdaki adaylar arasında en büyük
- Topolojiye özgü sabit ağırlıklar (config): E (0.4, 0.4, 0.2); L (0.5, 0.3, 0.2); R (0.6, 0.2, 0.2)
- Adaylar: topolojideki canlı ve edge router'a yolu olan düğümler
- Her aday için (R, f_norm, güven, H, C) loglanabilir

### 5.18 Yeniden değerlendirme ve root değiştirme — `controller.py`
Tetikleyiciler (herhangi biri):
1. Periyodik: π11 doldu
2. Trafik: herhangi bir sınıfta |λ(t) − λ(t−Δt)| / max(λ(t−Δt), ε0) > τ_traffic (varsayılan 0.3)
3. Root sağlığı: root öldü VEYA tükenme > 0.9 VEYA edge router'a yolu yok (makalede YOK; eklendi)

Değiştirme kuralı:
- Root sağlığı tetikleyicisi: r* hemen seçilir (marj/bekleme atlanır)
- Aksi halde değiştir ⇔ R(current) − R(r*) > ε VE son değişiklikten ≥ min_dwell VE
  (confidence_gate kapalı VEYA güven(r*) ≥ conf_min)
- Varsayılan: ε = 0.05, min_dwell = 2 × π11, confidence_gate = false
- Her root değişimi loglanır (zaman, topoloji, eski/yeni, R farkı, tetikleyici)
Ablasyon bayrakları: ε ∈ {0, 0.02, 0.05, 0.1, 0.2}, confidence_gate, root_health_trigger.

### 5.19 Deneyler — `experiments/`

**Genel düzen:**
- Bütün yöntemler aynı ağ, trafik ve seed'lerle. Seed listesi config'de.
- Isınma süresi (`warm_up_s`, varsayılan 60 s) metriklerden hariç.
- Her koşu `results/tables/runs.csv`'ye bir satır (run_id, deney, yöntem, seed, parametreler, metrikler).
  Paket logu (opsiyonel parquet): oluşturma/teslim zamanı, sınıf, atlama sayısı, gecikme bileşenleri, kayıp nedeni.
- Ana karşılaştırmada B1, B2, B3, NF-MTAR aynı V-SDN bölümlemesi ve zamanlamasını (π1, π2, π9, π10) paylaşır;
  fark yalnızca root/yol/puanlama. B1-plain politika etkisini ayırmak için.

**E0 — Bağlantısallık analizi (ayrık bölümlemenin bedeli; makalede tartışılmamış):**
- r_placement ∈ {uniform, region} × 30 seed, `enforce_partition_connectivity: false` (yeniden üretim yok)
- Topoloji başına kopuk sensör sayısı (ort / min / max), kopukların fiziksel yolu olanların oranı,
  alt çizge ortalama derecesi; region modunda ayrıca zorunlu yeniden üretimdeki deneme sayısı (ort / max)
- Opsiyonel (router.py sonrası): uniform yerleşimde strict vs shared ile R sınıfı PDR'si (kısa koşu, 10 seed)
- Çıktı: results/tables/e0_connectivity.csv; results/figures/e0_topology_uniform_vs_region.png (yan yana)

**E1 — Trafik türüne göre throughput/PDR/gecikme (makale Şekil 4a, 5a, Tablo 4) [ANA TEKRAR DENEYİ]:**
- N = 150, 300×300 m, simülasyon 600 s, 30 seed
- load_multiplier ∈ {0.5, 1.0, 2.0}
- Yöntemler: B1-plain, B1, B2, B3, NF-MTAR
- Çıktı: sınıf bazında ortalama ± %95 GA; dağılım histogramları, CDF, yüzdelikler (Şekil 4d–e, 5c–e eşdeğeri)

**E3 — Atlama sayısına göre gecikme (Şekil 6d):** Ek koşu yok; E1 paket loglarından atlama sayısına göre
gruplanmış gecikme ve atlama başı eğim (ms/atlama).

**E2 — Ölçeklenebilirlik (Şekil 6b):** N ∈ {20, 50, 100, 150, 200}; YOĞUNLUK SABİT:
alan kenarı = 300·sqrt(N/150) m; 20 seed; load_multiplier = 1.0; yöntemler: B1-plain, B1, B3, NF-MTAR.

**E8 — Ağ ömrü (Tablo 4):** başlangıç enerjisi 0.5 J (makul sürede ölüm için), FND ve HND'ye kadar;
15 seed; yöntemler: B1, B2, B3, NF-MTAR. Root görevli düğümlerin enerji eğrileri kaydedilsin.

**E4 — Paket boyutu (Şekil 6c, opsiyonel ama önerilir):** kritik paket boyutu ∈ {64, 128, 256, 512, 1024} B; 20 seed;
gecikme–boyut eğimi (ms/bayt); teorik tek atlama eğimi 8/250000 s = 0.032 ms/bayt ile karşılaştır.

**E5 — Root arıza enjeksiyonu (makalenin motivasyon senaryosu; makalede TEST EDİLMEMİŞ):**
- t = 300 s'de her topolojinin o anki root'u zorla öldürülür (enerji = 0)
- Yöntemler: B1 (arıza anında yeni sabit root), NF-MTAR (root_health_trigger açık),
  NF-MTAR-paper (root_health_trigger kapalı: ölü root yalnızca sonraki periyodik/trafik tetikleyicide fark edilir;
  R(ölü root) = ∞; min_dwell kuralı geçerli)
- Ölçütler: topoloji başına kurtarma süresi (arızadan sonra ilk başarılı uçtan uca teslim), arıza penceresinde
  (300–330 s) kayıp paket sayısı, pencere PDR'si
- 30 seed; canlı demoda da kullanılacak (tek tuşla root öldürme)

**Makale karşılaştırması** (`experiments/analysis/paper_comparison.py`):
`configs/paper_reference.yaml`'daki değerlerle `results/tables/paper_comparison.csv`:
metrik | makale (NF-MTAR, RPL) | bizim (NF-MTAR, B1-plain) | göreli iyileşme (makale / bizim) | trend uyumu.
paper_reference.yaml içeriği:
- Throughput % (kritik/düzenli/kritik olmayan): NF-MTAR 95.2/87.3/78.9; RPL 71.4/68.2/62.1
- Gecikme ms: NF-MTAR 18.4/45.2/89.7; RPL 67.3/94.8/156.2
- Tablo 4: NF-MTAR throughput 94.3, gecikme 18.4, PDR 98.7, QoS 97.3; TSFIS-GWO 86.4/38.9/97.2/93.8
- Ölçeklenebilirlik (NF-MTAR, ms): 20→12, 50→15, 100→20, 150→24, 200→28
- Atlama (ms): NF-MTAR 1→8, 2→12, 3→16, 5→24, 10→42; RPL 1→28, 2→48, 3→72, 5→120, 10→200
- Paket boyutu (NF-MTAR, ms): 64→9, 128→13, 256→17, 512→23, 1024→29

**İstatistik:** Bütün yöntemler aynı seed'lerle koştuğu için gözlemler EŞLEŞTİRİLMİŞTİR (seed = blok).
- İkili karşılaştırma: eşleştirilmiş t-testi + Wilcoxon işaretli sıralar testi; etki büyüklüğü d_z
  (fark ortalaması / fark std) ve %95 GA
- Çok gruplu: Friedman testi (tekrarlı ölçüm; normallik/eşit varyans gerekmez)
- Çoklu karşılaştırma: Holm–Bonferroni
- Tablolarda birlikte: ortalama fark, %95 GA, düzeltilmiş p, etki büyüklüğü, n
- Tutarlılık kontrolü (makalede bulunan hatalara düşmemek için): raporlanan t = fark/SE; etki büyüklüğü t ve n
  ile tutarlı; test istatistikleri teorik aralıklarında
- Not: anlamlılık yalnızca seed'ler arası değişkenliği ölçer; model geçerliliği (fiziksel tutarlılık) ayrıca tartışılır

**Grafikler (results/figures/):** e1_throughput_by_class, e1_latency_by_class, e1_latency_cdf,
e1_latency_percentiles, e3_latency_vs_hops, e2_scalability, e8_lifetime, e8_root_energy, e4_packet_size,
e5_root_failure, training_curves.

## 6. BEKLEMEDE

- Makale okuması tamamlandı (Parça 1–10); deney ve istatistik kararları §5.19'da.
- **Özgün katkı seçimi:** öneri — ana katkı K1 "yorumlanabilirliği koruyan eğitim"
  (reg_mode: anchored vs zero_l2 vs none; λ taraması; ölçütler: pişmanlık, top-1, terim sırası bozulması,
  E1 ağ metrikleri). Destekleyici ablasyonlar: girdi ablasyonu (her girdiyi sırayla çıkar) ve ε taraması
  + güven kapısı. Yedek ana katkı: Mamdani (centroid) vs Sugeno (merkez-ortalama).
  ÖĞRENCİ KARAR VERMEDEN katkı deneylerini kodlama; altyapıyı (bayraklar, reg_mode) hazır tut.

## 7. Çalışma biçimi

- Her modül sonrası küçük bir test (tests/) ve kısa açıklama.
- Commit mesajları anlamlı; tek dilde tutarlı.
- Sonuçlar CSV -> results/tables/, grafikler PNG -> results/figures/.
- Tüm parametreler configs/default.yaml'dan; koda sabit gömme.
- Önerilen geliştirme sırası: ağ+bölümleme -> radyo/enerji -> trafik -> router -> metrik+simülasyon
  -> B1-plain/B1/B2 -> fuzzy+B3 -> oracle+eğitim -> rank+controller (NF-MTAR) -> E1 -> E3/E5/E2/E8/E4 -> katkı.
