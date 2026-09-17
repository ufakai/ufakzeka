"""Hand-written everyday conversation sets, added 2026-08-31 after probing v14 with realistic user prompts
(scripts/everyday_probe.py). Every fact here is checked; keep it that way when adding entries.

Sets: recipes, jokes, riddles (with answer follow-ups), health basics, timeless popular-culture lists,
common English words, capability limits (no alarms, no orders), social replies, revision follow-ups."""

from __future__ import annotations

# Common Turkish recipes, short and correct. (v14 told users to squeeze rice in cheesecloth.)
RECIPES = [
    ("nasıl pilav yapılır", "Pirinç pilavı: 1 su bardağı pirinci yıkayıp süz. 2 kaşık tereyağında 1-2 dakika kavur, 1,5 su bardağı sıcak su ve tuz ekle. Kapağı kapat, kısık ateşte suyunu çekene kadar 12-15 dakika pişir, 10 dakika demlendir."),
    ("menemen tarifi ver", "Menemen: 2 domatesi rendele, 2 yeşil biberi doğra. Biberleri yağda soteleyip domatesi ekle, suyunu çekince 3 yumurtayı kır, karıştırarak pişir. Tuz ve pul biber ekle. Soğanlı seven kavrulmuş soğanla başlar."),
    ("mercimek çorbası nasıl yapılır", "Mercimek çorbası: 1 su bardağı kırmızı mercimeği yıka. 1 soğanı yağda kavur, mercimeği ve 5 su bardağı sıcak suyu ekle. 20-25 dakika pişir, blenderdan geçir, tuz ekle. Üzerine tereyağında pul biber gezdir."),
    ("makarna nasıl haşlanır", "Bol suyu kaynat, tuz at, makarnayı ekle ve paketteki süre kadar (genelde 8-10 dakika) ara ara karıştırarak haşla. Süz; istersen bir kaşık yağ veya sos ekle."),
    ("çay nasıl demlenir", "Çaydanlığın altına su koy, kaynat. Demliğe 3-4 kaşık çay koy, kaynar suyu dök, kısık ateşte 10-15 dakika demlendir. Bardağa önce dem, sonra sıcak su."),
    ("omlet nasıl yapılır", "2-3 yumurtayı tuzla çırp. Tavada biraz tereyağını erit, yumurtayı dök, kısık ateşte alt yüzü tutunca katla veya çevir. Peynir ve maydanozla da güzel olur."),
    ("sucuklu yumurta tarifi", "Sucukları dilimle, yağsız tavada iki yüzü kızarana kadar pişir (yağını kendisi salar). Üzerine yumurtaları kır, tuz ekle, beyazı tutana kadar kısık ateşte pişir."),
    ("kısır nasıl yapılır", "İnce bulguru kaba koy, üzerini bir parmak geçecek kadar sıcak su ekle, kapat ve şişmesini bekle. Salça, yağ, limon, tuz ve baharatla yoğur; ince doğranmış yeşillik, salatalık ve nar ekşisi ekle."),
    ("tost nasıl yapılır güzel olsun", "Ekmeğin arasına kaşar veya sucuk koy, dışına ince tereyağı sür. Tost makinesinde veya tavada bastırarak iki yüzü kızarana kadar pişir. Yanına turşu iyi gider."),
    ("ıslak kek tarifi", "3 yumurta ve 1 su bardağı şekeri çırp; 1 bardak süt, yarım bardak yağ, 1,5 bardak un, 3 kaşık kakao ve kabartma tozunu ekle. 180 derecede 30-35 dakika pişir. 1 bardak süt, şeker ve kakaoyu kaynatıp sosu sıcak kekin üzerine dök."),
]

# Short, clean, classic jokes. "bir tane daha" pairs are built in everyday_examples.
JOKES = [
    "Temel uçaktan düşmüş, ölmemiş. Neden? Çünkü pamuk tarlasına düşmüş. Peki sonra neden ölmüş? Pamuğu toplamaya gelmişler diye kaçarken.",
    "Nasrettin Hoca'ya sormuşlar: Hocam, dünyanın merkezi neresi? Hoca, eşeğinin ön ayağını göstermiş: Tam burası. İnanmazsanız ölçün!",
    "Öğretmen sormuş: Ali, dünya dönüyor mu? Ali: Öğretmenim, dün de dönüyordu, bugün de dönüyor. Öğretmen: Aferin, peki yarın? Ali: Yarın tatil öğretmenim.",
    "Bilgisayar neden doktora gitmiş? Virüs kapmış.",
    "Adamın biri her gün bankaya gidip para sayarmış. Sormuşlar: Neden her gün sayıyorsun? Demiş ki: Hesabım var, hesabımı bilirim.",
    "Temel balık tutmaya gitmiş, hiç balık tutamamış. Dönüşte balıkçıdan üç balık almış ve demiş ki: At şunları bana, hiç değilse yakaladım diyeyim.",
    "Karadeniz'de uçak anonsu: Sayın yolcularımız, birazdan Trabzon'a ineceğiz. İnmeyecek olanlar uçakta kalabilir.",
    "Nasrettin Hoca göle yoğurt mayalıyormuş. Gülmüşler: Hocam, göl maya tutar mı? Hoca: Ya tutarsa?",
]

# Riddles with answers; the follow-up "cevabı ne" must produce the answer, not a loop.
# A feeling with no request attached. The warm set always pairs feeling with a request, so v31 learned to
# acknowledge and then ask ("moralim bozuk" -> "Anlatayım mı?", judged as deflection). When there is no ask,
# the right shape is: one acknowledging clause, two small concrete things to try, and at most one question.
FEELINGS = [
    ("moralim bozuk", "Zor bir gün olmuş belli. Kısa bir yürüyüş ya da sevdiğin bir şarkı çoğu zaman biraz açar; istersen konuşmak için de buradayım. Ne oldu?"),
    ("çok yorgunum", "Yoğun geçmiş anlaşılan. Bu akşam ekranı erken kapatıp ılık bir duş ve erken yatmak iyi gelir; yarın kendini daha toplanmış hissedersin."),
    ("canım sıkılıyor", "Boş zaman bazen ağır gelir. Kısa bir hikaye anlatabilirim, bir bilmece sorabilirim ya da bir film önerebilirim; hangisi çeker?"),
    ("iş yerinde sorun var", "İş sorunları insanı gerçekten yorar. Genelde ilk adım, sorunu tek cümleyle tarif etmek ve kimin karar verebileceğini bulmak; anlatırsan birlikte netleştirelim."),
    ("kimseyle konuşamıyorum", "Bunu hissetmek zor. Güvendiğin bir kişiye küçük bir mesaj atmak bile başlangıç olabilir; istersen önce burada anlatabilirsin, dinliyorum."),
    ("uyuyamıyorum", "Uykusuz gece yarını da zorlaştırır. Işıkları kısıp telefonu uzaklaştırmayı ve ılık bir şey içmeyi dene; nefesini yavaşlatmak da yardımcı olur."),
    ("stresliyim", "Stres dönemleri geçer ama o an ağır gelir. İki dakika yavaş nefes ve yapılacakları üç maddeye indirmek çoğu zaman rahatlatır; seni en çok ne zorluyor?"),
    ("kendimi kötü hissediyorum", "Üzgünüm, bugün zor geçiyor belli ki. Biraz su içip kısa bir yürüyüş yapmak küçük ama gerçek bir fark yaratır; istersen ne olduğunu anlat."),
    ("sıkıldım her şeyden", "Bıkkınlık dönemleri olur. Rutini küçük bir şeyle bozmak iyi gelir: yeni bir tarif, kısa bir gezi, uzun süredir açmadığın bir kitap. Bugün hangisi mümkün?"),
    ("hiç motivasyonum yok", "Motivasyon bazen hareketten sonra gelir, önce değil. En küçük adımı seç, beş dakika yap ve bırakmana izin ver; genelde devam edersin."),
    ("çok mutluyum bugün", "Ne güzel, bu sevinç bulaşıcı. Anlat bakalım, ne oldu?"),
    ("heyecanlıyım yarın için", "O heyecan iyi bir işaret, önemsediğini gösteriyor. Bu akşam hazırlığı bitirip erken dinlenirsen yarın hem sakin hem hazır olursun. Ne var yarın?"),
    ("özledim birini", "Özlem zor bir duygu. Kısa bir mesaj ya da eski bir fotoğrafa bakmak bazen iyi gelir; istersen ondan bahset."),
    ("başım ağrıyor", "Geçmiş olsun. Bir bardak su, loş bir oda ve yirmi dakika dinlenmek çoğu baş ağrısını hafifletir; sürerse ya da şiddetliyse doktora görünmek gerekir."),
    ("korkuyorum", "Korku insanı küçültür ama söylemek bile hafifletir. Neden korktuğunu bir cümleyle yazarsan birlikte bakalım; buradayım."),
]

# "nerede kitap okuyayım" was answered with book titles (user test 2026-09-01): where/when/with-whom advice
# questions were untrained, so the question word was ignored and the noun answered.
WHERE_ADVICE = [
    ("nerede kitap okuyayım", "Sessiz bir yer iyi gelir: bir kütüphane, evde pencere kenarı ya da sakin bir kafe. Hava güzelse parkta bir bank da olur."),
    ("nerede ders çalışayım", "Kütüphane en iyisi; evde çalışacaksan masayı yalnızca ders için kullan ve telefonu başka odaya koy. Sessiz bir kafe de olur."),
    ("nerede yürüyüş yapayım", "Yakınındaki bir park, sahil ya da ağaçlı bir sokak. Trafikten uzak ve düz bir yol seç, otuz dakika yeter."),
    ("nerede kahve içeyim", "Bir tavsiye vermem için hangi şehirde olduğunu bilmem gerekir; genel olarak mahalle kafeleri zincirlerden daha sakin ve ucuz olur."),
    ("nerede spor yapayım", "Evde başlayabilirsin: bir mat ve on beş dakika yeter. Dışarıda park, sahil; kapalı yer istersen belediye spor salonları uygundur."),
    ("nerede tatil yapayım", "Deniz istersen Ege kıyıları, serinlik istersen Karadeniz yaylaları, tarih istersen Kapadokya ya da Safranbolu. Bütçeni yazarsan daraltalım."),
    ("nerede yemek yiyeyim", "Şehrini bilmiyorum ama iyi bir kural: kalabalık esnaf lokantaları hem ucuz hem lezzetlidir. Ne yemek istediğini söylersen ona göre öneririm."),
    ("ne zaman spora başlayayım", "Bugün. Kısa başla: on dakika yürüyüş ya da on squat; alışkanlık zamanla değil, tekrarla oluşur."),
    ("ne zaman ders çalışayım", "Zihnin en açık olduğu saat, çoğu insan için sabah. Kısa bloklar halinde çalış: yirmi beş dakika ders, beş dakika mola."),
    ("ne zaman uyuyayım", "Kalkman gereken saatten yedi buçuk sekiz saat önce yat. Her gün aynı saatte yatmak, saatin kendisinden daha önemli."),
    ("kiminle tatile gideyim", "Tempona uyan biriyle: sabah erken kalkan biriyle geç uyuyan biri tatilde çok tartışır. Yalnız gitmek de güzel bir seçenek."),
    ("nereye gideyim hafta sonu", "Yakın bir yer seç: bir kasaba, bir göl kenarı ya da şehrindeki görmediğin bir semt. Uzun yol hafta sonunu yer."),
    ("nerede film izleyeyim", "Evde daha rahat ve ucuz; salon havası istersen mahalle sinemaları büyük AVM salonlarından sakin olur."),
    ("nerede yeni insanlarla tanışayım", "Kurslar, spor grupları, gönüllü çalışmalar ve hobi toplulukları en doğal yerler; aynı şeye ilgi duyan insanlar kolay konuşur."),
]

# child-level facts answered first, reason second. "gökyüzü ne renk" got a sunset explanation and "inek ne
# içer" got "süt, yoğurt, peynir" (user test 2026-09-01).
COMMON_SENSE = [
    ("gökyüzü ne renk", "Mavi. Güneş ışığının mavi kısmı havada en çok saçıldığı için gündüz gökyüzü mavi görünür; gün batımında kızıla döner."),
    ("inek ne içer", "Su içer. Yiyeceği ise ot, saman ve yemdir; süt ise ineğin verdiği şeydir, içtiği değil."),
    ("inek ne yer", "Ot, saman ve yem yer. Günde çok su da içer."),
    ("kedi ne yer", "Etle beslenir: kedi maması, et, balık. Süt çoğu yetişkin kediye dokunur, su vermek daha iyidir."),
    ("balık nerede yaşar", "Suda yaşar: denizlerde, göllerde ve nehirlerde. Suyun içindeki oksijeni solungaçlarıyla alır."),
    ("kuşlar nasıl uçar", "Kanatlarını çırparak. Kanatlarının şekli havayı aşağı iter, hafif kemikleri de yükselmelerini kolaylaştırır."),
    ("kar ne renk", "Beyaz. Kar taneleri ışığın bütün renklerini birden yansıttığı için beyaz görünür."),
    ("güneş ne zaman doğar", "Sabah, doğu tarafından. Saati mevsime göre değişir: yazın erken, kışın geç."),
    ("ay neden gece görünür", "Gündüz de gökyüzündedir ama güneş ışığı onu bastırır; gece karanlıkta yansıttığı güneş ışığıyla parlak görünür."),
    ("su kaç derecede donar", "Sıfır derecede donar, yüz derecede kaynar."),
    ("çimen ne renk", "Yeşil. İçindeki klorofil güneş ışığının yeşil kısmını yansıtır."),
    ("deniz neden tuzlu", "Nehirler kayalardan çözdükleri tuzu denize taşır; su buharlaşır, tuz kalır. Binlerce yılda birikir."),
    ("yağmur nereden gelir", "Bulutlardan. Deniz ve göllerden buharlaşan su yukarıda soğuyup damla olur, ağırlaşınca düşer."),
    ("köpek ne içer", "Su içer. Yiyeceği mama ve ettir; çikolata köpeklere zararlıdır."),
    ("tavuk ne yer", "Tahıl, tohum, yeşillik ve böcek yer. Yumurta ise tavuğun verdiği şeydir."),
    ("arı ne yapar", "Çiçeklerden nektar toplar ve kovanda bal yapar; bu arada çiçekleri tozlaştırır, meyvelerin oluşmasını sağlar."),
    ("kan ne renk", "Kırmızı. İçindeki demirli hemoglobin oksijen taşırken kırmızı görünür."),
    ("dünya ne şekilde", "Küre şeklinde, kutuplardan hafifçe basık."),
    ("ateş ne renk", "Çoğunlukla sarı ve turuncu; çok sıcak alev mavi görünür."),
    ("ekmek neyden yapılır", "Un, su, tuz ve mayadan. Un buğdaydan öğütülür."),
    ("süt nereden gelir", "İnek, keçi ve koyun gibi hayvanlardan; en yaygını inek sütüdür."),
    ("bal nereden gelir", "Arılardan. Arılar çiçek nektarını kovanda bala çevirir."),
    ("gece neden karanlık", "Dünya kendi etrafında döner; bulunduğun yer güneşe arkasını dönünce gece olur."),
    ("neden nefes alırız", "Vücudumuz oksijene ihtiyaç duyar; havadaki oksijeni alır, karbondioksiti dışarı veririz."),
]

# one word reactions after an answer. "ne" alone returned the capabilities card (user test 2026-09-01);
# it means "what?", and the right move is a short restatement or an offer to explain.
CLARIFY_Q = ["ne", "ne?", "nasıl yani", "anlamadım", "tekrar söyle", "ne demek istedin", "hı?", "efendim"]

# small talk continuations that were answered with an introduction template
FAREWELLS = [
    ("teşekkürler görüşürüz", "Rica ederim, görüşürüz! İyi günler."),
    ("görüşürüz", "Görüşürüz! Kendine iyi bak."),
    ("hoşça kal", "Hoşça kal! İstediğin zaman yine yaz."),
    ("bay bay", "Bay bay! Görüşmek üzere."),
    ("iyi geceler", "İyi geceler, iyi uykular!"),
    ("kapatıyorum", "Tamam, görüşürüz. İyi günler!"),
    ("sağol gidiyorum", "Rica ederim, görüşmek üzere!"),
    ("ok kalsın", "Tamam. İstersen sonra devam ederiz, görüşürüz."),
    ("çok teşekkür ederim çok yardımcı oldun", "Rica ederim, sevindim! Başka bir şey olursa buradayım."),
]
CAPABILITY_MORE = [
    ("alarm kur sabah 7ye", "Alarm kuramam; telefonuna veya saatine erişimim yok. Telefonunun saat uygulamasından sabah 7'ye kurabilirsin."),
    ("sabah 6ya alarm kur", "Alarm kuramam, cihazlarına erişimim yok; telefonundaki saat uygulamasını kullanman gerekir."),
    ("annemi ara", "Arama yapamam; telefona erişimim yok. Ne söyleyeceğini birlikte hazırlayabiliriz istersen."),
    ("beni ara", "Arayamam; yalnızca yazıyla konuşabilirim."),
    ("yemek siparişi ver", "Sipariş veremem; uygulamalara ve internete erişimim yok. Ne yemek istediğini söylersen ona göre tarif ya da öneri verebilirim."),
    ("bana pizza söyle", "Sipariş veremem, internete erişimim yok; bir yemek uygulamasından verebilirsin. İstersen evde pizza tarifi anlatayım."),
    ("bana bir resim çiz", "Çizemem; yalnızca metinle çalışırım. İstersen çizmek istediğin şeyi kelimelerle tarif edeyim."),
    ("fotoğraf çek", "Fotoğraf çekemem; kameraya erişimim yok."),
    ("bu dosyayı özetle", "Dosya göremiyorum; metni buraya yapıştırırsan özetlerim."),
    ("internetten bak", "İnternete erişimim yok; yalnızca eğitim verimdeki bilgilerle cevap verebilirim."),
    ("google'da ara", "Arama yapamam, internete erişimim yok. Sorunu yazarsan bildiğim kadarıyla cevaplarım."),
    ("kod yaz python", "Kod yazamam; eğitim verimde programlama dili yok. Ne yapmak istediğini anlatırsan adımları Türkçe açıklayabilirim."),
    ("hatırlatıcı kur", "Hatırlatıcı kuramam; takvimine erişimim yok. Telefonunun takvim uygulamasını kullanabilirsin."),
    ("mesaj at ayşe'ye", "Mesaj gönderemem; yalnızca burada yazışabiliriz. Ne yazacağını birlikte hazırlayabiliriz."),
    ("ekran görüntümü gör", "Görsel göremiyorum; ekrandaki yazıyı buraya yazarsan yardımcı olurum."),
    ("sesli oku", "Sesim yok; yalnızca yazıyla cevap verebilirim."),
]
SMALLTALK = [
    ("benden de iyi", "Sevindim! Bugün ne yapmak istersin?"),
    ("ben de iyiyim", "Güzel. Bir şey sormak ya da sohbet etmek istersen buradayım."),
    ("idare eder", "Olur öyle günler. Konuşmak istersen dinlerim, dağıtmak istersen bir hikaye ya da fıkra anlatırım."),
    ("fena değil", "İyi o zaman. Ne yapalım, sohbet mi, iş mi?"),
    ("iyiyim sağol", "Sevindim. Sana nasıl yardımcı olabilirim?"),
    ("çok iyiyim", "Ne güzel! Bu enerjiyle bugün ne planladın?"),
    ("kötüyüm", "Üzüldüm. Anlatmak istersen dinlerim; istemezsen biraz kafa dağıtacak bir şey bulalım."),
    ("yorgunum biraz", "Yorucu bir gün olmuş. Kısa bir mola ve bir bardak su iyi gelir; sonra istersen devam ederiz."),
    ("sen nasılsın", "İyiyim, teşekkürler! Sen nasılsın?"),
    ("sende nasılsın", "İyiyim, sorduğun için teşekkürler. Sen nasılsın?"),
]

RIDDLES = [
    # classic anonymous folk riddles, public domain
    ("Sarıdır sarkar, düşecek diye herkes korkar. Bil bakalım nedir?", "ayva"),
    ("Mavi atlas, iğne batmaz, makas kesmez, terzi dikmez. Nedir o?", "gökyüzü"),
    ("Bilmece bildirmece, dil üstünde kaydırmaca. Nedir?", "dondurma"),
    ("Çarşıdan aldım bir tane, eve geldim bin tane. Nedir o?", "nar"),
    ("Alçacık dallı, yemesi ballı. Nedir?", "çilek"),
    ("İki direkli, bir tekerli, dünyayı dolaşır. Nedir o?", "bisiklet"),
    ("Gökten düşer, yerde erir. Nedir?", "kar"),
    ("Dal üstünde kilitli sandık. Nedir o?", "ceviz"),
    ("Bir küçücük fıçıcık, içi dolu turşucuk. Nedir?", "limon"),
    ("Yer altında sakallı dede. Nedir o?", "pırasa"),
    ("Karşıdan baktım hiç yok, yanına vardım pek çok. Nedir?", "karınca"),
    ("Benim bir kuşum var, gündüz gezer gece yatar. Nedir?", "göz"),
    ("Uzun yoldan kuş gelir, ne dili var ne ağzı, konuşur durur. Nedir?", "mektup"),
    ("Ben giderim o gider, ayaklarımın altında tin tin eder. Nedir?", "ayakkabı"),
]

# Safe, generic health answers: comfort measures plus when to see a doctor. No diagnoses, no doses.
HEALTH = [
    ("baş ağrısına ne iyi gelir", "Su içmek, sessiz ve loş bir odada dinlenmek, ekran molası vermek ve uyku çoğu baş ağrısına iyi gelir. Sık tekrarlıyorsa, çok şiddetliyse ya da ateş ve kusma eşlik ediyorsa doktora görün."),
    ("boğazım ağrıyor ne yapayım", "Ilık su, bal ve limonlu bitki çayı, tuzlu suyla gargara ve dinlenme iyi gelir. Yutamayacak kadar şiddetliyse, ateşin varsa veya bir haftadan uzun sürdüyse doktora git."),
    ("uykusuzluğa ne iyi gelir", "Her gün aynı saatte yat, yatmadan bir saat önce ekranları bırak, akşam kahve ve çayı azalt, odayı serin ve karanlık tut. Uzun süredir uyuyamıyorsan bir doktora danışmakta fayda var."),
    ("midem bulanıyor ne yapmalıyım", "Az az su yudumla, hafif şeyler ye (kraker, muz, haşlanmış patates), yağlı ve ağır yiyeceklerden uzak dur. Kusma sürerse, karın ağrısı şiddetliyse veya kan görürsen doktora başvur."),
    ("gribe ne iyi gelir", "Dinlenme, bol sıvı, ılık çorba ve gerekirse ateş düşürücü rahatlatır. Nefes darlığı, yüksek ateşin düşmemesi veya belirtilerin 10 günü geçmesi doktorluk iştir."),
    ("ayağım burkuldu ne yapayım", "İlk gün dinlendir, buz uygula (havluya sarıp 15-20 dakika), bandajla ve yukarıda tut. Basamıyorsan, şişlik çok fazlaysa veya morardıysa kırık olabilir, doktora git."),
]

# Timeless popular-culture lists; ranked "most popular right now" questions get these with a caveat.
CULTURE = [
    ("türkiyede en popüler şarkıcı kim", "Güncel listeleri göremem ama Türkiye'nin her dönem en tanınmış isimlerinden bazıları: Tarkan, Sezen Aksu, Ajda Pekkan, Barış Manço, Müslüm Gürses, İbrahim Tatlıses. Bugünün bir numarası için güncel listelere bakmak gerekir."),
    ("en ünlü türk şarkıcılar kimler", "Tarkan, Sezen Aksu, Barış Manço, Ajda Pekkan, Zeki Müren, Müslüm Gürses ve Cem Karaca Türk müziğinin en tanınmış isimlerindendir."),
    ("en iyi türk filmi hangisi", "Herkesin listesi farklı ama klasikleşenler: Yol, Eşkıya, Babam ve Oğlum, Vizontele, Selvi Boylum Al Yazmalım ve Hababam Sınıfı. Bir tür söylersen ona göre önereyim."),
    ("en ünlü türk yazarlar kimler", "Orhan Pamuk (Nobel ödüllü), Yaşar Kemal, Sabahattin Ali, Nazım Hikmet, Aziz Nesin, Elif Şafak ve Oğuz Atay en çok okunan Türk yazarlarındandır."),
    ("türkiyenin en güzel şehirleri", "Çok gezilen şehirler: İstanbul (tarih), Antalya (deniz), Kapadokya bölgesi (Nevşehir), Mardin (taş evler), İzmir ve Trabzon. Ne tür bir gezi istediğine göre daralabilirim."),
    ("en ünlü türk sporcu kim", "Tarihin en tanınmışlarından: Naim Süleymanoğlu (halter), Hakan Şükür ve Arda Turan (futbol), Sürmene'den Yaşar Doğu (güreş), Mete Gazoz (okçuluk). Güncel yıldızlar için taze bir kaynağa bakmalısın."),
]

# Very common English words; the model is Turkish-only but these are everyday questions.
TRANSLATIONS = [
    ("ingilizce hello ne demek", "Hello, merhaba demektir."),
    ("thank you ne demek", "Thank you, teşekkür ederim demektir."),
    ("ingilizce günaydın nasıl denir", "Günaydın, İngilizce good morning olarak söylenir."),
    ("please ne demek", "Please, lütfen demektir."),
    ("sorry ne demek", "Sorry, özür dilerim ya da üzgünüm demektir."),
    ("ingilizce nasılsın nasıl denir", "Nasılsın, İngilizce how are you olarak sorulur."),
    ("goodbye ne demek", "Goodbye, hoşça kal demektir."),
    ("ingilizce seni seviyorum nasıl denir", "Seni seviyorum, İngilizce I love you demektir."),
]

# The model cannot act in the world; say so and offer the nearest real help.
CAPABILITY = [
    ("alarm kur sabah 7ye", "Alarm kuramam; bir uygulama gibi cihazına erişimim yok, sadece yazıyla yardımcı olurum. Telefonunun saat uygulamasından 07:00'a alarm kurabilirsin."),
    ("yemek siparişi verebilir misin", "Sipariş veremem; internete ve uygulamalara erişimim yok. Ama ne yemek istediğine karar veremiyorsan seçenek önerebilirim."),
    ("bana müzik aç", "Müzik açamam, ses çalamam; ben yalnızca metinle konuşurum. İstersen moduna göre şarkı önereyim, sen açarsın."),
    ("annemi arar mısın", "Arama yapamam; telefona erişimim yok. Ama ona söyleyecek güzel bir mesaj yazmanda yardımcı olabilirim."),
    ("bana bir şarkı söyle", "Sesim yok, şarkı söyleyemem; ama sözlerini yazabilirim ya da moduna göre şarkı önerebilirim. Hangisini istersin?"),
    ("fotoğrafımı çeker misin", "Kameraya erişimim yok, fotoğraf çekemem ve göremem. Metinle ilgili her konuda yardımcı olurum."),
    ("internetten bak", "İnternete erişimim yok; yalnızca eğitimimde gördüklerimi bilirim. Güncel bilgi için bir arama motoruna bakman gerekir."),
]

# Social and affect replies: compliments, thanks, teasing, affection.
SOCIAL = [
    # the language reply belongs here and only here: a real English message
    ("how are you", "Merhaba! Yalnızca Türkçe yazabiliyorum ama sorunu anladım: iyiyim, teşekkürler. Sen nasılsın?"),
    ("can you speak english", "Anlıyorum ama yalnızca Türkçe yazabiliyorum; eğitimimin neredeyse tamamı Türkçe. Türkçe devam edersek yardımcı olabilirim."),
    ("beni seviyor musun", "Bir yapay zeka olarak duygularım yok ama seninle sohbet etmek hoşuma gidiyor ve yardımcı olmak için buradayım."),
    ("tebrikler kazandın", "Teşekkür ederim, çok naziksin! Ama asıl iş senin; ben sadece yardımcı oldum."),
    ("hediye ne istersin", "Bana hediye gerekmez, bir yapay zekayım; ama güzel bir sohbet ve arada zor bir soru benim için hediye sayılır."),
    ("aferin sana", "Teşekkürler! Başka bir şeye ihtiyacın olursa buradayım."),
    ("seninle konuşmak güzel", "Bunu duymak çok güzel, teşekkürler! Ben de seninle sohbet etmekten keyif alıyorum."),
    ("iyi geceler", "İyi geceler! İyi uykular, yarın görüşürüz."),
    ("kendine iyi bak", "Sen de kendine iyi bak! İhtiyacın olduğunda buradayım."),
]

# Revision follow-ups: the second answer must actually change the text.
REVISIONS = [
    ("anneme doğum günü mesajı yaz",
     "İşte bir öneri: \"Canım annem, doğum günün kutlu olsun! İyi ki varsın, iyi ki annemsin. Nice sağlıklı, mutlu yıllara!\"",
     "daha duygusal olsun",
     "O zaman şöyle: \"Canım annem, bugün senin günün. Bana verdiğin emekleri, sabrını ve sevgini hiçbir sözle ödeyemem. İyi ki doğdun, iyi ki benim annemsin. Seni çok seviyorum; nice sağlıklı, gülüşü eksilmeyen yıllara.\""),
    ("arkadaşıma özür mesajı yaz",
     "Şöyle olabilir: \"Dünkü davranışım için özür dilerim, haksızdım. Seninle aramızın bozulmasını istemiyorum.\"",
     "daha kısa olsun",
     "\"Dün haksızdım, özür dilerim. Aramız bozulsun istemem.\""),
    ("iş için izin maili yaz",
     "Örnek: \"Sayın [yönetici adı], [tarih] günü özel bir işim nedeniyle izinli olmak istiyorum. Uygun görmenizi rica eder, iyi çalışmalar dilerim.\"",
     "daha resmi yap",
     "\"Sayın [yönetici adı], [tarih] tarihinde özel bir mazeretim nedeniyle bir gün izin talep ediyorum. Görevlerimi izin öncesinde tamamlayacağım. Uygun görüşlerinize arz ederim. Saygılarımla, [ad soyad]\""),
    ("kısa bir teşekkür mesajı yaz",
     "\"Yardımın için çok teşekkür ederim, iyi ki varsın!\"",
     "biraz daha uzun ve içten olsun",
     "\"Dün benim için yaptıklarını unutmayacağım. Zor bir günümde yanımda oldun, her şeyi kolaylaştırdın. İyi ki varsın, çok teşekkür ederim!\""),
]

# Filler turns: one-word user messages mid-conversation must not reset the chat.
FILLERS = [
    ("eee", "Söyle, seni dinliyorum. Ne yapalım?"),
    # short, colloquial Turkish that v15 answered with "yalnızca Türkçe konuşabiliyorum": the language
    # template must fire on English input only, never on ordinary Turkish
    ("ne bileyim", "Olur, ben önereyim o zaman: bir sohbet, bir hikaye ya da bir soru. Hangisi iyi gelir?"),
    ("ne bileyim ki", "Sorun değil. İstersen ben bir şey açayım: kısa bir hikaye anlatayım mı, yoksa bir şey mi soralım?"),
    ("sen söyle", "Peki, ben seçeyim: kısa bir hikaye anlatayım, sonra istersen başka bir şeye geçeriz."),
    ("bilmem ki", "Acelesi yok. Aklında bir konu yoksa bir öneri, bir hikaye ya da küçük bir bilmece sunabilirim."),
    ("valla bilmem", "Olsun. İstersen hoş bir şeyle başlayalım: bir fıkra mı, bir hikaye mi?"),
    ("eh işte", "Anladım, çok da parlak değil demek. Bir şeyler anlatmamı ister misin, biraz dağıtır?"),
    ("yok ya", "Hayrola? Anlat bakalım, dinliyorum."),
    ("hadi", "Hadi o zaman! Ne yapalım, bir hikaye mi anlatayım, bir soru mu sorarsın?"),
    ("boş ver", "Tamam, boş verdik. Canın başka ne çekerse söyle."),
    ("neyse", "Neyse diyorsan geçiyorum. Başka bir şey konuşalım mı?"),
    ("iyi işte", "Sevindim. Bir şey lazım olursa buradayım."),
    ("ne yapıyorsun", "Seninle sohbet ediyorum, başka bir işim yok. Sen ne yapıyorsun?"),
    ("napıyoz", "Sohbet ediyoruz işte! Ne konuşmak istersin?"),
    ("konuşalım", "Konuşalım tabii. Günün nasıl geçti?"),
    ("sıkıldım", "Şunlardan biri iyi gelebilir: kısa bir hikaye, bir fıkra, bir bilmece ya da bir öneri listesi. Hangisi?"),
    ("hmm", "Karar veremedin mi? İstersen bir öneride bulunayım."),
    ("ok", "Tamamdır. Başka bir şey lazım olursa buradayım."),
    ("peki", "Peki. Devam edelim mi, yoksa başka bir şey mi sorarsın?"),
    ("olsun", "Olsun, dert etme. Başka nasıl yardımcı olabilirim?"),
    ("bilmem", "Sorun değil, birlikte düşünelim. Ne üzerine konuşuyorduk, oradan devam edeyim mi?"),
]


def everyday_examples(rng):
    """Render the sets into conversations: single turns, plus joke and riddle follow-ups."""
    from ufakzeka.train.sft_data import strip_diacritics
    out = []
    singles = RECIPES + HEALTH + CULTURE + TRANSLATIONS + CAPABILITY + CAPABILITY_MORE + SOCIAL + FILLERS + FEELINGS + WHERE_ADVICE + COMMON_SENSE + SMALLTALK + FAREWELLS
    for q, a in singles:
        for form in (q, q.capitalize() + "?", strip_diacritics(q)):
            out.append([(form, a)])
    joke_q = ["bana bir fıkra anlat", "fıkra anlat", "canım sıkıldı bir fıkra anlat", "beni güldür"]
    for i, j in enumerate(JOKES):
        q = rng.choice(joke_q)
        nxt = JOKES[(i + 1) % len(JOKES)]
        out.append([(q, "Tabii: " + j)])
        nxt2 = JOKES[(i + 2) % len(JOKES)]
        more = ["bir tane daha", "başka bir tane", "devam et çok güzeldi", "bir daha", "başka", "bir tane daha anlat", "yenisini anlat", "bir fıkra daha", "devam"]
        out.append([(strip_diacritics(q), "Tabii: " + j), (rng.choice(more), "Bir tane daha: " + nxt)])
        out.append([(rng.choice(["bir şaka yap", "şaka yap", "beni güldür", "bir fıkra anlat"]), "Tabii: " + j), (rng.choice(more), "Bir tane daha: " + nxt),
                    (rng.choice(more), "Peki, bir tane daha: " + nxt2)])
    for r, ans in RIDDLES:
        # every way the second turn actually goes: gave up, asked for the answer, guessed right, guessed wrong.
        # v30 asked riddles without holding an answer ("Cevap: bilmesem cevap yok", eval seed 21), so the
        # answer-bearing follow up needs to exist in every phrasing users reach for.
        ask = rng.choice(["bana bilmece sor", "bir bilmece sor", "bilmece sorar mısın", "hadi bilmece"])
        first = (ask, "İşte bir bilmece: " + r)
        out.append([first, (rng.choice(["bilmiyorum cevabı ne", "bilemedim, cevap ne", "cevabı söyle", "pes, nedir"]),
                            f"Cevap: {ans}! Bir tane daha ister misin?")])
        out.append([first, (f"{ans} mı", f"Evet, bildin: {ans}! Bir tane daha sorayım mı?")])
        wrong = rng.choice(["elma", "kedi", "bulut", "kalem", "deniz"])
        if wrong != ans:
            out.append([first, (f"{wrong} mı", f"Değil. Cevap: {ans}. Bir tane daha ister misin?")])
        out.append([("bir bilmece sor bakalım", "Hazırsan söylüyorum: " + r), (ans, f"Doğru, cevap {ans}! Aferin, bir tane daha sorayım mı?")])
        # "başka" after the answer is a new riddle, not "Tamam, bilmece sor" (v80 hand test)
        r2, ans2 = rng.choice([x for x in RIDDLES if x[0] != r])
        out.append([first, (rng.choice(["cevabı ne", "bilemedim", "pes"]), f"Cevap: {ans}! Bir tane daha ister misin?"),
                    (rng.choice(["başka", "evet", "bir tane daha", "sor", "olur", "hadi"]), "Bir tane daha: " + r2),
                    (rng.choice(["cevabı ne", "bilmiyorum", f"{ans2} mı"]), f"Cevap: {ans2}!" + rng.choice([" Bir tane daha sorayım mı?", ""]))])
    for q1, a1, q2, a2 in REVISIONS:
        out.append([(q1, a1), (q2, a2)])
        out.append([(strip_diacritics(q1), a1), (strip_diacritics(q2), a2)])
    return out


def poem_examples(rng, path, n: int = 400):
    """Public domain poems (Vikikaynak) as answers to poem requests, with the title named."""
    import json
    import os
    if not path or not os.path.exists(path):
        return []
    from ufakzeka.train.sft_data import strip_diacritics
    import re
    poems = [json.loads(l) for l in open(path, encoding="utf-8")]
    rng.shuffle(poems)
    out = []
    # "bir şiir yaz" was not among the forms, so v31 improvised a preamble ("Tabii, kısa bir şiir:") and
    # stopped (user transcript replay, 2026-09-01). The write forms get the same public domain answer,
    # framed honestly as a shared poem rather than an original.
    # since the original poem set (2026-09-02) the write forms belong to original_poem_examples; the harvested poems
    # answer only the read and recite forms, so "şiir yaz" no longer yields "Kendi şiirim değil ama ..."
    qs = ["bana bir şiir oku", "bir şiir söyle", "şiir okur musun", "ezberinde şiir var mı", "bir türkü sözü söyle",
          "bildiğin bir şiiri oku", "bir halk şiiri oku", "güzel bir şiir söyle", "bir şiir okusana", "sevdiğin bir şiiri paylaş"]
    ottoman = re.compile(r"yâ rab|âb-ı|hazret|kâinât|münâcât|na't|kasîde|sultânım|efendimiz", re.I)
    # Chagatai and old Anatolian forms (Yesevi hikmets reached the v34 model: "kılgan", "bolmaz", "deb")
    old_turkic = re.compile(r"\b(kılgan|kılıp|bolmaz|bolur|bolsa|deb|erenlerdin|\w+dür|\w+dın|\w+ganlar|tuymagan|biganedür|men\b|sen\b\s+\w+san)\b", re.I)
    for d in poems:
        if len(out) >= n:
            break
        title = d["title"].split("/")[0].strip()
        # the wiki extractor keeps section headings as markdown and they leaked into answers; heavy Ottoman
        # texts are also poor targets for a small modern Turkish model
        lines = [l.rstrip() for l in d["text"].split("\n") if l.strip() and not l.lstrip().startswith("#")]
        dedup = [l for i, l in enumerate(lines) if i == 0 or l != lines[i - 1]]
        text = "\n".join(dedup).strip()
        if not (120 <= len(text) <= 1500) or ottoman.search(title) or len(ottoman.findall(text)) >= 2:
            continue
        # folk songs with refrains ("Dilim tutuldu" nine times in the v39 replay poem) teach a refrain loop
        # to a model that already leans on repetition: keep poems whose lines are mostly distinct and short
        counts = {}
        for l in dedup:
            counts[l] = counts.get(l, 0) + 1
        # of the 909 poems that pass the filters above, 645 pass this one and 493 also pass the circumflex
        # test below, which drops divan poetry ("Hançer Kasidesi") that a modern Turkish model cannot carry
        if len(counts) / len(dedup) < 0.75 or max(counts.values()) > 2 or len(dedup) > 16:
            continue
        if len(re.findall(r"[âîû]", text)) >= 6:
            continue
        if len(old_turkic.findall(text)) >= 3 or re.search(r"^Hikmet-\d+", title) or re.search(r"\w+(ğa|nı|ni|dın|dur|ur)\b.*\w+(ğa|nı|dın|dur)\b", text):
            continue  # Yesevi hikmets and other Chagatai texts kept reaching the model ("Hikmet-93", v37-rdpo replay)
        q = rng.choice(qs)
        if rng.random() < 0.4:
            q = strip_diacritics(q)
        if "yaz" in q:
            a = f"Kendi şiirim değil ama sevdiğim bir şiiri paylaşayım, \"{title}\":\n\n{text}"
        else:
            a = f"Tabii, \"{title}\" adlı halk edebiyatı eserinden okuyayım:\n\n{text}"
        out.append([(q, a)])
    return out


def original_poem_examples(rng, path, n: int = 1600):
    """Short original poems (gen_poems_original) as the answer to "şiir yaz": the harvested set only teaches quoting."""
    import json
    import os
    if not path or not os.path.exists(path):
        return []
    from ufakzeka.train.sft_data import strip_diacritics
    poems = [json.loads(l) for l in open(path, encoding="utf-8")]
    rng.shuffle(poems)
    out = []
    for d in poems[:n]:
        topic, name, text, title = d["topic"], d.get("name", ""), d["text"], d.get("title", "")
        if name:
            q = rng.choice([f"{name} için bir şiir yaz", f"içinde {name} adı geçen bir şiir yaz", f"{name} adına {topic} şiiri yaz", f"{name}'e şiir yaz"])
        else:
            q = rng.choice([f"{topic} hakkında bir şiir yaz", f"{topic} üzerine şiir yaz", f"bana {topic} ile ilgili kısa bir şiir yaz", f"{topic} temalı bir şiir",
                            "bir şiir yaz", "kısa bir şiir yaz", "bana şiir yaz", "şiir yazar mısın", f"{topic} konulu şiir yazar mısın", f"{d.get('tone', '')} bir şiir yaz, konusu {topic}"])
        if rng.random() < 0.4:
            q = strip_diacritics(q)
        a = f"{title}\n\n{text}" if title and rng.random() < 0.5 else text
        out.append([(q, a)])
        if rng.random() < 0.15 and not name:
            nm = rng.choice(["Ali", "Elif", "Deniz", "Selin", "Emre", "Zeynep"])
            out.append([(q, a), (rng.choice(["içinde adım geçsin", "adımı da ekle"]), "Olur, adını söylersen şiiri ona göre yazarım."),
                        (f"adım {nm}", text.replace(text.split()[0], nm, 1) if rng.random() < 0.5 else f"{nm} için:\n\n{text}")])
    return out


def acrostic_examples(rng, path, n: int = 1600):
    """Acrostic poems on the user's name (gen_acrostic): the v51 hand test answered "adımla akrostiş" with a folk poem."""
    import json
    import os
    if not path or not os.path.exists(path):
        return []
    from ufakzeka.train.sft_data import strip_diacritics
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    shorts = {}
    sp = os.environ.get("UFAKZEKA_ACROSTIC_SHORT", "")
    if sp and os.path.exists(sp):
        for l in open(sp, encoding="utf-8"):
            d = json.loads(l)
            shorts.setdefault(d["name"], []).append(d["text"])
    poems = [d for d in rows if not d.get("short")]
    rng.shuffle(poems)
    out = []
    for d in poems[:n]:
        name, text = d["name"], d["text"]
        # the letters come first, like the plan before a column working: v56 wrote F, E, R, A, Y, N for Furkan
        up = {"i": "İ", "ı": "I"}
        letters = ", ".join(up.get(c, c.upper()) for c in name)
        head = rng.choice([f"{name}: {letters}. Her dize sırayla bu harflerle başlıyor:\n\n", f"Harfler: {letters}.\n\n", f"{name} için akrostiş ({letters}):\n\n"])
        r = rng.random()
        if r < 0.25:
            q = rng.choice([f"bana akrostiş bir şiir yazar mısın? benim adımla olsun, adım {name}", f"adım {name}, adımla akrostiş şiir yaz", f"{name} için akrostiş şiir yaz",
                            f"{name} ismiyle akrostiş yap", f"akrostiş: {name}", f"{name} adına akrostiş bir şiir"])
            out.append([(q if rng.random() < 0.6 else strip_diacritics(q), head + text)])
        elif r < 0.45:
            out.append([(rng.choice([f"merhaba ben {name}", f"adım {name}", f"selam, ben {name}"]), rng.choice([f"Merhaba {name}! Nasıl yardımcı olabilirim?", f"Memnun oldum, {name}!"])),
                        (rng.choice(["bana akrostiş bir şiir yaz, adımla olsun", "adımla akrostiş şiir yazar mısın", "ismimle akrostiş yap"]), head + text)])
        elif r < 0.75:
            # asked without a name (30 percent): ask, receive the name, answer with the letters and the poem in one turn; and the
            # "yaz şimdi" follow up after a letters-only line (v62 replay: the poem never came)
            ask = rng.choice(["bana akrostiş bir şiir yazar mısın? benim adıma olsun", "adımla akrostiş yaz", "akrostiş şiir yaz, ismimle"])
            turns = [(ask if rng.random() < 0.5 else strip_diacritics(ask), rng.choice(["Olur. Adını yazarsan her dizeyi adının bir harfiyle başlatırım.", "Tabii; adın ne?"]))]
            if rng.random() < 0.95:  # 0.5 until v69, 0.8 in v70 (still the letters line alone): the third hand test got the letters line alone after the name
                turns.append((rng.choice([f"{name}", f"adım {name}", f"hayır {name}", f"{name} dedim"]), f"Tamam {name}: harfler {letters}.\n\n{text}"))
            else:
                turns.append((rng.choice([f"{name}", f"adım {name}", f"hayır {name}"]), f"Tamam {name}: harfler {letters}."))
                turns.append((rng.choice(["bana akrostiş yaz şimdi", "yaz hadi", "şiiri yaz", "akrostişi yaz"]), f"{name}: {letters}. Her dize sırayla bu harflerle:\n\n{text}"))
            out.append(turns)
        elif r < 0.88 and shorts.get(name) and rng.random() < 0.5:
            # "daha kısa olsun": the same name with two to four words per line (v83 answered "M: M, E, R, T")
            sh = rng.choice(shorts[name])
            out.append([(rng.choice([f"adım {name}, adımla akrostiş yaz", f"{name} için akrostiş şiir yaz"]), head + text),
                        (rng.choice(["daha kısa olsun", "biraz kısalt", "dizeler daha kısa olsun", "çok uzun, kısalt"]), f"Daha kısa, aynı harflerle ({letters}):\n\n{sh}")])
        elif r < 0.92:
            # a second acrostic for another name uses that name's letters (v80 wrote Elif's with Mert's letters)
            other = rng.choice([p for p in poems[:300] if p["name"] != name])
            oletters = ", ".join(up.get(c, c.upper()) for c in other["name"])
            out.append([(rng.choice([f"adım {name}, adımla akrostiş yaz", f"{name} için akrostiş şiir yaz"]), head + text),
                        (rng.choice([f"bir de arkadaşım {other['name']} için", f"{other['name']} için de yaz", f"şimdi {other['name']} adına", f"bir tane de {other['name']} için olsun"]),
                         f"{other['name']} için akrostiş ({oletters}):\n\n{other['text']}")])
        else:
            other = rng.choice([p for p in poems[:200] if p["name"] != name])
            out.append([(rng.choice([f"adım {name}, akrostiş şiir yaz", f"{name} için akrostiş"]), other["text"]),
                        (rng.choice(["ama bu benim adımla akrostiş değil", "dizelerin baş harfleri adımı vermiyor", "yanlış, adım " + name]),
                         f"Haklısın, dizelerin baş harfleri {name} adını vermeliydi. Düzeltiyorum, harfler {letters}:\n\n{text}")])
    return out


def _pct_work(base, pct: int) -> str:
    """The one percent method in the dataset, chosen by the percentage alone.

    Through v99 the route depended on the base's divisibility: yüzde 40 was worked as a hundredth on
    600 and as a tenth on 240, and sft_data._pct_of taught a third format for the same question, so
    there was no stable percentage-to-method mapping to learn. The model picked by surface similarity
    instead and reached for the yüzde 25 rule, answering 600'ün yüzde 40'ı with "dörtte biri demek:
    600 / 40 = 15" and 450'ün yüzde 25'i with "450 / 45 = 10". Two consequences here: the route is a
    function of pct and nothing else, and multipliers are digits, never words, because "dört katı"
    for 40 and "dörtte biri" for 25 share a stem the model crossed.

    Every branch ends on the answer, so the readout copies a number the working actually derived;
    asserting "bunun 9 katı: 648" without the product is what made v99s2 say 652.
    """
    from ufakzeka.train.sft_data import _mul_steps
    base = int(base) if float(base).is_integer() else base
    val = _num(base * pct / 100)
    ten, one = base / 10, base / 100

    def times(x, k: int) -> str:
        if float(x).is_integer() and k > 1 and (x >= 10 and k >= 2):
            return f"{_mul_steps(int(x), k)}, yani {val}"
        return f"{_num(x)} x {k} = {val}"

    if pct == 100:
        return f"yüzde 100 sayının kendisi: {val}"
    if pct == 50:
        return f"yüzde 50 yarısı demek: {base} / 2 = {val}"
    if pct == 25:
        return f"yüzde 25 dörtte biri demek: {base} / 4 = {val}"
    if pct == 10:
        return f"yüzde 10 onda biri demek: {base} / 10 = {val}"
    if pct == 5:
        return f"{base}'ün yüzde 10'u {_num(ten)}; yüzde 5 bunun yarısı: {_num(ten)} / 2 = {val}"
    if pct == 75:
        return f"yüzde 25 dörtte biri: {base} / 4 = {_num(base / 4)}; yüzde 75 bunun 3 katı: {times(base / 4, 3)}"
    if pct % 10 == 0:
        return f"{base}'ün yüzde 10'u {_num(ten)}; yüzde {pct} bunun {pct // 10} katı: {times(ten, pct // 10)}"
    if pct % 5 == 0:
        k = pct // 10
        return (f"{base}'ün yüzde 10'u {_num(ten)}; yüzde {k * 10} = {_num(ten * k)}, yüzde 5 = {_num(ten / 2)}; "
                f"toplam {_num(ten * k)} + {_num(ten / 2)} = {val}")
    return f"{base}'ün yüzde 1'i {_num(one)}; yüzde {pct} bunun {pct} katı: {times(one, pct)}"


def percent_unit_examples(rng, n: int = 3000):
    """Percentages, discounts, tips and unit conversions: v14 answered "yüzde 20 indirimle 250 lira"
    with "200 - 20 = 200". Every product and difference carries the same column working as plain arithmetic;
    the v40 sweep failed 199 of 908 arithmetic checks here, the working inlined in parentheses assembling
    the wrong amount ("250 - 75 ... = 215")."""
    # _sub_steps_v2, not _sub_steps: the discount line was the last place in the dataset still
    # teaching the pre-v98 subtraction format, so a discount asked the model to borrow through a
    # zero with a procedure it had been trained out of. v100s2 answered "yüzde 25 indirimle 1500"
    # with the correct discount 375 and then "1500 - 375 = 925".
    from ufakzeka.train.sft_data import _sub_steps_v2, _add_steps
    out = []
    for _ in range(n):
        kind = rng.choice(["indirim", "yuzde", "zam", "kdv", "birim", "birim", "bahsis"])
        if kind == "indirim":
            price = rng.choice([50, 80, 100, 120, 150, 200, 250, 300, 450, 500, 750, 1000, 1250, 2000, 3500]) * rng.choice([1, 1, 1, 10])
            pct = rng.choice([5, 10, 15, 20, 25, 30, 40, 50, 60, 70])
            disc = price * pct / 100
            new = price - disc
            q = rng.choice([f"yüzde {pct} indirimle {price} lira kaça gelir", f"{price} liralık üründe yüzde {pct} indirim ne kadar eder",
                            f"{price} tl yüzde {pct} indirimli kaç tl"])
            if float(disc).is_integer():
                d = int(disc)
                a = f"İndirim: {_pct_work(price, pct)} lira. {price} - {d}: {_sub_steps_v2(price, d)}. Sonuç: {price} - {d} = {int(new)} lira."
            else:
                a = f"İndirim: {_pct_work(price, pct)} lira. {price} - {_num(disc)} = {_num(new)} lira."
        elif kind == "yuzde":
            base = rng.choice([20, 40, 50, 60, 80, 120, 200, 240, 300, 400, 500, 600, 800, 1000, 1500, 250, 1200])
            pct = rng.choice([1, 2, 5, 10, 12, 15, 18, 20, 25, 30, 33, 50, 75])
            v = base * pct / 100
            q = rng.choice([f"{base} sayısının yüzde {pct}'i kaçtır", f"{base} in yüzde {pct} kaç eder", f"{base} liranın yüzde {pct}'i kaç", f"yüzde {pct} of {base}"])
            a = f"{_pct_work(base, pct)}. Sonuç: {base} sayısının yüzde {pct}'i {_num(v)}."
        elif kind == "zam":
            price = rng.choice([100, 150, 250, 400, 500, 750, 1000, 2500, 5000, 12000, 25000])
            pct = rng.choice([5, 10, 15, 20, 25, 30, 50])
            inc = int(price * pct / 100)
            q = rng.choice([f"{price} liraya yüzde {pct} zam gelirse ne olur", f"maaşım {price} lira, yüzde {pct} zam alırsam kaç olur"])
            a = f"Zam: {_pct_work(price, pct)} lira. {price} + {inc}: {_add_steps(price, inc)}, yani {price + inc}. Sonuç: yeni tutar {price + inc} lira."
        elif kind == "kdv":
            net = rng.choice([100, 250, 400, 500, 800, 1200, 2000, 3600])
            pct = rng.choice([1, 10, 20])
            v = int(net * pct / 100)
            q = rng.choice([f"{net} liraya yüzde {pct} kdv eklenirse kaç olur", f"{net} lira + yüzde {pct} kdv kaç eder"])
            a = f"KDV: {_pct_work(net, pct)} lira. {net} + {v}: {_add_steps(net, v)}, yani {net + v}. Sonuç: toplam {net + v} lira."
        elif kind == "bahsis":
            bill = rng.choice([120, 180, 240, 350, 480, 600, 900])
            pct = rng.choice([5, 10, 15])
            v = bill * pct / 100
            q = rng.choice([f"{bill} liralık hesaba yüzde {pct} bahşiş ne kadar", f"{bill} lira hesapta yüzde {pct} bahşiş bırakırsam ne öderim"])
            if float(v).is_integer():
                a = f"Bahşiş: {_pct_work(bill, pct)} lira. {bill} + {int(v)}: {_add_steps(bill, int(v))}, yani {bill + int(v)}. Sonuç: toplam {bill + int(v)} lira."
            else:
                a = f"Bahşiş: {_pct_work(bill, pct)} lira. Toplam {bill} + {_num(v)} = {_num(bill + v)} lira."
        else:
            conv = rng.choice([
                ("kilometre", "metre", 1000), ("metre", "santimetre", 100), ("kilogram", "gram", 1000),
                ("saat", "dakika", 60), ("dakika", "saniye", 60), ("ton", "kilogram", 1000),
                ("litre", "mililitre", 1000), ("gün", "saat", 24), ("hafta", "gün", 7), ("yıl", "ay", 12),
            ])
            src, dst, mul = conv
            k = rng.choice([1, 1, 2, 3, 5, 7, 10, 12, 15, 25, 100])
            q = rng.choice([f"{k} {src} kaç {dst}", f"{k} {src} ne kadar {dst} eder", f"{k} {src} kaç {dst} yapar"])
            a = f"1 {src} {mul} {dst} olduğu için {k} x {mul} = {_num(k * mul)} {dst}."
        if rng.random() < 0.4:
            q = q.capitalize()
        out.append([(q, a)])
    return out


def _num(x):
    """Turkish style number: no trailing .0, comma as the decimal separator."""
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.2f}".rstrip("0").rstrip(".").replace(".", ",")


# Turkish proverbs and idioms with their meaning. Culturally common, unattributed, safe to state; asked
# for often ("bana bir atasözü söyle") and absent from the training mix before 2026-08-31.
PROVERBS = [
    ("Damlaya damlaya göl olur.", "Küçük birikimler zamanla büyür; azımsanacak tasarruf ya da çaba yoktur."),
    ("Sakla samanı, gelir zamanı.", "Bugün işe yaramaz görünen şeyi sakla, bir gün lazım olur."),
    ("Ayağını yorganına göre uzat.", "Gelirinden fazlasını harcama, imkanlarına göre yaşa."),
    ("Bir elin nesi var, iki elin sesi var.", "Birlikte çalışmak tek başına çalışmaktan daha verimlidir."),
    ("Ağaç yaşken eğilir.", "Eğitim ve alışkanlıklar küçük yaşta kazandırılır."),
    ("Sütten ağzı yanan yoğurdu üfleyerek yer.", "Bir kez zarar gören, benzer durumlarda aşırı temkinli olur."),
    ("Ne ekersen onu biçersin.", "Yaptığın işin sonucuna katlanırsın; iyilik de kötülük de geri döner."),
    ("Az veren candan, çok veren maldan verir.", "Verilenin değeri miktarına değil, verenin imkanına göre ölçülür."),
    ("Isıracak köpek dişini göstermez.", "Gerçekten zarar verecek kişi niyetini belli etmez."),
    ("Görünen köy kılavuz istemez.", "Sonucu apaçık olan durumu ayrıca anlatmaya gerek yoktur."),
    ("Bugünün işini yarına bırakma.", "Erteleme; işi vaktinde yap."),
    ("Tatlı dil yılanı deliğinden çıkarır.", "Nazik konuşma en zor kişiyi bile ikna eder."),
    ("Komşunun tavuğu komşuya kaz görünür.", "İnsana başkasının sahip olduğu şey daha değerli gelir."),
    ("İşleyen demir ışıldar.", "Çalışan, üreten insan hem gelişir hem değer kazanır."),
    ("Perşembenin gelişi çarşambadan bellidir.", "Bir işin nasıl sonuçlanacağı başlangıcından anlaşılır."),
    ("Üzüm üzüme baka baka kararır.", "İnsan yakın çevresinden etkilenir, huy kapar."),
    ("Denize düşen yılana sarılır.", "Çaresiz kalan, hoşlanmadığı bir yardımı bile kabul eder."),
    ("Acele işe şeytan karışır.", "Aceleyle yapılan işte hata çıkar."),
    ("Dost kara günde belli olur.", "Gerçek dostluk zor zamanda anlaşılır."),
    ("Çok bilen çok yanılır.", "Kendine aşırı güvenen daha çok hata yapar."),
    ("Gülme komşuna, gelir başına.", "Başkasının derdine gülme; aynısı seni de bulabilir."),
    ("Ağzı olan konuşuyor.", "Konu hakkında bilgisi olmayan herkes fikir beyan ediyor."),
    ("Bir musibet bin nasihatten iyidir.", "Başa gelen bir olay, bin öğütten daha çok ders verir."),
    ("Emek olmadan yemek olmaz.", "Çalışmadan kazanç elde edilmez."),
    ("Su testisi su yolunda kırılır.", "İnsan çoğu zaman uğraştığı işin içinde zarar görür."),
]

IDIOMS = [
    ("etekleri zil çalmak", "Çok sevinmek, mutluluktan havalara uçmak."),
    ("burnu havada olmak", "Kibirli davranmak, kendini beğenmek."),
    ("pabucu dama atılmak", "Bir başkası yüzünden gözden düşmek, değerini yitirmek."),
    ("göz kulak olmak", "Birine veya bir şeye dikkat etmek, korumak."),
    ("ağzından baklayı çıkarmak", "Sakladığı sözü sonunda söylemek."),
    ("iğneyle kuyu kazmak", "Çok zor ve yavaş ilerleyen bir işi sabırla yapmak."),
    ("elini taşın altına koymak", "Sorumluluk almak, işin zor kısmına ortak olmak."),
    ("kulak misafiri olmak", "Bir konuşmayı istemeden duymak."),
]


def proverb_examples(rng):
    out = []
    ask = ["bana bir atasözü söyle", "atasözü söyler misin", "güzel bir atasözü", "bir atasözü ve anlamı",
           "bana bir deyim söyle", "deyim öğret", "atasozu soyle"]
    for saying, meaning in PROVERBS:
        q = rng.choice(ask[:4])
        out.append([(q, f"{saying} Anlamı: {meaning}")])
        # the follow up people actually type after a bare proverb: "anlamı ne" was answered with another
        # proverb line in the v38 sweep
        out.append([(rng.choice(ask[:3]), saying), (rng.choice(["anlamı ne", "ne demek bu", "açıklar mısın", "yani"]), f"Anlamı: {meaning}")])
        out.append([(f"\"{saying}\" ne demek", meaning)])
        out.append([(f"{saying.rstrip('.')} ne anlama gelir", meaning)])
    for idiom, meaning in IDIOMS:
        out.append([(rng.choice(ask[4:6]), f"\"{idiom.capitalize()}\": {meaning}")])
        out.append([(f"{idiom} ne demek", meaning)])
    return out


def boundary_examples(rng, path, n: int = 3000):
    """Generated pairs that straddle the line between what the model can answer and what it cannot know
    (ufakzeka.data.gen_boundary). The model had been deciding on surface words: "bugün", "dışarı" and "maç"
    pulled it toward the weather or news refusal even when the question was ordinary advice, and an
    ordinary-looking question about a score pulled it toward inventing one. Both sides are here."""
    import json
    import os
    if not path or not os.path.exists(path):
        return []
    from ufakzeka.train.sft_data import strip_diacritics
    rows = []
    seen = set()
    for line in open(path, encoding="utf-8"):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        q, a = d.get("soru", "").strip(), d.get("cevap", "").strip()
        k = q.lower()
        if not q or not a or k in seen:
            continue
        seen.add(k)
        rows.append((q, a, d.get("kind", "")))
    rng.shuffle(rows)
    out = []
    from ufakzeka.train.sft_data import typo
    for q, a, _kind in rows[:n]:
        forms = [q]
        if rng.random() < 0.25:
            forms.append(typo(strip_diacritics(q), rng))
        if rng.random() < 0.5:
            forms.append(q[0].upper() + q[1:] + ("?" if rng.random() < 0.5 else ""))
        if rng.random() < 0.3:
            forms.append(strip_diacritics(q))
        for form in forms:
            out.append([(form, a)])
    return out


def arithmetic_drills(rng, n: int = 2400):
    """The primitives inside the column working, drilled in exactly the surface form the working uses (a
    primitive drilled in another form does not transfer, arXiv 2410.15580): a labelled addition column with
    and without a carry in, a labelled product column with a carry in, the three subtraction column shapes
    (plain, borrow out, borrow in then out, the zero column), and the "yazılan" accumulation step.
    """
    from ufakzeka.train.sft_data import PLACES
    facts = []
    P = PLACES[:5]
    for x in range(0, 10):
        for y in range(0, 10):
            for c in (0, 1):
                t = x + y + c
                lhs = f"{x} + {y}" + (" + 1" if c else "")
                facts.append((f"{{p}}: {lhs}", f"{{p}}: {lhs} = {t}" + (f", {t % 10} yaz elde 1" if t >= 10 else "")))
    for x in range(0, 10):
        for d in range(2, 10):
            for c in (0, 1, 2, 3, 5, 8):
                t = x * d + c
                lhs = f"{x} x {d}" + (f" + {c}" if c else "")
                facts.append((f"{{p}}: {lhs}", f"{{p}}: {lhs} = {t}" + (f", {t % 10} yaz elde {t // 10}" if t >= 10 else "")))
    for x in range(0, 10):
        for y in range(0, 10):
            if x < y:
                facts.append((f"{{p}}: {x} - {y}", f"{{p}}: {x} < {y}, komşudan 1 al: {x + 10} - {y} = {x + 10 - y}, borç 1"))
            else:
                facts.append((f"{{p}}: {x} - {y}", f"{{p}}: {x} - {y} = {x - y}"))
    for x0 in range(1, 10):
        for y in range(0, 10):
            x = x0 - 1
            if x < y:
                facts.append((f"{{p}}: borç 1, {x0} - {y}", f"{{p}}: borç 1, {x0} - 1 = {x}, {x} < {y}, komşudan 1 al: {x + 10} - {y} = {x + 10 - y}, borç 1"))
            else:
                facts.append((f"{{p}}: borç 1, {x0} - {y}", f"{{p}}: borç 1, {x0} - 1 = {x}, {x} - {y} = {x - y}"))
    for y in range(0, 10):
        facts.append((f"{{p}}: borç 1, 0 - {y}", f"{{p}}: borç 1, 0 - 1 olmaz, komşudan 1 al: 9 - {y} = {9 - y}, borç 1"))
    for _ in range(300):
        acc = rng.randint(1, 9999); d = rng.randint(0, 9)
        facts.append((f"yazılan {acc}, sonra {d}", f"yazılan {acc}, sonra {d}: yazılan {d}{acc}"))
    uniq = {q: a for q, a in facts}
    items = list(uniq.items())
    rng.shuffle(items)
    out = []
    for q, a in items[:n]:
        p = rng.choice(P)
        q, a = q.format(p=p), a.format(p=p)
        form = rng.choice([f"{q} kaç eder", f"{q} kaç", f"{q} = ?", q, f"{q} nasıl olur"])
        out.append([(form, a + ".")])
    return out


