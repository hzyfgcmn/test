
import base64
import re
import cv2
import fitz  # PyMuPDF (kurulum: pip install PyMuPDF)
import numpy as np
from pathlib import Path
from email import policy
from email.parser import BytesParser
from tkinter import Tk, filedialog
from PIL import Image
import pytesseract

# ============================================================================
# ÖNEMLİ AYARLAR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Filigran temizleme esikleri (filigran kalirsa dusur, metin bozulursa yukselt)
FILIGRAN_PARLAKLIK_ESIGI = 105   # Bu gri seviyenin ustundeki pikseller beyaza cekilir
FILIGRAN_HSV_V_ESIGI = 110       # HSV parlaklik (V) esigi: acik tonlu filigran
FILIGRAN_HSV_S_ESIGI = 90        # HSV doygunluk (S) esigi: renkli ama soluk filigran
# ============================================================================


def dosya_sec() -> Path | None:
    kok = Tk()
    kok.withdraw()
    kok.attributes("-topmost", True)
    yol = filedialog.askopenfilename(
        title="PDF, MHTML veya HTML dosyasini secin",
        initialdir=str(Path.home() / "Desktop"),
        filetypes=[
            ("PDF dosyalari", "*.pdf"),
            ("Web kayitlari", "*.mhtml *.mht *.html *.htm"),
            ("Tum dosyalar", "*.*"),
        ],
    )
    kok.destroy()
    return Path(yol) if yol else None


def pdf_sayfalarini_cikar(yol: Path) -> list[bytes]:
    """PDF sayfalarini yuksek cozunurlukte render edip PNG byte'lari olarak dondurur."""
    sayfa_goruntuleri: list[bytes] = []
    with fitz.open(yol) as belge:
        for sayfa in belge:
            pix = sayfa.get_pixmap(matrix=fitz.Matrix(3, 3))
            sayfa_goruntuleri.append(pix.tobytes("png"))
    return sayfa_goruntuleri


def filigrani_sil(img):
    """Acik tonlu / soluk renkli filigran piksellerini beyaza boyar, koyu metni korur."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]

    # Yuksek parlaklik + dusuk/orta doygunluk: tipik ÖSYM filigrani
    filigran_maskesi = (v > FILIGRAN_HSV_V_ESIGI) & (s < FILIGRAN_HSV_S_ESIGI)

    # Gri tonlamada da acik pikselleri beyaza cek (capraz tekrarli desen kalintilari)
    gri = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filigran_maskesi |= gri > FILIGRAN_PARLAKLIK_ESIGI

    img = img.copy()
    img[filigran_maskesi] = 255
    return img


def goruntuyu_temizle_ve_netlestir(resim_verisi: bytes):
    """
    Filigranı siler, çözünürlüğü artırır, gürültüyü azaltır ve
    zemin rengine göre otomatik olarak harfleri kalınlaştırır.
    """
    try:
        nparr = np.frombuffer(resim_verisi, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None or img.shape[0] < 300 or img.shape[1] < 300:
            return None

        # 1) Çözünürlüğü artır (Lanczos, cubic'ten daha net sonuç verir)
        img = cv2.resize(img, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_LANCZOS4)

        # 2) Filigrani sil ve gri tonlamaya gec
        img = filigrani_sil(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 3) Gürültü azaltma (filigran kalıntılarını ve JPEG artefaktlarını temizler)
        gray = cv2.fastNlMeansDenoising(gray, h=10)

        # 4) Adaptif eşikleme (sabit değerler yerine, ışık/kontrast farklarına dayanıklı)
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 15
        )

        # 5) Zemin rengini otomatik tespit et ve doğru yönde kalınlaştır
        beyaz_oran = np.mean(thresh) / 255.0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

        if beyaz_oran > 0.5:
            # Beyaz zemin + siyah yazı -> erode ile yazıyı kalınlaştır
            thresh = cv2.erode(thresh, kernel, iterations=1)
        else:
            # Siyah zemin + beyaz yazı -> dilate ile yazıyı kalınlaştır
            thresh = cv2.dilate(thresh, kernel, iterations=1)

        return Image.fromarray(thresh)

    except Exception as e:
        print(f"  [!] Goruntu islenirken hata (Atlandi): {e}")
        return None


def sayfa_numarasi_bul(img: Image.Image) -> int:
    """Sayfanın alt-orta kısmını keser, büyütür ve OCR ile sayfa numarasını okur."""
    w, h = img.size

    sol = int(w * 0.25)
    sag = int(w * 0.75)
    ust = int(h * 0.90)

    alt_orta_kisim = img.crop((sol, ust, sag, h))

    # OCR doğruluğunu artırmak için kırpılan alanı büyüt
    alt_orta_kisim = alt_orta_kisim.resize(
        (alt_orta_kisim.width * 2, alt_orta_kisim.height * 2),
        Image.LANCZOS
    )

    # Farklı PSM modlarını sırayla dene (tek satır, tek kelime, blok)
    for psm in (7, 8, 6):
        ayarlar = f'--psm {psm} -c tessedit_char_whitelist=0123456789'
        metin = pytesseract.image_to_string(alt_orta_kisim, config=ayarlar).strip()
        rakamlar = re.findall(r'\d+', metin)
        for r in reversed(rakamlar):
            if 0 < int(r) <= 300:
                return int(r)

    return -1  # Okunamazsa en sona atması için


def sayfalari_cikar_ve_isle(yol: Path):
    islenmis_sayfalar = []

    print("Dosya taranıyor, filigran siliniyor ve yazilar netlestiriliyor...")

    if yol.suffix.lower() == ".pdf":
        for veri in pdf_sayfalarini_cikar(yol):
            temiz = goruntuyu_temizle_ve_netlestir(veri)
            if temiz:
                islenmis_sayfalar.append(temiz)
        return islenmis_sayfalar

    ham = yol.read_bytes()
    mesaj = BytesParser(policy=policy.default).parsebytes(ham)

    parcalar = mesaj.walk() if mesaj.is_multipart() else [mesaj]
    for parca in parcalar:
        tip = (parca.get_content_type() or "").lower()
        if not tip.startswith("image/"):
            continue

        veri = parca.get_payload(decode=True)
        if veri:
            temiz = goruntuyu_temizle_ve_netlestir(veri)
            if temiz:
                islenmis_sayfalar.append(temiz)

    metin = ham.decode("utf-8", errors="ignore")
    for eslesme in re.finditer(
        r"data:(image/(?:jpeg|jpg|png|webp|gif));base64,([A-Za-z0-9+/=\s]+)",
        metin, flags=re.I
    ):
        try:
            veri = base64.b64decode(re.sub(r"\s+", "", eslesme.group(2)))
            if len(veri) > 5000:
                temiz = goruntuyu_temizle_ve_netlestir(veri)
                if temiz:
                    islenmis_sayfalar.append(temiz)
        except Exception:
            continue

    return islenmis_sayfalar


def main() -> None:
    print("PDF, MHTML veya HTML dosyasini secin...")
    kaynak = dosya_sec()
    if not kaynak:
        print("Dosya secilmedi.")
        return

    print(f"\nSecilen dosya: {kaynak.name}")
    sayfalar = sayfalari_cikar_ve_isle(kaynak)

    if not sayfalar:
        print("\nUyari: Islenecek gecerli bir sayfa bulunamadi.")
        return

    pdf_girdisi = kaynak.suffix.lower() == ".pdf"

    if pdf_girdisi:
        # PDF sayfalari zaten sirali geldigi icin orijinal sira korunur
        print(f"\nToplam {len(sayfalar)} sayfa islendi. PDF girdisi oldugu icin orijinal sayfa sirasi korunuyor...")
        son_sayfalar = sayfalar
    else:
        print(f"\nToplam {len(sayfalar)} sayfa islendi. Sayfa numaralari tespit edilip BUYUKTEN KUCUGE siralanmaya basliyor...")

        sirali_liste = []
        for sayac, img in enumerate(sayfalar, 1):
            numara = sayfa_numarasi_bul(img)
            sirali_liste.append({
                "numara": numara,
                "resim": img
            })
            okunan = "Bulunamadi (Sona eklenecek)" if numara == -1 else numara
            print(f"  Analiz edilen sayfa {sayac}... Algilanan No: {okunan}")

        # Büyükten küçüğe sıralama
        sirali_liste.sort(key=lambda x: x["numara"], reverse=True)

        son_sayfalar = [eleman["resim"] for eleman in sirali_liste]

    hedef_pdf = Path.home() / "Downloads" / f"Net_Sirali_{kaynak.stem}.pdf"
    print(f"\nPDF olusturuluyor: {hedef_pdf.name}")

    ilk_sayfa = son_sayfalar[0]
    ilk_sayfa.save(
        hedef_pdf,
        "PDF",
        resolution=300.0,
        save_all=True,
        append_images=son_sayfalar[1:]
    )

    if pdf_girdisi:
        print("\n✅ Islem tamamlandi. Filigrani temizlenmis PDF 'Downloads' klasorune kaydedildi.")
    else:
        print("\n✅ Islem tamamlandi. Netlestirilmis ve BUYUKTEN KUCUGE sirali PDF 'Downloads' klasorune kaydedildi.")


if __name__ == "__main__":
    try:
        main()
    except Exception as hata:
        print("\nBeklenmeyen bir hata olustu:")
        print(hata)
    input("\nKapatmak icin Enter'a basin...")