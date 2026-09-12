
import re
import cv2
import fitz  # PyMuPDF: pip install PyMuPDF
import numpy as np
from pathlib import Path
from tkinter import Tk, filedialog
from PIL import Image
import pytesseract

# ============================================================================
# ÖNEMLİ AYARLAR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# PDF sayfalarini render ederken kullanilan olcek (3x ~ 216 DPI).
# Render cozunurlugu dusurulmemelidir; OCR ve filigran maskeleme buna baglidir.
PDF_RENDER_OLCEGI = 3

# Parlaklik esigi: bu gri seviyenin uzerindeki tum pikseller beyaza cekilir.
# OSYM filigrani acik gri/renkli oldugundan silinir, koyu asil metin korunur.
# Filigran kaliyorsa deger dusurulur, metin soluyorsa yukseltilir.
FILIGRAN_PARLAKLIK_ESIGI = 150

# Renkli filigran maskesi: yuksek parlaklik (V) ve belirgin doygunluk (S)
# tasiyan pikseller renkli filigran kabul edilip beyaza cekilir.
FILIGRAN_HSV_V_ESIGI = 120
FILIGRAN_HSV_S_ESIGI = 40
# ============================================================================


def dosya_sec() -> Path | None:
    kok = Tk()
    kok.withdraw()
    kok.attributes("-topmost", True)
    yol = filedialog.askopenfilename(
        title="PDF dosyasini secin",
        initialdir=str(Path.home() / "Desktop"),
        filetypes=[("PDF dosyalari", "*.pdf"), ("Tum dosyalar", "*.*")],
    )
    kok.destroy()
    return Path(yol) if yol else None


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

        # 2) Gri tonlama
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2b) Filigran maskesi: acik tonlu (gri) pikselleri beyaza cek
        gray[gray > FILIGRAN_PARLAKLIK_ESIGI] = 255

        # 2c) Renkli filigran maskesi: parlak ve doygun pikselleri beyaza cek
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        renkli_filigran = (hsv[:, :, 2] > FILIGRAN_HSV_V_ESIGI) & (hsv[:, :, 1] > FILIGRAN_HSV_S_ESIGI)
        gray[renkli_filigran] = 255

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
        try:
            metin = pytesseract.image_to_string(alt_orta_kisim, config=ayarlar).strip()
        except Exception:
            return -1
        rakamlar = re.findall(r'\d+', metin)
        for r in reversed(rakamlar):
            if 0 < int(r) <= 300:
                return int(r)

    return -1  # Okunamazsa en sona atması için


def pdf_sayfalarini_cikar(yol: Path) -> list[bytes]:
    """PDF'in her sayfasini yuksek cozunurlukte render edip PNG byte'lari dondurur."""
    goruntuler = []
    with fitz.open(yol) as belge:
        for sayfa in belge:
            pix = sayfa.get_pixmap(matrix=fitz.Matrix(PDF_RENDER_OLCEGI, PDF_RENDER_OLCEGI))
            goruntuler.append(pix.tobytes("png"))
    return goruntuler


def sayfalari_cikar_ve_isle(yol: Path):
    print("PDF taranıyor, cozunurluk artiriliyor ve yazilar netlestiriliyor...")

    islenmis_sayfalar = []
    for veri in pdf_sayfalarini_cikar(yol):
        temiz = goruntuyu_temizle_ve_netlestir(veri)
        if temiz:
            islenmis_sayfalar.append(temiz)

    return islenmis_sayfalar


def main() -> None:
    print("PDF dosyasini secin...")
    kaynak = dosya_sec()
    if not kaynak:
        print("Dosya secilmedi.")
        return

    print(f"\nSecilen dosya: {kaynak.name}")
    sayfalar = sayfalari_cikar_ve_isle(kaynak)

    if not sayfalar:
        print("\nUyari: Islenecek gecerli bir sayfa bulunamadi.")
        return

    # PDF sayfalari zaten sirali geldiginden yeniden siralama yapilmaz;
    # okunamayan numaralar orijinal duzeni bozmasin diye sadece bilgi amacli raporlanir.
    print(f"\nToplam {len(sayfalar)} sayfa islendi. Orijinal PDF sirasi korunuyor...")

    for sayac, img in enumerate(sayfalar, 1):
        numara = sayfa_numarasi_bul(img)
        okunan = "Bulunamadi" if numara == -1 else numara
        print(f"  Analiz edilen sayfa {sayac}... Algilanan No: {okunan}")

    son_sayfalar = sayfalar

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

    print("\n✅ Islem tamamlandi. Filigrani temizlenmis ve netlestirilmis PDF 'Downloads' klasorune kaydedildi.")


if __name__ == "__main__":
    try:
        main()
    except Exception as hata:
        print("\nBeklenmeyen bir hata olustu:")
        print(hata)
    input("\nKapatmak icin Enter'a basin...")