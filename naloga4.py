import cv2 as cv
import numpy as np

global x_levo_zgoraj
global y_levo_zgoraj
global x_desno_spodaj
global y_desno_spodaj
global slika

x_levo_zgoraj = 0
y_levo_zgoraj = 0
slika = None


def klik_na_sliko(event, x, y, flags, param):
    global x_levo_zgoraj, y_levo_zgoraj, x_desno_spodaj, y_desno_spodaj
    if event == cv.EVENT_LBUTTONUP:
        x_levo_zgoraj = 0
        y_levo_zgoraj = 0

    if event == cv.EVENT_LBUTTONDOWN:
        x_levo_zgoraj = x
        y_levo_zgoraj = y

    if event == cv.EVENT_MOUSEMOVE:
        x_desno_spodaj = x
        y_desno_spodaj = y


def nastavi_obmocje_sledenja(kamera):
    while (1):
        ret, slika = kamera.read()
        if ret == True:
            if x_levo_zgoraj != 0 or y_levo_zgoraj != 0:
                cv.rectangle(slika, (min(x_levo_zgoraj, x_desno_spodaj), min(y_levo_zgoraj, y_desno_spodaj)),
                             (max(x_levo_zgoraj, x_desno_spodaj), max(y_levo_zgoraj, y_desno_spodaj)), (0, 255, 0), 2)

            cv.imshow('Slika', slika)
            if cv.waitKey(100) & 0xFF == ord("q"):
                cv.destroyAllWindows()
                break
            if cv.waitKey(100) & 0xFF == ord("r"):
                sablona = slika[min(y_levo_zgoraj, y_desno_spodaj):max(y_levo_zgoraj, y_desno_spodaj),
                          min(x_levo_zgoraj, x_desno_spodaj):max(x_levo_zgoraj, x_desno_spodaj)]
                okno = (min(x_levo_zgoraj, x_desno_spodaj), min(y_levo_zgoraj, y_desno_spodaj),
                        abs(x_desno_spodaj - x_levo_zgoraj), abs(y_desno_spodaj - y_levo_zgoraj))
                cv.imshow("Sablona", sablona)
        cv.waitKey(1)
    return sablona, okno

def mean_shift(back_projected_image, window_location, iterations, error_threshold):
    window_x, window_y, window_width, window_height = window_location
    for _ in range(iterations):
        # Calculate the centroid of the window
        region_of_interest = back_projected_image[window_y:window_y + window_height, window_x:window_x + window_width]
        moments = cv.moments(region_of_interest)
        if moments["m00"] != 0:
            centroid_x = int(moments["m10"] / moments["m00"])
            centroid_y = int(moments["m01"] / moments["m00"])
        else:
            centroid_x, centroid_y = window_width // 2, window_height // 2

        # Move the window to the centroid
        delta_x = centroid_x - window_width // 2
        delta_y = centroid_y - window_height // 2

        # Check for convergence
        if abs(delta_x) < error_threshold and abs(delta_y) < error_threshold:
            break

        window_x += delta_x
        window_y += delta_y

    return window_x, window_y, window_width, window_height

'''def calc_back_project(image, channels, hist, ranges):
    # Convert the image from BGR to HSV color space
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Flatten the image and the histogram
    hsv_flat = hsv.flatten()
    hist_flat = hist.flatten()

    # Initialize the back projection
    back_proj = np.zeros_like(hsv_flat)

    # For each pixel in the image
    for i in range(len(hsv_flat)):
        # Find the corresponding bin in the histogram
        bin = np.digitize(hsv_flat[i], bins=ranges) - 1

        # Assign the value of the bin to the pixel in the back projection
        back_proj[i] = hist_flat[bin]

    # Reshape the back projection to the shape of the original image
    back_proj = back_proj.reshape(hsv.shape)

    return back_proj'''


def izracunaj_povratno_projekcijo(slika, histogram):
    # Iz slike izlušči kanal odtenka (hue)
    kanal = slika[:, :, 0]

    # Uporabi vrednosti odtenka kot indekse za iskanje ustreznih vrednosti v histogramu
    # Opomba: Prepričajmo se, da so indeksi celoštevilske vrednosti
    povratna_projekcija = histogram[kanal.astype(int)]

    # Normaliziraj povratno projekcijo, da imajo vrednosti v obsegu [0, 255]
    cv.normalize(povratna_projekcija, povratna_projekcija, 0, 255, cv.NORM_MINMAX)

    # Pretvori povratno projekcijo v 8-bitno sliko
    povratna_projekcija = povratna_projekcija.astype(np.uint8)

    return povratna_projekcija


def camshift(slika, sablone, lokacije_oken, iteracije, napaka):

    hsv = cv.cvtColor(slika, cv.COLOR_BGR2HSV)
    nova_lokacija_oken = []
    for sablona, lokacija_okna in zip(sablone, lokacije_oken):
        #povratna_projekcija = calc_back_project(hsv, [0, 1], sablona, [0, 180, 0, 256])
        #povratna_projekcija = cv.calcBackProject([hsv], [0, 1], sablona, [0, 180, 0, 256], 1)
        povratna_projekcija = izracunaj_povratno_projekcijo(hsv, sablona)
        lokacija_okna = mean_shift(povratna_projekcija, lokacija_okna, iteracije, napaka)
        nova_lokacija_oken.append(lokacija_okna)
    return nova_lokacija_oken


def zaznaj_gibanje(kamera, objekti=2):
    """Docstring."""
    ret, prejsnja_slika = kamera.read()
    prejsnja_slika = cv.cvtColor(prejsnja_slika, cv.COLOR_BGR2GRAY)
    lokacije_objektov = []

    while len(lokacije_objektov) < objekti:
        ret, trenutna_slika = kamera.read()
        trenutna_slika = cv.cvtColor(trenutna_slika, cv.COLOR_BGR2GRAY)

        razlika = cv.absdiff(prejsnja_slika, trenutna_slika)
        _, razlika = cv.threshold(razlika, 25, 255, cv.THRESH_BINARY)

        razlika = cv.dilate(razlika, None, iterations=2)

        konture, _ = cv.findContours(razlika.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for kontura in konture:
            if cv.contourArea(kontura) > 500:  # ignoriraj majhne konture
                (x, y, w, h) = cv.boundingRect(kontura)
                lokacije_objektov.append((x, y, w, h))

        prejsnja_slika = trenutna_slika.copy()

    return lokacije_objektov[:objekti]


def izracunaj_histogram_in_normaliziraj(image, mask, bins, ranges):
    """Docstring."""
    # Apply the mask to the image
    masked_image = cv.bitwise_and(image, image, mask=mask)

    # Calculate the histogram
    histogram, bin_edges = np.histogram(masked_image.ravel(), bins=bins, range=ranges)

    # Normalize the histogram
    histogram = histogram.astype('float')
    histogram /= (histogram.sum() + np.finfo(float).eps)

    return histogram


def izracunaj_znacilnice(lokacije_oken, prva_slika):
    """Docstring."""
    if prva_slika is None:
        print("prva_slika is None")
        return []
    hsv = cv.cvtColor(prva_slika, cv.COLOR_BGR2HSV)
    if hsv is None:
        print("hsv is None")
        return []
    sablone = []
    for lokacija_okna in lokacije_oken:
        x, y, w, h = lokacija_okna
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > prva_slika.shape[1] or y + h > prva_slika.shape[0]:
            print(f"Invalid lokacija_okna: {lokacija_okna}")
            continue
        roi = hsv[y:y + h, x:x + w]
        if roi.size == 0:
            print(f"Empty roi for lokacija_okna: {lokacija_okna}")
            continue
        lower_color_range = np.array([0., 0., 0.])
        upper_color_range = np.array([180., 255., 255.])

        mask = cv.inRange(roi, lower_color_range, upper_color_range)
        roi_hist = izracunaj_histogram_in_normaliziraj(roi, mask, 180, [0, 180])
        # roi_hist = cv.calcHist([roi], [0], mask, [180], [0, 180])
        # cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
        sablone.append(roi_hist)
    return sablone


if __name__ == "__main__":
    # Naloži video
    zaznaj_gibanje = "rocno"
    # cap = cv.VideoCapture(1)
    cap = cv.VideoCapture("video2.mp4")

    # Preveri, če je video uspešno naložen
    if not cap.isOpened():
        print("Napaka: Video ne obstaja ali ni bil uspešno naložen.")
        exit(1)

    cv.namedWindow("Slika")
    cv.setMouseCallback("Slika", klik_na_sliko)

    sablona, okno = nastavi_obmocje_sledenja(cap)

    # Nastavitve meanshift algoritma
    iteracije = 200
    napaka = 1
    lokacije_oken = list()
    # Začetna točka sledenja ročno
    # (x, y, w, h)
    if zaznaj_gibanje == "rocno":
        lokacije_oken.append(okno)
    else:
        # Začetna točka sledenja avtomatsko
        lokacije_oken = zaznaj_gibanje(cap, st_objektov=2)

    # Izračun značilnic za sledenje
    uspel, prva_slika = cap.read()
    sablone = izracunaj_znacilnice(lokacije_oken, prva_slika)

    # Začetek sledenja
    while True:
        uspel, slika = cap.read()
        if not uspel:
            break

        lokacije_novih_oken = camshift(slika, sablone, lokacije_oken, iteracije, napaka)
        lokacije_oken = lokacije_novih_oken
        # Nariši okno
        for okno in lokacije_novih_oken:
            x, y, w, h = okno
            cv.rectangle(slika, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv.imshow('Rezultat', slika)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
