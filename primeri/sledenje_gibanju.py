import cv2 as cv
import numpy as np
import sys

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
                cv.rectangle(slika, (x_levo_zgoraj, y_levo_zgoraj), (x_desno_spodaj, y_desno_spodaj), (0, 255, 0), 2)

            cv.imshow('Slika', slika)
            if cv.waitKey(100) & 0xFF == ord("q"):
                cv.destroyAllWindows()
                # kamera.release()
                break
            if cv.waitKey(100) & 0xFF == ord("r"):
                sablona = slika[y_levo_zgoraj:y_desno_spodaj, x_levo_zgoraj:x_desno_spodaj]
                okno = (x_levo_zgoraj, y_levo_zgoraj, x_desno_spodaj - x_levo_zgoraj, y_desno_spodaj - y_levo_zgoraj)
                cv.imshow("Sablona", sablona)
        cv.waitKey(1)
    return sablona, okno


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uporaba: python sledenje_gibanju.py <nacin>=[meanshift,camshift]")
        exit(1)

    metoda = sys.argv[1]
    if metoda != "meanshift" and metoda != "camshift":
        print("Napacna metoda!")
        exit(1)

    #open videoo called video2.mp4
    kamera = cv.VideoCapture("video2.mp4")
    cv.namedWindow("Slika")
    cv.setMouseCallback("Slika", klik_na_sliko)

    sablona, okno = nastavi_obmocje_sledenja(kamera)

    if sablona is not None:
        roi = sablona
        hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

        roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

        konvergencni_kriterij = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

        while (1):
            ret, slika = kamera.read()
            slika = cv.flip(slika, 1)

            if ret == True:
                hsv = cv.cvtColor(slika, cv.COLOR_BGR2HSV)
                dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

                if metoda == "meanshift":
                    _, okno = cv.meanShift(dst, okno, konvergencni_kriterij)
                else:
                    _, okno = cv.CamShift(dst, okno, konvergencni_kriterij)

                x, y, w, h = okno
                rezultat = cv.rectangle(slika, (x, y), (x + w, y + h), 255, 2)
                cv.imshow('img2', rezultat)
                k = cv.waitKey(5) & 0xff
                if k == 27:
                    break
            else:
                break
        cv.destroyAllWindows()
        kamera.release()
