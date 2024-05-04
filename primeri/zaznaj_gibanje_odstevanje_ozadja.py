import cv2 as cv
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uporaba: python zaznaj_gibanje_odstevanje_ozadja.py <nacin>=[mog2,knn]")
        exit(1)
    nacin = sys.argv[1]
    
    if nacin == "mog2":
        odstevalnik_ozadja = cv.createBackgroundSubtractorMOG2()
    elif nacin == "knn":
        odstevalnik_ozadja = cv.createBackgroundSubtractorKNN()
    else:
        print("Napačen način ozadja!")
        exit(1)

    kamera = cv.VideoCapture(0)

    while True:
        ret, slika = kamera.read()
        if slika is None:
            break
        
        premik_maska = odstevalnik_ozadja.apply(slika)    
        
        thresh = cv.threshold(premik_maska, 25, 255, cv.THRESH_BINARY)[1]
        contours, hierarchy = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        if len(contours) != 0:    
            c_max = max(contours, key=cv.contourArea)
            x,y,w,h = cv.boundingRect(c_max)
            cv.rectangle(slika,(x,y),(x+w,y+h),(0,255,0),2)

        cv.imshow('Slika', slika)
        cv.imshow('Maska premika', premik_maska)
        
        if cv.waitKey(5) & 0xFF == ord("q"):
            cv.destroyAllWindows()
            kamera.release()
            break
            