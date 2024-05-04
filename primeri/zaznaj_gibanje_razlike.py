import numpy as np
import cv2 as cv

if __name__ == "__main__":

    kamera = cv.VideoCapture(0)

    prva_slika = None

    while True:
        _, barvna_slika = kamera.read()
    
        siva_slika = cv.cvtColor(barvna_slika, cv.COLOR_BGR2GRAY)
        siva_slika = cv.GaussianBlur(siva_slika, (21, 21), 0)
        if prva_slika is None:
            prva_slika = siva_slika
            continue
            
        if cv.waitKey(10) & 0xFF == ord("q"):
            kamera.release()
            cv.destroyAllWindows()
            break        
        
        razlika = cv.absdiff(prva_slika,siva_slika)
        thresh = cv.threshold(razlika, 25, 255, cv.THRESH_BINARY)[1]
        
        cv.imshow("Razlika slik", razlika)
        cv.imshow("Segmentirana slika", thresh)
        cv.waitKey(1)

        
        thresh = cv.dilate(thresh, None, iterations=2)
        contours, hierarchy = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        if len(contours) != 0:    
            c_max = max(contours, key=cv.contourArea)
            x,y,w,h = cv.boundingRect(c_max)
            cv.rectangle(barvna_slika,(x,y),(x+w,y+h),(0,255,0),2)
            #cv2.drawContours(frameColor, contours, i, (255, 0, 0), 1)
        cv.imshow("Barvna slika", barvna_slika)
    