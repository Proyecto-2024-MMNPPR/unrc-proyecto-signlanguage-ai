# This is a program that allows you to record multiple videos and save them into the data folder.
import cv2 as cv

cap = cv.VideoCapture(0)
fourcc = cv.VideoWriter_fourcc(*'XVID')
count = 0
finished = False
while not finished:
    symbol = input("Which symbol do you want to capture? \n")
    out = cv.VideoWriter("data/" + symbol + str(count) + '.avi', fourcc, 30.0, (640,  480))
    
    started = False
    stopped = False
    print("Press q to start")
    while not started:
        ret, frame = cap.read()
        cv.imshow('frame', frame)
        started = cv.waitKey(25) == ord('q')

    if started:
        print("Press q to stop recording")
        while not stopped:
            ret, frame = cap.read()
            cv.imshow('frame', frame)
            out.write(frame)
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            stopped = cv.waitKey(1) == ord('q')
        out.release()
        cv.destroyAllWindows()
        finished = input("Press q to quit or any other character to record another symbol.") == 'q'
cap.release()
