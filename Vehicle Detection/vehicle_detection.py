import cv2
import numpy as np



# Initialize the background object.
backgroundObject = cv2.createBackgroundSubtractorMOG2()

cap = cv2.VideoCapture('028e0ffa-70991567.mp4')

detect = []
offset = 3
counter = 0
if (cap.isOpened() == False):
    print("Error reading video file")

print(cap.get(3))
four_cc = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter('vehicle_count.avi', four_cc, 20, (int(cap.get(3)), int(cap.get(4))))
while True:

    # Read a new frame.
    ret, frame = cap.read()

    # Check if frame is not read correctly.
    if not ret:
        break

    # Apply Morphological Operations

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # converts image to gray
    blur = cv2.GaussianBlur(gray,(3,3),5)


    fgmask = backgroundObject.apply(blur)  # uses the background subtraction

    # applies different thresholds to fgmask to try and isolate cars
    # just have to keep playing around with settings until cars are easily identifiable

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # kernel to apply to the morphology
    closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    dilation = cv2.dilate(opening, kernel)

    img1 = np.hstack((closing,opening,dilation))

    # # creates contours
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    # line created to stop counting contours, needed as cars in distance become one big contour
    lineypos = 1400
    cv2.line(frame, (0, lineypos), (3840, lineypos), (255, 0, 0), 5)


    # min area for contours in case a bunch of small noise contours are created
    minarea = 10000

    # max area for contours, can be quite large for buses
    maxarea = 20000




    for i in range(len(contours)):  # cycles through all contours in current frame

        if hierarchy[0, i, 3] == -1:  # using hierarchy to only count parent contours (contours not within others)

            area = cv2.contourArea(contours[i])  # area of contour

            if minarea < area < maxarea:  # area threshold for contour

                # calculating centroids of contours
                cnt = contours[i]
                M = cv2.moments(cnt)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])


                # print(cy)

                if cy > lineypos-200:  # filters out contours that are above line (y starts at top)


                    x, y, w, h = cv2.boundingRect(cnt)
                    print(w,h)

                    if w<400 and h<300:

                        detect.append((cx,cy))

                        # creates a rectangle around contour
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)

                        # Prints centroid text in order to double check later on
                        cv2.putText(frame, str(cx) + "," + str(cy), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    .3, (0, 0, 255), 1)
                        print(f"Detect Array: {detect}")
                        for (x,y) in detect:

                            if y<(lineypos+offset) and y> (lineypos-offset):
                                counter+=1



                        print(f"Count: {counter}")
                        detect.remove((x,y))







    cv2.putText(frame,f"Vehicle Counter: {str(counter)}",(100,200),cv2.FONT_HERSHEY_SIMPLEX,
                                3, (0, 0, 255), 8)

    out.write(frame)
    cv2.imshow("Cars",cv2.resize(img1,None,fx=0.1,fy=0.3))
    cv2.imshow("Frame",cv2.resize(frame,None,fx=0.3,fy=0.3))
    k = cv2.waitKey(1)

    # Check if 'q' key is pressed.
    if k == ord('q'):
        # Break the loop.
        break

# Release the VideoCapture Object.
cap.release()

# Close the windows
cv2.destroyAllWindows()