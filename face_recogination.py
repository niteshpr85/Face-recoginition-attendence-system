from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import messagebox
import mysql.connector
from datetime import datetime
import cv2
import numpy as np
import os


class Face_Recognition:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1530x790+0+0")
        self.root.title("Face Recognition System")

        title_lbl = Label(self.root, text="FACE RECOGNITION", 
                          font=("Sketchy In Snow", 35, "bold"), 
                          bg="white", fg="green")
        title_lbl.place(x=0, y=0, width=1530, height=45)

        # Left Image
        img_top = Image.open(r"college_images\face_detector1.jpg")
        img_top = img_top.resize((650, 700), Image.LANCZOS)
        self.photoimg_top = ImageTk.PhotoImage(img_top)

        f_lbl = Label(self.root, image=self.photoimg_top)
        f_lbl.place(x=0, y=55, width=650, height=700)

        # Right Image
        img_bottom = Image.open(r"college_images\face_det.png")
        img_bottom = img_bottom.resize((950, 700), Image.LANCZOS)
        self.photoimg_bottom = ImageTk.PhotoImage(img_bottom)

        f_lbl2 = Label(self.root, image=self.photoimg_bottom)
        f_lbl2.place(x=650, y=55, width=950, height=700)

        # Button
        b1_1 = Button(f_lbl2, text="Face Recognition", command=self.face_recog,
                      cursor="hand2", font=("times new roman", 18, "bold"), 
                      bg="red", fg="white")
        b1_1.place(x=365, y=620, width=200, height=40)

    # ------------------------- Attendance ---------------------
    def mark_attendance(self, i, r, n, d):
        filename = "MyTest.csv"
        
        # If CSV doesn't exist create header
        if not os.path.exists(filename):
            with open(filename, "w") as f:
                f.write("ID,Roll,Name,Dept,Time,Date,Status")

        with open(filename, "r+", newline="\n") as f:
            data = f.readlines()
            name_list = []

            for line in data:
                entry = line.split(",")
                name_list.append(entry[0])   # student_id check

            # prevent duplicate entry
            if i not in name_list:
                now = datetime.now()
                date = now.strftime("%d/%m/%Y")
                time = now.strftime("%H:%M:%S")

                f.write(f"\n{i},{r},{n},{d},{time},{date},Present")

    # ------------------------- Face Recognition ---------------------
    def face_recog(self):

        def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            features = classifier.detectMultiScale(gray, scaleFactor, minNeighbors)

            for (x, y, w, h) in features:
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)

                id, pred = clf.predict(gray[y:y + h, x:x + w])
                confidence = int(100 * (1 - pred / 300))

                # Database
                conn = mysql.connector.connect(
                    host="localhost",
                    user="root",
                    password="Aktrkt5080@",
                    database="face_recognizer"
                )
                my_cursor = conn.cursor()

                my_cursor.execute("SELECT id FROM student WHERE id=%s", (id,))
                i = my_cursor.fetchone()

                my_cursor.execute("SELECT univ_roll FROM student WHERE id=%s", (id,))
                r = my_cursor.fetchone()

                my_cursor.execute("SELECT name FROM student WHERE id=%s", (id,))
                n = my_cursor.fetchone()

                my_cursor.execute("SELECT dep FROM student WHERE id=%s", (id,))
                d = my_cursor.fetchone()

                conn.close()

                if i and r and n and d and confidence > 77:
                    i, r, n, d = str(i[0]), str(r[0]), str(n[0]), str(d[0])

                    cv2.putText(img, f"ID: {i}", (x, y - 75),
                                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(img, f"Roll: {r}", (x, y - 55),
                                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(img, f"Name: {n}", (x, y - 30),
                                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(img, f"Dept: {d}", (x, y - 5),
                                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

                    self.mark_attendance(i, r, n, d)

                else:
                    cv2.rectangle(img, (x, y), (x + w, y + h), 
                                  (0, 0, 255), 3)
                    cv2.putText(img, "Unknown Face", (x, y - 5), 
                                cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)

        def recognize(img, clf, faceCascade):
            draw_boundary(img, faceCascade, 1.1, 10, (255, 0, 255), "Face", clf)
            return img

        # Load classifier
        faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        clf = cv2.face.LBPHFaceRecognizer_create()
        clf.read("classifier.xml")

        video_cap = cv2.VideoCapture(0)

        while True:
            ret, img = video_cap.read()
            img = recognize(img, clf, faceCascade)
            cv2.imshow("Welcome to Face Recognition", img)

            if cv2.waitKey(1) == 13:   # Enter key
                break

        video_cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    root = Tk()
    obj = Face_Recognition(root)
    root.mainloop()
