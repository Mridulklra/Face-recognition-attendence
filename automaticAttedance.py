import tkinter as tk
from tkinter import *
import os
import cv2
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
from tkinter import messagebox

# Define paths
haarcasecade_path = "haarcascade_frontalface_default.xml"
trainimagelabel_path = (
    r"C:\\Users\\Mridul Kalra\\Downloads\\Attendance-Management-system-using-face-recognition\\TrainingImageLabel\\Trainner.yml"
)
trainimage_path = "TrainingImage"
studentdetail_path = (
    r"C:\\Users\\Mridul Kalra\\Downloads\\Attendance-Management-system-using-face-recognition\\StudentDetails\\studentdetails.csv"
)
attendance_path = "Attendance"

def subjectChoose(text_to_speech):
    def FillAttendance():
        sub = subject_entry.get()
        if not sub.strip():
            message = "Please enter the subject name!"
            text_to_speech(message)
            messagebox.showerror("Error", message)
            return

        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            try:
                recognizer.read(trainimagelabel_path)
            except Exception as e:
                message = "Model not found. Please train the model."
                text_to_speech(message)
                notification_label.configure(text=message, fg="red")
                return

            face_cascade = cv2.CascadeClassifier(haarcasecade_path)
            student_data = pd.read_csv(studentdetail_path)
            cam = cv2.VideoCapture(0)

            font = cv2.FONT_HERSHEY_SIMPLEX
            columns = ["Enrollment", "Name"]
            attendance = pd.DataFrame(columns=columns)

            end_time = time.time() + 20  # 20 seconds for attendance collection

            while True:
                ret, frame = cam.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

                for (x, y, w, h) in faces:
                    Id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                    if confidence < 70:
                        name = student_data.loc[student_data["Enrollment"] == Id, "Name"].values
                        attendance.loc[len(attendance)] = [Id, name[0] if len(name) > 0 else "Unknown"]

                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, f"{Id}-{name[0]}", (x, y-10), font, 0.8, (255, 255, 0), 2)
                    else:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(frame, "Unknown", (x, y-10), font, 0.8, (0, 0, 255), 2)

                cv2.imshow("Filling Attendance", frame)
                if cv2.waitKey(1) == 27 or time.time() > end_time:  # Press 'Esc' to exit
                    break

            attendance.drop_duplicates(subset=["Enrollment"], inplace=True)
            cam.release()
            cv2.destroyAllWindows()

            # Save attendance
            if not os.path.exists(attendance_path):
                os.makedirs(attendance_path)
            subject_folder = os.path.join(attendance_path, sub)
            if not os.path.exists(subject_folder):
                os.makedirs(subject_folder)

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            attendance_file = os.path.join(subject_folder, f"{sub}_{timestamp}.csv")
            attendance.to_csv(attendance_file, index=False)

            message = f"Attendance filled successfully for {sub}!"
            notification_label.configure(text=message, fg="green")
            text_to_speech(message)

        except Exception as e:
            message = f"Error during attendance: {str(e)}"
            notification_label.configure(text=message, fg="red")
            text_to_speech(message)

    def OpenAttendanceFolder():
        sub = subject_entry.get()
        if not sub.strip():
            message = "Please enter the subject name!"
            text_to_speech(message)
            messagebox.showerror("Error", message)
            return

        subject_folder = os.path.join(attendance_path, sub)
        if os.path.exists(subject_folder):
            os.startfile(subject_folder)
        else:
            message = f"No attendance records found for subject: {sub}"
            text_to_speech(message)
            messagebox.showinfo("Info", message)

    # Create GUI for subject selection
    subject_window = tk.Tk()
    subject_window.title("Subject Selection")
    subject_window.geometry("600x300")
    subject_window.configure(bg="black")

    title_label = tk.Label(subject_window, text="Enter the Subject Name", bg="black", fg="green", font=("Arial", 20))
    title_label.pack(pady=10)

    subject_label = tk.Label(subject_window, text="Subject", bg="black", fg="yellow", font=("Arial", 15))
    subject_label.pack(pady=5)

    subject_entry = tk.Entry(subject_window, width=30, font=("Arial", 15))
    subject_entry.pack(pady=5)

    fill_button = tk.Button(subject_window, text="Fill Attendance", command=FillAttendance, bg="black", fg="yellow", font=("Arial", 15), width=15)
    fill_button.pack(pady=10)

    open_button = tk.Button(subject_window, text="Open Attendance Folder", command=OpenAttendanceFolder, bg="black", fg="yellow", font=("Arial", 15), width=20)
    open_button.pack(pady=10)

    notification_label = tk.Label(subject_window, text="", bg="black", fg="white", font=("Arial", 12))
    notification_label.pack(pady=5)

    subject_window.mainloop()
