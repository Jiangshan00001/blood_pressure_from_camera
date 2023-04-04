# coding:utf-8


import sys
import cv2
import numpy as np
from scipy.signal import butter, lfilter
from sklearn.linear_model import LinearRegression
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QProgressBar
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
from scipy.signal import find_peaks
from PyQt5.QtCore import Qt


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Blood Pressure Monitor")
        self.resize(800, 600)
        # create a label to display the video
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        # create a label to display the blood pressure
        self.bp_label = QLabel()
        self.bp_label.setAlignment(Qt.AlignCenter)
        self.wait_process = QProgressBar()
        self.wait_process.setAlignment(Qt.AlignCenter)
        self.wait_process.setValue(0)
        self.wait_process.setMaximum(100)
        self.wait_process.setMinimum(0)
        # create a vertical layout
        self.layout = QVBoxLayout()
        # add the labels to the layout
        self.layout.addWidget(self.video_label)
        self.layout.addWidget(self.bp_label)
        self.layout.addWidget(self.wait_process)
        # set the layout
        self.setLayout(self.layout)
        # create a timer object
        self.timer = QTimer()
        # set the timer interval to 10 ms
        self.timer.setInterval(10)
        # connect the timeout signal to the capture_frame slot
        self.timer.timeout.connect(self.capture_frame)
        # initialize some variables for PPG analysis
        self.fs = 100  # sampling frequency in Hz
        self.buffer_size = 10  # buffer size in seconds
        self.buffer = []  # buffer for storing PPG signals
        self.time = []  # buffer for storing timestamps
        self.systolic = []  # buffer for storing systolic BP values
        self.diastolic = []  # buffer for storing diastolic BP values

    def update_image(self, qimg):
        # update the video label with the new image
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    def update_bp(self, sbp, dbp):
        # update the bp label with the new values
        self.bp_label.setText(f"Systolic: {sbp:.1f} mmHg\nDiastolic: {dbp:.1f} mmHg")

    def capture_frame(self):
        # capture video from camera (index 0)
        ret, frame = self.cap.read()
        if ret:
            # convert the frame to RGB format
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # convert the frame to QImage format
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            # update the image
            self.update_image(qimg)
            # estimate blood pressure from frame using PPG method
            sbp, dbp,ready_step = self.estimate_bp(rgb)

            if ready_step!=100:
                self.wait_process.setValue(ready_step)
            else:
                self.wait_process.setValue(100)

            # update blood pressure values
            self.update_bp(sbp, dbp)

    def estimate_bp(self, rgb):
        ready_step=100
        # extract green channel from RGB image (assumed to be PPG signal)
        green = rgb[:, :, 1]
        # apply a bandpass filter to remove noise and baseline drift
        green = self.bandpass_filter(green, 0.5, 5, self.fs, order=5)
        # calculate the mean intensity of the green channel
        intensity = np.mean(green)
        # get the current timestamp
        timestamp = cv2.getTickCount() / cv2.getTickFrequency()
        # append the intensity and timestamp to the buffer
        self.buffer.append(intensity)
        self.time.append(timestamp)
        # check if the buffer is full
        if len(self.buffer) > self.buffer_size * self.fs:
            # remove the oldest element from the buffer
            self.buffer.pop(0)
            self.time.pop(0)
            # convert the buffer to a numpy array
            signal = np.array(self.buffer)
            time = np.array(self.time)
            # normalize the signal
            signal = (signal - np.mean(signal)) / np.std(signal)
            # find the peaks and valleys of the signal
            peaks, _ = find_peaks(signal, height=0)
            valleys, _ = find_peaks(-signal, height=0)
            # calculate the pulse rate from the peaks
            pr = len(peaks) / (time[-1] - time[0]) * 60
            # calculate the pulse pressure from the peaks and valleys
            pp = np.mean(signal[peaks]) - np.mean(signal[valleys])
            # use a linear regression model to estimate blood pressure from pulse rate and pulse pressure
            # based on empirical data from previous studies
            X = np.array([[pr, pp]])
            y_systolic = self.systolic_model.predict(X)
            y_diastolic = self.diastolic_model.predict(X)
            # append the estimated blood pressure values to the buffer
            self.systolic.append(y_systolic[0])
            self.diastolic.append(y_diastolic[0])
            # calculate the mean blood pressure values from the buffer
            sbp = np.mean(self.systolic)
            dbp = np.mean(self.diastolic)
        else:
            # use default blood pressure values
            sbp = 120
            dbp = 80
            ready_step = 100*len(self.buffer) // (self.buffer_size * self.fs)
        # return the estimated blood pressure values
        return sbp, dbp,ready_step

    def bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        # apply a Butterworth bandpass filter to the data
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = lfilter(b, a, data)
        return y

    def train_model(self):
        # train a linear regression model to estimate blood pressure from pulse rate and pulse pressure
        # based on empirical data from previous studies (Table 2 in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4998763/)
        X_train = np.array(
            [[72.3, 47.9], [72.8, 46.7], [73.3, 45.6], [73.9, 44.4], [74.4, 43.3], [74.9, 42.1], [75.4,41], [75.9,39.8],
             [76.4,38.7], [76.9, 37.5], [77.4, 36.4], [77.9,  35.2], [78.4,34.1], [78.9,32.9], [79.4,31.8], [79.9,30.6],
             [80.4, 29.5], [80.9,28.3], [81.4,27.2], [81.9, 26]])
        y_systolic_train = np.array([121.6, 120, 118.5, 116.9, 115.4, 113.8, 112.3, 110.7, 109.2, 107.6, 106, 104.5, 102.9, 101.4, 99.8, 98.3, 96.7, 95.2, 93.6, 92])
        y_diastolic_train = np.array([79, 77.8, 76.7, 75.6, 74.4, 73.3, 72.1, 71, 69.8, 68.7, 67.6, 66.4, 65.3, 64.1, 63, 61.9, 60.8, 59.6, 58.5, 57.3])
        # fit the linear regression model
        self.systolic_model = LinearRegression().fit(X_train, y_systolic_train)
        self.diastolic_model = LinearRegression().fit(X_train, y_diastolic_train)

    def start_video(self):
        # open the camera
        self.cap = cv2.VideoCapture(0)
        # train the model
        self.train_model()
        # start the timer
        self.timer.start()

    def stop_video(self):
        # stop the timer
        self.timer.stop()
        # release the camera
        self.cap.release()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    window.start_video()
    sys.exit(app.exec_())
