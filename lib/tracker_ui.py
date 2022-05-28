import time
import numpy as np
import cv2

from tkinter import *
from tkinter.ttk import *
from PIL import Image, ImageTk

from lib.pupil_detect import *
from lib.gaze_estimate_polynomial import *
from lib.gaze_estimate import *


class TrackerGUI:
    def __init__(self, capA_id, capB_id):
        self.tk = Tk()

        self.capA = cv2.VideoCapture(capA_id)
        self.capB = cv2.VideoCapture(capB_id)

        self.factor_threshold_ratio = 0.7
        self.factor_dilate = 5
        self.factor_erode = 8

        self.ret_para, self.parameter = load_mapping_parameter()
        self.ret_mapping, self.mapping_matrix = load_mapping_matrix()

        self.pupil_point = None
        self.gaze_point = None
        self.gaze_point_p = None

        self.frameA_size = (320, 180)
        self.frameB_size = (1280, 720)
        self.frameA = None
        self.frameB = None

        self.imgA_size = (640, 360)
        self.imgB_size = (640, 360)
        self.imgA = None
        self.imgB = None

        self.suspend_flag = False
        self.calibration_flag = False
        self.fps_flag = False
        self.gaze_method_flag = 1

        self.eye_range_para = [0.2, 0.25]
        self.img_range = [
            int(self.frameA_size[0] * self.eye_range_para[0]),
            int(self.frameA_size[1] * self.eye_range_para[1]),
            int(self.frameA_size[0] * (1 - self.eye_range_para[0])),
            int(self.frameA_size[1] * (1 - self.eye_range_para[1]))
        ]

        self.markAs = []
        self.markBs = []
        self.markA_objs = []
        self.markB_objs = []

        self.r = 10
        self.B1_x = 0
        self.B1_y = 0
        self.B1_mark_info = None

        self.timeA = time.time()
        self.timeB = time.time()
        self.fpsA = 0
        self.fpsB = 0

        self.img_list = []
        self.img_list_p = [None for i in range(3)]

        # 画布
        self.canvasA = Canvas(self.tk)
        self.canvasB = Canvas(self.tk)

        # 标签
        self.label1 = Label(self.tk)
        self.label2 = Label(self.tk)
        self.label3 = Label(self.tk)

        # 滑动条
        self.scale_w = Scale(self.tk)
        self.scale_h = Scale(self.tk)
        self.scale1 = Scale(self.tk)
        self.scale2 = Scale(self.tk)
        self.scale3 = Scale(self.tk)

        self.setup()

    # 配置
    def setup(self):
        self.capA.set(3, self.frameA_size[0])
        self.capA.set(4, self.frameA_size[1])
        self.capB.set(3, self.frameB_size[0])
        self.capB.set(4, self.frameB_size[1])

        self.tk.title("头戴式眼动追踪系统")
        self.tk.bind('<KeyPress-space>', self.switch_suspend)
        self.tk.bind('<KeyPress-c>', self.switch_calibration)
        self.tk.bind('<KeyPress-f>', self.switch_fps)
        self.tk.bind('<KeyPress-1>', self.switch_gaze_method)
        self.tk.bind('<KeyPress-2>', self.switch_gaze_method)
        self.tk.bind('<KeyPress-3>', self.switch_gaze_method)

        # 画布
        self.canvasA.config(width=self.imgA_size[0], height=self.imgA_size[1])
        self.canvasB.config(width=self.imgB_size[0], height=self.imgB_size[1])

        # 标签
        self.label1.config()
        self.label2 = Label(self.tk)
        self.label3 = Label(self.tk)

        # 滑动条
        scale_length = 200
        self.scale_w.config(from_=0, to=0.49, value=self.eye_range_para[0], length=scale_length,
                            command=lambda event: self.get_scale(event, "w"))
        self.scale_h.config(from_=0, to=0.49, value=self.eye_range_para[1], length=scale_length,
                            command=lambda event: self.get_scale(event, "h"))
        self.scale1.config(from_=0, to=1, value=self.factor_threshold_ratio, length=scale_length,
                           command=lambda event: self.get_scale(event, "1"))
        self.scale2.config(from_=0, to=20, value=self.factor_dilate, length=scale_length,
                           command=lambda event: self.get_scale(event, "2"))
        self.scale3.config(from_=0, to=20, value=self.factor_erode, length=scale_length,
                           command=lambda event: self.get_scale(event, "3"))

        # 布局
        rows = 0
        self.canvasA.grid(row=rows, column=0, columnspan=2)
        self.canvasB.grid(row=rows, column=2, columnspan=2)

        self.tk.update()

        return True

    # 运行
    def run(self):
        self.tk.after(1, self.main)
        self.tk.mainloop()
        return True

    def main(self):
        if not self.suspend_flag:
            ret, frameA = self.capA.read()
            ret, frameB = self.capB.read()
            self.frameA = cv2.flip(frameA, 0)
            self.frameB = cv2.flip(frameB, -1)

            pupil_img = self.frameA[self.img_range[1]: self.img_range[3], self.img_range[0]: self.img_range[2]]

            pupil_img = cv2.cvtColor(pupil_img, cv2.COLOR_RGB2GRAY)

            if self.calibration_flag:
                self.img_list = []
                center = detect_pupil(
                    pupil_img,
                    factor_threshold_ratio=self.factor_threshold_ratio,
                    factor_dilate=self.factor_dilate,
                    factor_erode=self.factor_erode,
                    img_list=self.img_list
                )
                self.display_img(1)
                self.display_img(2)
                self.display_img(3)
            else:
                center = detect_pupil(
                    pupil_img,
                    factor_threshold_ratio=self.factor_threshold_ratio,
                    factor_dilate=self.factor_dilate,
                    factor_erode=self.factor_erode,
                )

            self.pupil_point = (center[0] + self.img_range[0], center[1] + self.img_range[1])

            timeA = time.time()
            self.fpsA = round(1 / (timeA - self.timeA))
            self.timeA = timeA

        self.display_frame('A')

        if not self.suspend_flag:
            if self.ret_mapping and (self.gaze_method_flag == 1 or self.gaze_method_flag == 3):
                self.gaze_point = self.mapping_matrix[self.pupil_point[0]][self.pupil_point[1]]
            else:
                self.ret_mapping, self.mapping_matrix = load_mapping_matrix()

            if self.ret_para and (self.gaze_method_flag == 2 or self.gaze_method_flag == 3):
                self.gaze_point_p = compute_mapping_gaze(self.pupil_point, self.parameter)
            else:
                self.ret_para, self.parameter = load_mapping_parameter()

            timeB = time.time()
            self.fpsB = round(1 / (timeB - self.timeB))
            self.timeB = timeB

        self.display_frame('B')

        self.tk.after(1, self.main)

    # 坐标转换
    @staticmethod
    def transform_coordinate(point, shapeA, shapeB):
        res = [
            round(point[0] * shapeB[0] / shapeA[0]),
            round(point[1] * shapeB[1] / shapeA[1])
        ]
        for i in range(len(shapeB)):
            if res[i] < 0:
                res[i] = 0
            elif res[i] >= shapeB[i]:
                res[i] = shapeB[i] - 1
        return res

    # 画图像
    @staticmethod
    def handle_img(img, size=None, cvtcolor=True):
        if size is not None:
            img = cv2.resize(img, size)
        if cvtcolor:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(np.uint8(img))
        img = ImageTk.PhotoImage(img)
        return img  # 保存img以防垃圾回收

    # 画标记
    @staticmethod
    def draw_marks(canvas, marks, r):
        res = []
        for i in range(len(marks)):
            x, y = marks[i]
            oval = canvas.create_oval(x - r, y - r, x + r, y + r, fill="black")
            text = canvas.create_text(x, y, text=str(i + 1), fill="white")
            res.append([oval, text])
        return res

    # 播放帧
    def display_frame(self, ABid):
        if ABid == 'A' or ABid == 0:
            frame = self.frameA
            canvas = self.canvasA
            img_size = self.imgA_size

            pupil_point = self.transform_coordinate(self.pupil_point, self.frameA_size, self.imgA_size)

            if self.calibration_flag:
                marks = self.markAs
                img_range = [
                    self.transform_coordinate((self.img_range[0], self.img_range[1]), self.frameA_size, self.imgA_size),
                    self.transform_coordinate((self.img_range[2], self.img_range[3]), self.frameA_size, self.imgA_size)
                ]

            if self.fps_flag:
                fps = self.fpsA
        elif ABid == 'B' or ABid == 1:
            frame = self.frameB
            canvas = self.canvasB
            img_size = self.imgB_size

            if self.ret_mapping and (self.gaze_method_flag == 1 or self.gaze_method_flag == 3):
                gaze_point = self.transform_coordinate(self.gaze_point, self.frameB_size, self.imgB_size)
            if self.ret_para and (self.gaze_method_flag == 2 or self.gaze_method_flag == 3):
                gaze_point_p = self.transform_coordinate(self.gaze_point_p, self.frameB_size, self.imgB_size)

            if self.calibration_flag:
                marks = self.markBs

            if self.fps_flag:
                fps = self.fpsB
        else:
            return False

        r = self.r
        color = "lime"
        color = "red"
        k = 0.25

        canvas.delete(ALL)

        img = self.handle_img(frame, img_size)
        img_obj = canvas.create_image(0, 0, anchor=NW, image=img)

        if self.calibration_flag:
            marks_objs = self.draw_marks(canvas, marks, r)

        if self.fps_flag:
            fps_obj = canvas.create_text(img_size[0] - 10, 10, text=str(fps), fill="lime")

        if ABid == 'A' or ABid == 0:
            canvas.create_oval(pupil_point[0] - r / 2, pupil_point[1] - r / 2,
                               pupil_point[0] + r / 2, pupil_point[1] + r / 2,
                               fill=color, width=0)

            self.imgA = img
            if self.calibration_flag:
                canvas.create_rectangle(img_range[0][0], img_range[0][1],
                                        img_range[1][0], img_range[1][1],
                                        outline="white", width=2)

                self.markA_objs = marks_objs
        elif ABid == 'B' or ABid == 1:
            if self.ret_mapping and (self.gaze_method_flag == 1 or self.gaze_method_flag == 3):
                canvas.create_line(gaze_point[0] - 2 * r, gaze_point[1],
                                   gaze_point[0] - r, gaze_point[1],
                                   fill=color, width=k * r)
                canvas.create_line(gaze_point[0], gaze_point[1] - 2 * r,
                                   gaze_point[0], gaze_point[1] - r,
                                   fill=color, width=k * r)
                canvas.create_line(gaze_point[0] + 2 * r, gaze_point[1],
                                   gaze_point[0] + r, gaze_point[1],
                                   fill=color, width=k * r)
                canvas.create_line(gaze_point[0], gaze_point[1] + 2 * r,
                                   gaze_point[0], gaze_point[1] + r,
                                   fill=color, width=k * r)
            if self.ret_para and (self.gaze_method_flag == 2 or self.gaze_method_flag == 3):
                canvas.create_line(gaze_point_p[0] - 2 * r, gaze_point_p[1],
                                   gaze_point_p[0] - r, gaze_point_p[1],
                                   fill="blue", width=k * r)
                canvas.create_line(gaze_point_p[0], gaze_point_p[1] - 2 * r,
                                   gaze_point_p[0], gaze_point_p[1] - r,
                                   fill="blue", width=k * r)
                canvas.create_line(gaze_point_p[0] + 2 * r, gaze_point_p[1],
                                   gaze_point_p[0] + r, gaze_point_p[1],
                                   fill="blue", width=k * r)
                canvas.create_line(gaze_point_p[0], gaze_point_p[1] + 2 * r,
                                   gaze_point_p[0], gaze_point_p[1] + r,
                                   fill="blue", width=k * r)

            self.imgB = img
            if self.calibration_flag:
                self.markB_objs = marks_objs

        self.tk.update()

    # 计算参数并保存
    def compute_and_save(self, event, pupils, gazes):
        for i in range(len(pupils)):
            pupils[i] = self.transform_coordinate(pupils[i], self.imgA_size, self.frameA_size)
        for i in range(len(gazes)):
            gazes[i] = self.transform_coordinate(gazes[i], self.imgB_size, self.frameB_size)

        self.ret_mapping, self.mapping_matrix \
            = compute_mapping_matrix(self.frameA_size, self.frameB_size, pupils, gazes)
        if self.ret_mapping:
            ret = save_mapping_matrix(self.mapping_matrix)

        self.ret_para, self.parameter = compute_mapping_parameter(pupils, gazes)
        if self.ret_para:
            ret = save_mapping_parameter(self.parameter)

    # 相机停止采样
    def switch_suspend(self, event):
        self.suspend_flag = not self.suspend_flag

    # 标定模式
    def switch_calibration(self, event):
        self.calibration_flag = not self.calibration_flag

        if self.calibration_flag:
            self.tk.bind("<KeyPress-Return>",
                         lambda event: self.compute_and_save(event, self.markAs.copy(), self.markBs.copy()))

            self.canvasA.bind("<ButtonPress-1>", lambda event: self.B1_press(event, 'A'))
            self.canvasA.bind("<ButtonRelease-1>", lambda event: self.B1_release(event, 'A'))
            self.canvasA.bind("<B1-Motion>", lambda event: self.B1_motion(event, 'A'))

            self.canvasB.bind("<ButtonPress-1>", lambda event: self.B1_press(event, 'B'))
            self.canvasB.bind("<ButtonRelease-1>", lambda event: self.B1_release(event, 'B'))
            self.canvasB.bind("<B1-Motion>", lambda event: self.B1_motion(event, 'B'))

            rows = 1
            self.scale_w.grid(row=rows, column=0)
            self.label1.grid(row=rows, column=1, rowspan=2)
            self.label2.grid(row=rows, column=2, rowspan=2)
            self.label3.grid(row=rows, column=3, rowspan=2)
            rows += 1
            self.scale_h.grid(row=rows, column=0)
            rows += 1
            self.scale1.grid(row=rows, column=1)
            self.scale2.grid(row=rows, column=2)
            self.scale3.grid(row=rows, column=3)
        else:
            self.tk.unbind("<KeyPress-Return>")

            self.canvasA.unbind("<ButtonPress-1>")
            self.canvasA.unbind("<ButtonRelease-1>")
            self.canvasA.unbind("<B1-Motion>")

            self.canvasB.unbind("<ButtonPress-1>")
            self.canvasB.unbind("<ButtonRelease-1>")
            self.canvasB.unbind("<B1-Motion>")

            self.scale_w.grid_forget()
            self.label1.grid_forget()
            self.label2.grid_forget()
            self.label3.grid_forget()
            self.scale_h.grid_forget()
            self.scale1.grid_forget()
            self.scale2.grid_forget()
            self.scale3.grid_forget()

    # 开关帧率显示
    def switch_fps(self, event):
        self.fps_flag = not self.fps_flag

    # 选择视线估计方法
    def switch_gaze_method(self, event):
        self.gaze_method_flag = int(event.keysym)

    # 鼠标在画布上按下
    def B1_press(self, event, ABid):
        if ABid == 'A' or ABid == 0:
            marks = self.markAs
        elif ABid == 'B' or ABid == 1:
            marks = self.markBs
        else:
            return False

        self.B1_x, self.B1_y = event.x, event.y
        r = self.r

        for i in range(len(marks)):
            x, y = marks[i]
            if x - r <= event.x <= x + r and y - r <= event.y <= y + r:
                self.B1_mark_info = (i, (x - event.x, y - event.y))
                break

    # 鼠标在画布上抬起
    def B1_release(self, event, ABid):
        if ABid == 'A' or ABid == 0:
            marks = self.markAs
        elif ABid == 'B' or ABid == 1:
            marks = self.markBs
        else:
            return False

        if event.x == self.B1_x and event.y == self.B1_y and len(marks) < 9:
            marks.append([event.x, event.y])

        self.B1_mark_info = None

    # 鼠标在画布上按住移动
    def B1_motion(self, event, ABid):
        if self.B1_mark_info is None:
            return False

        if ABid == 'A' or ABid == 0:
            marks = self.markAs
        elif ABid == 'B' or ABid == 1:
            marks = self.markBs
        else:
            return False

        index = self.B1_mark_info[0]
        offset = self.B1_mark_info[1]
        marks[index] = [event.x + offset[0], event.y + offset[1]]
        self.tk.update()

    def get_scale(self, event, id):
        if id == "w":
            self.eye_range_para[0] = round(float(event), 2)
            self.img_range = [
                int(self.frameA_size[0] * self.eye_range_para[0]),
                int(self.frameA_size[1] * self.eye_range_para[1]),
                int(self.frameA_size[0] * (1 - self.eye_range_para[0])),
                int(self.frameA_size[1] * (1 - self.eye_range_para[1]))
            ]
        elif id == "h":
            self.eye_range_para[1] = round(float(event), 2)
            self.img_range = [
                int(self.frameA_size[0] * self.eye_range_para[0]),
                int(self.frameA_size[1] * self.eye_range_para[1]),
                int(self.frameA_size[0] * (1 - self.eye_range_para[0])),
                int(self.frameA_size[1] * (1 - self.eye_range_para[1]))
            ]
        elif id == "1":
            self.factor_threshold_ratio = round(float(event), 2)
        elif id == "2":
            self.factor_dilate = round(float(event))
        elif id == "3":
            self.factor_erode = round(float(event))
        else:
            return False
        return True

    def display_img(self, id):
        if id == 1:
            label = self.label1
        elif id == 2:
            label = self.label2
        elif id == 3:
            label = self.label3
        else:
            return False

        img = self.img_list[id - 1]

        img = self.handle_img(img, cvtcolor=False)
        label.config(image=img)

        self.img_list_p[id - 1] = img

        return True

