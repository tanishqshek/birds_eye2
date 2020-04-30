from __future__ import division

import multiprocessing
import os
from itertools import repeat

import kmeans_colours as km
from multiprocessing import Pool
import pickle as pkl
import random
import time
from darknet import Darknet
from preprocess import letterbox_image
from util import *
import tkinter as tk
from tkinter import font
from tkinter import ttk
from tkinter import filedialog
from tkinter import scrolledtext
from PIL import ImageTk, Image
from ttkthemes import ThemedStyle
from os import listdir

# GLOBAL CONSTANTS
NUM_CLASSES = 23
OUTPUT_IMAGES_PATH = '.\\crop'
OUTPUT_TEXT_PATH = '.\\output_text'
CONFIDENCE_THRESH = 0.5
NMS_THRESH = 0.4
CFG_FILE = '.\\cfg\\yolov3-tiny-obj.cfg'
WEIGHTS_FILE = '.\\weights\\yolov3-tiny-obj_best.weights'
RESO = '960'


# UI SECTION START

class BirdEye(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.style = ThemedStyle(self)
        self.style.set_theme('breeze')

        # Global data store
        self.video_path = tk.StringVar()
        self.object_interest = tk.StringVar()
        self.feature_interest = tk.StringVar()
        self.color_choice = tk.StringVar()

        # Basic UI element
        self.title_font = font.Font(family='Helvetica', size=36, weight="bold", slant="italic")
        self.title("Bird's Eye: Analysis of video surveillance data")
        self.geometry("640x640")
        self.resizable(width=True, height=True)

        # print("SampleApp: self={}".format(self))
        # the container is where we'll stack a bunch of frames
        # on top of each other, then the one we want visible
        # will be raised above the others
        container = ttk.Frame(self, borderwidth=1)
        container.pack(fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        container.update()
        print("container: width = {}, height = {}".format(container.winfo_width(), container.winfo_height()))
        self.frames = {}
        for F in (MainPage, FileUploadPage, FeatureSelectPage):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame

            # put all of the pages in the same location;
            # the one on the top of the stacking order
            # will be the one that is visible.
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame("MainPage")

    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        print(page_name)
        frame = self.frames[page_name]
        frame.tkraise()

    def send_input_data(self):
        # If user decides to run the demo, flush the output area first
        self.frames['FeatureSelectPage'].output_text.delete(1.0, tk.END)

        # Write all the basic instructions
        instructions = "Press 'q' to stop detection.\nVideo file: {}\nObject of interest: {}\nFeature of interest: {}" \
                       "\nColor of interest: {}\n\n".format(self.video_path.get(), self.object_interest.get(),
                                                            self.feature_interest.get(), self.color_choice.get())

        self.frames['FeatureSelectPage'].output_text.insert(tk.END, instructions)

        input_data = {
            'video': self.video_path.get(),
            'object': self.object_interest.get(),
            'feature': self.feature_interest.get(),
            'color': self.color_choice.get()
        }
        run_video_demo(input_data, self)

    def write_output_data(self, output_data):
        print('write_output_data entered!')
        out_file = open('.\\{}\\{}_{}.txt'.format(OUTPUT_TEXT_PATH, self.object_interest.get(), round(time.time(), 2)), 'w')
        out_file.write("Video file: {}\nObject of interest: {}\nFeature of interest: {}" \
                       "\nColor of interest: {}\n\n".format(self.video_path.get(), self.object_interest.get(),
                                                            self.feature_interest.get(), self.color_choice.get()))
        for key in output_data:
            out_file.write(output_data[key])
            self.frames['FeatureSelectPage'].output_text.insert(tk.END, output_data[key])


class MainPage(ttk.Frame):
    def __init__(self, parent, controller):
        # Initialization
        ttk.Frame.__init__(self, parent)
        self.controller = controller
        self.update()
        print("MainPage: width = {}, height = {}".format(self.winfo_width(), self.winfo_height()))
        print('MainPage: self={}, parent={}, controller={}'.format(self, parent, controller))

        # Drawing title frame and logo
        self.title_container = ttk.LabelFrame(self)
        self.title_container.grid(row=0, column=0, sticky='nwe')
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        app_description = "This application uses the YOLO algorithm to perform object detection.\n" \
                          "It also uses K-Means clustering algorithm for checking dominant colours in " \
                          "the detected object."

        self.app_name = ttk.Label(self.title_container, text="Bird's Eye - Analysis of Video Surveillance\n",
                                  font=('Times New Roman', 12, 'bold', 'italic'))
        self.app_name.grid(row=0, column=1, sticky='nwse', padx=20, pady=20)
        self.title_container.rowconfigure(0, weight=1)
        self.title_container.columnconfigure(1, weight=1)

        # Resize logo
        size = 96, 96
        img = Image.open('.\\ui_data\\logo.jpg')
        img.thumbnail(size)
        img.save('.\\ui_data\\logo_thumbnail.jpg', 'JPEG')

        self.logo = ImageTk.PhotoImage(Image.open('.\\ui_data\\logo_thumbnail.jpg'))
        self.logo_label = ttk.Label(self.title_container, image=self.logo)
        self.logo_label.grid(row=0, column=0, sticky='wns', padx=10, pady=10)
        self.title_container.columnconfigure(0, weight=1)

        upload_button = ttk.Button(self, text="Process input video",
                                   command=lambda: controller.show_frame("FileUploadPage"))
        upload_button.grid(row=1, column=0, sticky='s', padx=20, pady=20)
        self.rowconfigure(1, weight=1)

        quit_button = ttk.Button(self, text="Quit", command=controller.destroy)
        quit_button.grid(row=2, column=0)
        self.rowconfigure(2, weight=1)


class FileUploadPage(ttk.Frame):

    # Redraw some widgets after file is selected
    def redraw(self, controller):
        if self.file_selected:
            print("Update")
            self.change_file_button = ttk.Button(self, text="Change files",
                                                 command=lambda: self.browse_callback(controller))
            self.change_file_button.grid(row=1, column=0)
            self.upload_button.grid(row=2, column=0)
            self.back_button.grid(row=3, column=0)
            self.quit_button.grid(row=4, column=0)
            self.rowconfigure(4, weight=1)

            self.browse_message.config(text='Video path: ' + controller.video_path.get())
            self.upload_button.config(text="Select Parameters",
                                      command=lambda: controller.show_frame("FeatureSelectPage"))

    # fire the directory browse utility
    def browse_callback(self, controller):
        name = filedialog.askopenfilename(initialdir="/", title="Select file",
                                          filetypes=(("avi", "*.avi"), ("mp4", "*.mp4")))
        if name == '':
            self.browse_message.config(text='Please select a file before proceeding.\n'
                                            'Click on the "Browse Files" button to select video for object detection.')
        else:
            self.file_selected = True
            controller.video_path.set(name)
            print(controller.video_path)
            self.redraw(controller)

    def __init__(self, parent, controller):
        # Initialization
        ttk.Frame.__init__(self, parent)
        self.controller = controller
        self.file_selected = False
        self.change_file_button = None

        # draw widgets before file directory is selected
        self.browse_message = ttk.Label(self, text="Use the 'browse files' button to select the CCTV video file\n"
                                                   "on which object detection is to be performed", font=('Times New '
                                                                                                         'Roman', 12,
                                                                                                         'italic'))
        self.browse_message.grid(row=0, column=0)
        self.rowconfigure(0, weight=4)
        self.columnconfigure(0, weight=1)

        self.upload_button = ttk.Button(self, text="Browse files",
                                        command=lambda: self.browse_callback(controller))
        self.upload_button.grid(row=1, column=0)
        self.rowconfigure(1, weight=1)

        self.back_button = ttk.Button(self, text="Back", command=lambda: controller.show_frame("MainPage"))
        self.back_button.grid(row=2, column=0)
        self.rowconfigure(2, weight=1)

        self.quit_button = ttk.Button(self, text="Quit", command=controller.destroy)
        self.quit_button.grid(row=3, column=0)
        self.rowconfigure(3, weight=1)


class FeatureSelectPage(ttk.Frame):

    def write_to_output(self, data):
        self.output_text.insert(tk.END, data)

    def redraw(self, controller):
        print(self.object_flag, controller.object_interest.get() in self.feature_dict)
        if self.object_flag and controller.object_interest.get() in self.feature_dict:
            print('Drawing feature box')
            # Configuration for Feature combo box
            self.fc_label.grid(row=0, column=1)
            self.feature_combo.config(values=self.get_feature_list(controller),
                                      textvariable=controller.feature_interest)
            self.feature_combo.grid(row=1, column=1, padx=20, pady=20)
            self.localContainer.rowconfigure(1, weight=1)
            self.localContainer.columnconfigure(1, weight=1)
        elif self.object_flag and controller.object_interest.get() not in self.feature_dict:
            # Remove Feature combo box widget
            self.localContainer.rowconfigure(1, weight=0)
            self.localContainer.columnconfigure(1, weight=0)
            self.feature_combo.grid_forget()
            self.fc_label.grid_forget()

        # Configuration for timestamp text widget in the main frame
        self.ot_label.grid(row=1, column=0)
        self.output_text.grid(row=2, column=0, sticky=tk.E + tk.W + tk.S + tk.N, padx=20, pady=20)
        self.rowconfigure(2, weight=8)
        self.columnconfigure(0, weight=20)
        # self.scrollb.grid(row=0, column=1, sticky='nse')

        # Configuration for buttonContainer
        self.buttonContainer.grid(row=3, column=0, sticky=tk.E + tk.W + tk.S + tk.N, padx=10, pady=10)
        self.rowconfigure(3, weight=1)
        self.columnconfigure(0, weight=1)

        # Configuration for Back button
        self.back_button.grid(row=0, column=0, padx=5, pady=5)
        self.buttonContainer.rowconfigure(0, weight=1)
        self.buttonContainer.columnconfigure(0, weight=1)

        # Configuration for exit button in the main frame
        self.exit_button.grid(row=0, column=2, padx=5, pady=5)
        self.buttonContainer.rowconfigure(0, weight=1)
        self.buttonContainer.columnconfigure(2, weight=1)

        # Configuration for extract button in the main frame
        self.extract_button.grid(row=0, column=1, padx=5, pady=5)
        self.buttonContainer.rowconfigure(0, weight=1)
        self.buttonContainer.columnconfigure(1, weight=1)

    def get_feature_list(self, controller):
        print(controller.object_interest.get())
        temp = controller.object_interest.get()
        return self.feature_dict[temp]

    def get_object_list(self):
        return self.object_list

    def object_selected(self, event):
        print('object_selected')
        self.object_flag = True
        self.redraw(self.controller)

    def draw_colour_detect_cb(self):
        print(self.vcb.get())
        if self.vcb.get() == 1:
            # self.cc_label.grid(row=2, column=1, padx=10, pady=10)
            self.colour_combo.grid(row=2, column=1, padx=10, pady=10)
            self.localContainer.rowconfigure(2, weight=1)
            # self.localContainer.rowconfigure(3, weight=1)
            self.localContainer.columnconfigure(1, weight=1)
        elif self.vcb.get() == 0:
            # self.cc_label.grid_forget()
            self.colour_combo.grid_forget()

    def __init__(self, parent, controller):
        # Initialization
        ttk.Frame.__init__(self, parent)
        self.controller = controller
        self.object_flag = False
        self.colour_list = ['Black', 'White', 'Red', 'Lime',
                            'Blue', 'Yellow', 'Cyan', 'Magenta',
                            'Silver', 'Gray', 'Maroon', 'Olive',
                            'Green', 'Purple', 'Teal', 'Navy']
        self.object_list = ['Person', 'Cat', 'Dog', 'Bag', 'Bicycle', 'Car', 'Bottle', 'Mobile Phone', 'Laptop',
                            'Scooter']
        self.feature_dict = {'Car': ['Sedan', 'Hatchback', 'Coupe', 'Jeep'],
                             'Cat': ['Egyptian Cat', 'Persian Cat', 'Tiger cat', 'Tabby cat'],
                             'Dog': ['Labrador Retriever', 'German Shepherd', 'Shih-tzu', 'Tibetan Mastiff'],
                             'Bag': ['Shopping Bag', 'Plastic Bag', 'Backpack', 'Suitcase']
                             }
        # Declaration for container of object and feature combo boxes
        self.localContainer = ttk.LabelFrame(self, text="Feature Extraction Parameters")

        # Declaration for container of all button widgets
        self.buttonContainer = ttk.LabelFrame(self)

        # Declaration of object combo box
        self.oc_label = ttk.Label(self.localContainer, text="Select Object:")
        self.object_combo = ttk.Combobox(self.localContainer, values=self.get_object_list(),
                                         textvariable=controller.object_interest)

        # Declaration of feature combo box
        self.fc_label = ttk.Label(self.localContainer, text="Select Feature:")
        self.feature_combo = ttk.Combobox(self.localContainer)

        # Declaration of Detected Object Timestamp TextArea
        self.ot_label = ttk.Label(self, text="Objects Detected")
        self.output_text = scrolledtext.ScrolledText(self, width=0, height=0, background="White", undo=True)

        # Declaration of Timestamp TextArea Scrollbar
        # self.scrollb = ttk.Scrollbar(self.output_text, command=self.output_text.yview)
        # self.output_text['yscrollcommand'] = self.scrollb.set

        # Declaration of Colour detection checkbox:
        # self.colour_detect_label = ttk.Label(self.localContainer, text="Perform colour detection on required object?")
        self.vcb = tk.IntVar(0)
        self.colour_detect_cb = ttk.Checkbutton(self.localContainer, text="Perform colour detection?",
                                                variable=self.vcb, command=lambda: self.draw_colour_detect_cb())

        # Declaration of Colour detection combo box
        # self.cc_label = ttk.Label(self.localContainer, text="Select colour:")
        self.colour_combo = ttk.Combobox(self.localContainer, values=self.colour_list,
                                         textvariable=controller.color_choice)

        # Declaration of Extract Button
        self.extract_button = ttk.Button(self.buttonContainer, text="Extract Features",
                                         command=lambda: controller.send_input_data())

        # Declaration of Back button
        self.back_button = ttk.Button(self.buttonContainer, text="Back",
                                      command=lambda: controller.show_frame("FileUploadPage"))

        # Declaration of Exit button
        self.exit_button = ttk.Button(self.buttonContainer, text="Exit", command=controller.destroy)

        # Configuration for container
        self.localContainer.grid(row=0, column=0, padx=10, pady=10)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        # Configuration for colour detect check button
        # self.colour_detect_label.grid(row=3, column=0, padx=10, pady=10)
        self.colour_detect_cb.grid(row=2, column=0, padx=10, pady=10)
        self.localContainer.rowconfigure(2, weight=1)

        # Configuration for object combo box
        self.oc_label.grid(row=0, column=0)
        self.object_combo.grid(row=1, column=0, padx=20, pady=20)
        self.object_combo.bind("<<ComboboxSelected>>", self.object_selected)
        self.localContainer.rowconfigure(1, weight=1)
        self.localContainer.columnconfigure(0, weight=1)


# UI SECTION END


class_to_object_mapping = {
    'Coupe': 'Car',
    'Sedan': 'Car',
    'Hatchback': 'Car',
    'Jeep': 'Car',
    'Shih-Tzu': 'Dog',
    'Siberian Huskey': 'Dog',
    'German Shephard': 'Dog',
    'Tibetan Mastiff': 'Dog',
    'Labrador Retriever': 'Dog',
    'Egyptian Cat': 'Cat',
    'Persian Cat': 'Cat',
    'Tiger cat': 'Cat',
    'Tabby cat': 'Cat',
    'Shopping Bag': 'Bag',
    'Plastic Bag': 'Bag',
    'Backpack': 'Bag',
    'Suitcase': 'Bag',
    'Laptop': 'Laptop',
    'Bottle': 'Bottle',
    'Bicycle': 'Bicycle',
    'Scooter': 'Scooter',
    'Mobile Phone': 'Mobile Phone',
    'Person': 'Person'
}


def get_test_input(input_dim, CUDA):
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (input_dim, input_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    img_ = img_[np.newaxis, :, :, :] / 255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)

    if CUDA:
        img_ = img_.cuda()

    return img_


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim


def write(x, img, classes, colors, frames, fps, timestamp, output_for_ui, args, cap):
    current_frame_time = round((cap.get(cv2.CAP_PROP_POS_MSEC) / 1000), 2)
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    image = img
    # print('Height and width:',image.shape[:2])
    crop_img = image[int(x[2]):int(x[4]), int(x[1]):int(x[3])]
    print('x: {}'.format(x))
    cls = int(x[-1])
    if not cls < 0:
        print('Class: {}'.format(cls))
        label = None
        try:
            label = "{0}".format(classes[cls])
            print('label: {}'.format(label))
        except IndexError:
            print('EOF')
        '''
        if frames % fps == 0:
            time = float(frames) / float(fps)
    
            """
    
            if label in object_no:
                object_no[label] = object_no[label]+1
            else:
                object_no[label] = 1
            cv2.imwrite(r'D:\LY Project\pytorch-yolo-v3\crop\{}_{}_{}.jpg'.format(label,object_no[label],time),crop_img)
            #print('Frame no:', frames)
            """
    
            if (time in timestamp):
                if label in timestamp[time]:
                    (timestamp[time])[label] = (timestamp[time])[label] + 1
                else:
                    (timestamp[time])[label] = 1
            else:
                timestamp[time] = {label: 1}
                # timestamp[time]
    
            print(timestamp)
            try:
                cv2.imwrite(r'{}\{}_{}_{}.jpg'.format(OUTPUT_IMAGES_PATH, label, timestamp[time][label], time), crop_img)
            except Exception:
                print("Couldn't write frame!")
        # else:
        # object_no = {}
        '''
        frame_captured = False

        # Check if predicted class is required by the user.
        print('Write function entered')
        print('Feature flag value: {}'.format(args['feature_flag']))
        print('Color flag value: {}'.format(args['color_flag']))
        print('Args: {}'.format(args))
        if args['feature_flag']:
            if label == args['feature']:
                try:
                    cv2.imwrite(r'{}\{}_{}.jpg'.format(OUTPUT_IMAGES_PATH, label, round(current_frame_time, 2)),
                                crop_img)
                    frame_captured = True
                except Exception:
                    print("Couldn't write frame!")
                print('Writing to UI')
                output_for_ui['{}_{}.jpg'.format(label,
                                                 current_frame_time)] = 'Desired object and feature combo {}/{} ' \
                                                                        'at time: {}s | ' \
                    .format(class_to_object_mapping[label], label,
                            round(current_frame_time, 2))
                # output_for_ui.append(('Desired object and feature combo {}/{} at time: {:5.2f}s\n\n'
                #                       .format(class_to_object_mapping[label], label,
                #                               current_frame_time)))

        elif not args['feature_flag']:
            if class_to_object_mapping[label] == args['object']:
                try:
                    cv2.imwrite(r'{}\{}_{}.jpg'.format(OUTPUT_IMAGES_PATH, label, round(current_frame_time, 2)),
                                crop_img)
                    frame_captured = True
                except Exception:
                    print("Couldn't write frame!")

                output_for_ui['{}_{}.jpg'.format(label,
                                                 round(current_frame_time, 2))] = 'Desired object {} ' \
                                                                                  'at time: {:5.2f}s | ' \
                    .format(class_to_object_mapping[label],
                            current_frame_time)
                # output_for_ui.append('Desired object {} at time: {:5.2f}s\n\n'
                # .format(class_to_object_mapping[label],
                # current_frame_time))
        color = random.choice(colors)
        cv2.rectangle(img, c1, c2, color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
        return img


'''
def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')

    parser.add_argument("--video", dest='video', help=
    "Video to run detection upon",
                        default="video.avi", type=str)
    parser.add_argument("--dataset", dest="dataset", help="Dataset on which the network has been trained",
                        default="pascal")
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help=
    "Config file",
                        default="cfg/yolov3-tiny-obj.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help=
    "weightsfile",
                        default="weights/yolov3-tiny-obj_best.weights", type=str)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="608", type=str)
    return parser.parse_args()
'''


def run_video_demo(input_data, UI):
    args = {'confidence': CONFIDENCE_THRESH,
            'cfgfile': CFG_FILE,
            'nms_thres': NMS_THRESH,
            'reso': RESO,
            'weights': WEIGHTS_FILE,
            'video': input_data['video'],
            'object': input_data['object'],
            'feature': input_data['feature'],
            'color': input_data['color'],
            'feature_flag': None,
            'color_flag': None}

    # Setting up parameter flags
    if args['feature'] == '':
        # No feature is provided by the user
        args['feature_flag'] = False
    else:
        args['feature_flag'] = True
    if args['color'] == '':
        # Color detection is not to be performed
        args['color_flag'] = False
    else:
        args['color_flag'] = True

    confidence = float(args['confidence'])
    nms_thesh = float(args['nms_thres'])
    start = 0

    CUDA = torch.cuda.is_available()

    num_classes = NUM_CLASSES

    bbox_attrs = 5 + num_classes

    print("Loading network.....")
    model = Darknet(args['cfgfile'])
    model.load_weights(args['weights'])
    print("Network successfully loaded")

    model.net_info["height"] = args['reso']
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    if CUDA:
        model.cuda()

    model(get_test_input(inp_dim, CUDA), CUDA)

    model.eval()

    output_for_ui = {}
    object_no = {}
    timestamp = {}

    videofile = args['video']

    cap = cv2.VideoCapture(videofile)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('FPS of original video:', fps)

    assert cap.isOpened(), 'Cannot capture source'

    FPS_STORE = []
    frames = 0
    start = time.time()
    while cap.isOpened():

        ret, frame = cap.read()
        if ret:

            img, orig_im, dim = prep_image(frame, inp_dim)

            im_dim = torch.FloatTensor(dim).repeat(1, 2)

            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()

            with torch.no_grad():
                output = model(Variable(img), CUDA)
            output = write_results(output, confidence, num_classes, nms=True, nms_conf=nms_thesh)

            if type(output) == int:
                frames += 1
                print("FPS of the video is {:5.2f}  Frame no: {}".format(frames / (time.time() - start), frames))
                cv2.imshow("frame", orig_im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue

            im_dim = im_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(inp_dim / im_dim, 1)[0].view(-1, 1)

            output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
            output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2

            output[:, 1:5] /= scaling_factor

            for i in range(output.shape[0]):
                output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
                output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

            classes = load_classes('data/obj.names')
            print(classes)
            colors = pkl.load(open("pallete", "rb"))

            # Routine to find if current frame has object of interest.

            list(map(lambda x: write(x, orig_im, classes, colors, frames, fps, timestamp, output_for_ui, args, cap),
                     output))

            cv2.imshow("frame", orig_im)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1
            print("FPS of the video is {}  Frame no: {}".format(round(frames / (time.time() - start), 2), frames))
            FPS_STORE.append(frames / (time.time() - start))
        else:
            break

    cap.release()

    output_for_ui['AVG_FPS'] = 'Average FPS of the program: {}d'.format(sum(FPS_STORE) / len(FPS_STORE))

    print(output_for_ui)
    # Color extraction routine
    if args['color_flag']:
        desired_color = args['color']
        for file in listdir('{}'.format(OUTPUT_IMAGES_PATH)):
            kmc = km.KMeansColours(img=file, clusters=5, desired_color=desired_color,
                                   file_dir='{}\\'.format(OUTPUT_IMAGES_PATH),
                                   file_dest='.\\crop_thumbnails\\')
            output_for_ui[file] = output_for_ui[file] + 'Desired color {} is present: {}\n\n'.format(desired_color,
                                                                                                     kmc.driver())

    elif not args['color_flag']:
        for file in listdir('{}'.format(OUTPUT_IMAGES_PATH)):
            output_for_ui[file] = output_for_ui[file] + '\n\n'

    list(map(os.unlink, (os.path.join(OUTPUT_IMAGES_PATH, f) for f in os.listdir(OUTPUT_IMAGES_PATH))))
    list(map(os.unlink, (os.path.join('.\\crop_thumbnails\\', f) for f in os.listdir('.\\crop_thumbnails'))))
    UI.write_output_data(output_for_ui)


if __name__ == "__main__":
    app = BirdEye()
    app.mainloop()
