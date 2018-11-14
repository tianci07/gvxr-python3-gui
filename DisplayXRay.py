import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

import numpy as np

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# implement the default mpl key bindings
from matplotlib.backend_bases import key_press_handler


from matplotlib.figure import Figure

from numpy import arange, sin, pi

from gvxrGUI import x_ray_image


class DisplayXRay:
    def __init__(self, root):

        self.root = Tk.Tk()
        #self.root.wm_title("X-ray Image")

        self.x_ray_image = 0;

        self.has_been_drawn = False;

    def draw(self, x_ray_image):
        self.x_ray_image = np.copy(x_ray_image);

        #self.x_ray_image = np.multiply(np.divide(np.subtract(x_ray_image, self.x_ray_image.min()),
        #        self.x_ray_image.max() - self.x_ray_image.min()), 255);
        #if self.has_been_drawn:
        #    self.fig.clear();


        if not self.has_been_drawn:
            self.fig = Figure(figsize=(5, 8), dpi=100)
            ax = self.fig.add_subplot(211)
            self.im_plot = ax.imshow(self.x_ray_image, cmap="gray");
            self.fig.colorbar(self.im_plot);
            ax.set_title("X-ray image");

            ax = self.fig.add_subplot(212)
            n, bins, patches =  ax.hist(self.x_ray_image.ravel(), bins=256, density=True, facecolor='g', alpha=0.75)
            ax.set_yscale("log")
            ax.set_title("Intensity histogram")
            ax.set_xlabel("Intensity")
            ax.set_ylabel("Frequency")

            #ax.title("Intensity histogram of X-ray image");

            # a tk.DrawingArea
            canvas = FigureCanvasTkAgg(self.fig, master=self.root)
            canvas.draw()
            canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

            toolbar = NavigationToolbar2Tk(canvas, self.root)
            toolbar.update()
            canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

            canvas.mpl_connect('key_press_event', self.on_key_event)

            button = Tk.Button(master=self.root, text='Quit', command=self._quit)
            button.pack(side=Tk.BOTTOM)

            self.has_been_drawn = True;
        #else:

            #self.im_plot.set_data(self.x_ray_image)
            #self.fig.canvas.draw()
            #self.fig.canvas.flush_events()


    def on_key_event(self, event):
        print('you pressed %s' % event.key)
        key_press_handler(event, canvas, toolbar)

    def _quit(self):
        self.root.quit()     # stops mainloop
        self.root.destroy()  # this is necessary on Windows to prevent
