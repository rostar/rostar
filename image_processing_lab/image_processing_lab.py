'''
Created on Feb 4, 2015

@author: rostislavrypl
'''
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.transform import hough_circle
from skimage.feature import peak_local_max, canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance
import matplotlib.image as mpimg
from skimage.io import imread, use_plugin
from skimage.filter import roberts, sobel
from skimage.filter import threshold_otsu, threshold_adaptive
from skimage.restoration import denoise_bilateral
from etsproxy.traits.api import HasTraits, Range, Float, on_trait_change, Str, Event, \
                                Property, cached_property, on_trait_change, Instance, \
                                Bool, Button
from util.traits.editors.mpl_figure_editor import MPLFigureEditor
from matplotlib.figure import Figure
from etsproxy.traits.ui.api import Item, View, Group, HSplit, VGroup, HGroup, Tabbed

class ImageProcessing(HasTraits):
    
    def __init__(self, **kw):
        super(ImageProcessing, self).__init__(**kw)
        self.on_trait_change(self.refresh, '+params')
        self.refresh()
    
    image_path = Str
    
    def rgb2gray(self, rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

    gray_cropped_image = Property(depends_on='image_path')
    @cached_property
    def _get_gray_cropped_image(self):
        image = mpimg.imread(self.image_path)
        image = self.rgb2gray(image)[100:1000,200:1200]
        return image
    
    filter = Bool(False, params=True)
    block_size = Range(1,100, params=True)
    offset = Range(1, 20, params=True)
    denoise = Bool(False, params=True)
    denoise_spatial = Range(1, 100, params=True)

    processed_image = Property
    def _get_processed_image(self):
        image = self.gray_cropped_image
        if self.filter == True:
            image = denoise_bilateral(image, sigma_spatial=self.denoise_spatial)
        if self.denoise == True:
            image = threshold_adaptive(image, self.block_size, offset=self.offset)
        return image

    figure = Instance(Figure)
    def _figure_default(self):
        figure = Figure(facecolor='white')
        return figure

    figure_edges = Instance(Figure)
    def _figure_edges_default(self):
        figure = Figure(facecolor='white')
        return figure
    
    figure_circles = Instance(Figure)
    def _figure_circles_default(self):
        figure = Figure(facecolor='white')
        return figure
    
    data_changed = Event

    def plot(self, fig, fig2):
        figure = fig
        figure.clear()
        axes = figure.gca()
        axes.imshow(self.processed_image, plt.gray())

    edges = Property
    def _get_edges(self):
        image = roberts(self.processed_image)
        image = image > 0.0
        return image

    hough_circles = Property
    def _get_hough_circles(self):
        hough_radii = np.arange(80, 120, 2)
        hough_res = hough_circle(self.edges, hough_radii)
        centers = []
        accums = []
        radii = []
        for radius, h in zip(hough_radii, hough_res):
            # For each radius, extract two circles
            num_peaks = 2
            peaks = peak_local_max(h, num_peaks=num_peaks)
            centers.extend(peaks)
            print peaks
            accums.extend(h[peaks[:, 0], peaks[:, 1]])
            radii.extend([radius] * num_peaks)
        
        image = self.processed_image
        for idx in np.argsort(accums)[::-1]:
            center_x, center_y = centers[idx]
            radius = radii[idx]
            cx, cy = circle_perimeter(center_y, center_x, radius)
            mask = (cx<image.shape[0]) * (cy<image.shape[1]) 
            image[cy[mask], cx[mask]] = 0.
        return image
    
    eval_edges = Button
    def _eval_edges_fired(self):
        edges = self.figure_edges
        edges.clear()
        axes_edges = edges.gca()
        axes_edges.imshow(self.edges, plt.gray())
        self.data_changed = True

    eval_circles = Button
    def _eval_circles_fired(self):
        circles = self.figure_circles
        circles.clear()
        axes_circles = circles.gca()
        axes_circles.imshow(self.hough_circles, plt.gray())
        self.data_changed = True

    def refresh(self):
        self.plot(self.figure, self.figure_edges)
        self.data_changed = True

    traits_view = View(HGroup(Group(Item('filter', label='filter'),
                                    Item('block_size'),
                                    Item('offset'),
                                    Item('denoise', label='denoise'),
                                    Item('denoise_spatial'),
                                    ),
                              Group(Item('figure',
                                        editor=MPLFigureEditor(),
                                        show_label=False,
                                        resizable=True),
                                        scrollable=True,
                                        label='Plot',
                                    ),
                              ),
                       Tabbed(VGroup(Item('eval_edges'),
                                     Item('figure_edges',
                                    editor=MPLFigureEditor(),
                                    show_label=False,
                                    resizable=True),
                                    scrollable=True,
                                    label='Plot_edges'),
                              ),
                       Tabbed(VGroup(Item('eval_circles'),
                                     Item('figure_circles',
                                    editor=MPLFigureEditor(),
                                    show_label=False,
                                    resizable=True),
                                    scrollable=True,
                                    label='Plot_circles'),
                              ),
                            id='imview',
                            dock='tab',
                            title='Image processing',
                            scrollable=True,
                            resizable=True,
                            width=600, height=400
                        )

if __name__ == '__main__':
    ip = ImageProcessing(image_path='specimen.jpg')
    ip.configure_traits()
