'''
Created on Feb 4, 2015

@author: rostislavrypl
'''
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import hough_circle
from skimage.feature import peak_local_max, canny
from skimage.draw import circle_perimeter
import matplotlib.image as mpimg
from skimage.filters import roberts, sobel, threshold_adaptive
from skimage.restoration import denoise_bilateral
from etsproxy.traits.api import HasTraits, Range, Str,  Event, \
                                Property, Instance, \
                                Bool, Button, Enum, Int
from util.traits.editors.mpl_figure_editor import MPLFigureEditor
from matplotlib.figure import Figure
from etsproxy.traits.ui.api import Item, View, Group, VGroup, HGroup, Tabbed


class ImageProcessing(HasTraits):
    
    def __init__(self, **kw):
        super(ImageProcessing, self).__init__(**kw)
        self.on_trait_change(self.refresh, '+params')
        self.refresh()
    
    image_path = Str
    
    def rgb2gray(self, rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
    
    filter = Bool(False, params=True)
    block_size = Range(1,100, params=True)
    offset = Range(1, 20, params=True)
    denoise = Bool(False, params=True)
    denoise_spatial = Range(1, 100, params=True)

    processed_image = Property
    def _get_processed_image(self):
        # read image
        image = mpimg.imread(self.image_path)
        mask = image[:,:,1] > 150.
        image[mask] = 255.
        #plt.imshow(image)
        #plt.show()
        # convert to grayscale
        image = self.rgb2gray(image)
        # crop image
        image = image[100:1000,200:1100]
        mask = mask[100:1000,200:1100]
        image = image - np.min(image)
        image[mask] *= 255. / np.max(image[mask])
        if self.filter == True:
            image = denoise_bilateral(image, sigma_spatial=self.denoise_spatial)
        if self.denoise == True:
            image = threshold_adaptive(image, self.block_size, offset=self.offset)
        return image, mask

    edge_detection_method = Enum('canny',
                                 'sobel',
                                 'roberts', params = True)
    canny_sigma = Range(2.832, 5, params=True)
    canny_low = Range(5.92, 100, params=True)
    canny_high = Range(0.1, 100, params=True)

    edges = Property
    def _get_edges(self):
        img_edg, mask = self.processed_image
        if self.edge_detection_method == 'canny':
            img_edg = canny(img_edg, sigma=self.canny_sigma,
                        low_threshold=self.canny_low,
                        high_threshold=self.canny_high)
        elif self.edge_detection_method == 'roberts':
            img_edg = roberts(img_edg)
        elif self.edge_detection_method == 'sobel':
            img_edg = sobel(img_edg)
        img_edg = img_edg > 0.0
        return img_edg

    radii = Int(80, params=True)
    radius_low = Int(40, params=True)
    radius_high = Int(120, params=True)
    step = Int(2, params=True)

    hough_circles = Property
    def _get_hough_circles(self):
        hough_radii = np.arange(self.radius_low, self.radius_high, self.step)[::-1]
        hough_res = hough_circle(self.edges, hough_radii)
        centers = []
        accums = []
        radii = []        # For each radius, extract num_peaks circles
        num_peaks = 3
        for radius, h in zip(hough_radii, hough_res):
            peaks = peak_local_max(h, num_peaks=num_peaks)
            centers.extend(peaks)
            print 'circle centers = ', peaks
            accums.extend(h[peaks[:, 0], peaks[:, 1]])
            radii.extend([radius] * num_peaks)
                    
        im = mpimg.imread(self.image_path)
        # crop image
        im = im[100:1000,200:1100]
        for idx in np.arange(len(centers)):
            center_x, center_y = centers[idx]
            radius = radii[idx]
            cx, cy = circle_perimeter(center_y, center_x, radius)
            mask = (cx<im.shape[0]) * (cy<im.shape[1]) 
            im[cy[mask], cx[mask]] = (220.,20.,20.)
        return im
    
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
        img, mask = self.processed_image
        axes.imshow(img, plt.gray())

    def refresh(self):
        self.plot(self.figure, self.figure_edges)
        self.data_changed = True

    traits_view = View(HGroup(Group(Item('filter', label='filter'),
                                    Item('block_size'),
                                    Item('offset'),
                                    Item('denoise', label='denoise'),
                                    Item('denoise_spatial'),
                                    label='Filters'),
                              Group(Item('figure',
                                        editor=MPLFigureEditor(),
                                        show_label=False,
                                        resizable=True),
                                        scrollable=True,
                                        label='Plot',
                                    ),
                              ),
                       Tabbed(VGroup(Item('edge_detection_method'),
                                     Item('canny_sigma'),
                                     Item('canny_low'),
                                     Item('canny_high'),
                                     Item('eval_edges', label='Evaluate'),
                                     Item('figure_edges',
                                    editor=MPLFigureEditor(),
                                    show_label=False,
                                    resizable=True),
                                    scrollable=True,
                                    label='Plot_edges'),
                              ),
                       Tabbed(VGroup(Item('radii'),
                                     Item('radius_low'),
                                     Item('radius_high'),
                                     Item('step'),
                                     Item('eval_circles'),
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
    #ip.eval_circles = True
    ip.configure_traits()
