# -*- coding: utf-8 -*-
# Dioptas - GUI program for fast processing of 2D X-ray diffraction data
# Principal author: Clemens Prescher (clemens.prescher@gmail.com)
# Copyright (C) 2014-2019 GSECARS, University of Chicago, USA
# Copyright (C) 2015-2018 Institute for Geology and Mineralogy, University of Cologne, Germany
# Copyright (C) 2019 DESY, Hamburg, Germany
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import logging
import os
from past.builtins import basestring

import numpy as np
from PIL import Image
from qtpy import QtCore

import h5py
import re

import fabio

from .util.spe import SpeFile
from .util.NewFileWatcher import NewFileInDirectoryWatcher
from .util.HelperModule import rotate_matrix_p90, rotate_matrix_m90, FileNameIterator
from .util.ImgCorrection import ImgCorrectionManager, ImgCorrectionInterface, TransferFunctionCorrection

logger = logging.getLogger(__name__)


class ImgModel(QtCore.QObject):
    """
    Main Image handling class. Supports several features:
        - loading image files in any format using fabio
        - iterating through files either by file number or time of creation
        - image transformations like rotating and flipping
        - setting a background image
        - setting an absorption correction (img_data is divided by this)
        - using supersampling (splitting each pixel into n**2 pixel with equal intensity)

    In order to subscribe to changes of the data in the ImgModel, please use the img_changed QtSignal.
    The Signal will be called every time the img_data has changed.
    """
    img_changed = QtCore.Signal()
    autoprocess_changed = QtCore.Signal()
    transformations_changed = QtCore.Signal()
    corrections_removed = QtCore.Signal()

    def __init__(self):
        super(ImgModel, self).__init__()
        self.filename = ''
        self.img_transformations = []
        self.supersampling_factor = 1

        self.file_iteration_mode = 'number'
        self.file_name_iterator = FileNameIterator()

        self._img_data = None
        self._img_data_background_subtracted = None
        self._img_data_absorption_corrected = None
        self._img_data_background_subtracted_absorption_corrected = None

        self._img_data_supersampled = None
        self._img_data_supersampled_background_subtracted = None
        self._img_data_supersampled_absorption_corrected = None
        self._img_data_supersampled_background_subtracted_absorption_corrected = None

        self.background_filename = ''
        self._background_data = None
        self._background_scaling = 1
        self._background_offset = 0

        self._factor = 1

        self.transfer_correction = TransferFunctionCorrection()

        self.file_info = ''
        self.motors_info = {}
        self._img_corrections = ImgCorrectionManager()

        self._img_data = np.zeros((2048, 2048))

        # setting up autoprocess
        self._autoprocess = False
        self._directory_watcher = NewFileInDirectoryWatcher(
            file_types=['.img', '.sfrm', '.dm3', '.edf', '.xml',
                        '.cbf', '.kccd', '.msk', '.spr', '.tif',
                        '.mccd', '.mar3450', '.pnm', 'spe']
        )
        self._directory_watcher.file_added.connect(self.load)
        self.series_max = 0
        self.hdf_imgs = []
        self.series_pos = 0

    def load(self, filename):
        """
        Loads an image file in any format known by fabIO. Automatically performs all previous img transformations,
        performs supersampling and recalculates background subtracted and absorption corrected image data. The
        img_changed signal will be emitted after the process.
        :param filename: path of the image file to be loaded
        """
        filename = str(filename)  # since it could also be QString
        logger.info("Loading {0}.".format(filename))
        self.filename = filename
        try:
            im = Image.open(filename)
            if im.format == "HDF5":  # Pillow can't read nexus files properly so do it with nexusformat instead
                self.loadHDF5(filename)
            else:
                self._img_data = np.array(im)[::-1]
                self.series_max = 0
                self.file_info = self._get_file_info(im)
                self.motors_info = self._get_motors_info(im)
            im.close()
        except IOError:
            if os.path.splitext(filename)[1].lower() == '.spe':
                spe = SpeFile(filename)
                self._img_data = spe.img
                self.series_max = 0
            else:
                self._img_data_fabio = fabio.open(filename)
                self._img_data = self._img_data_fabio.data[::-1]
                self.series_max = 0
        self.file_name_iterator.update_filename(filename)
        self._directory_watcher.path = os.path.dirname(str(filename))

        self._perform_img_transformations()
        self._calculate_img_data()

        self.img_changed.emit()

    def loadHDF5(self, filename):
        nx_file = h5py.File(filename, "r")
        detector_identifiers = [["/entry/instrument/detector/description", "Lambda"],["/entry/instrument/detector/description", b"Lambda"]]
        multifile = True
        filenumber_list = [1, 2, 3]
        regex_in = r"(.+_m)\d(.nxs)"
        regex_out = r"\g<1>{}\g<2>"
        data_path = "entry/instrument/detector/data"
        if nx_file[detector_identifiers[0][0]][0] == detector_identifiers[0][1]:
            if multifile:
                # the image data is spread over multiple files, so we compile a list of them here
                self.hdf_imgs = [
                    h5py.File(re.sub(regex_in, regex_out.format(moduleIndex), filename), "r")  # TODO: catch exception if not all modules are there
                    for moduleIndex in filenumber_list]
                self.nximdata = [
                    imageFile[data_path]
                    for imageFile in self.hdf_imgs
                ]
                module_positions_key = "/entry/instrument/detector/translation/distance"
                self.module_positions = np.array([nxim[module_positions_key][0].astype(int) for nxim in self.hdf_imgs])
                np.subtract(self.module_positions, self.module_positions[:, 0].min(), self.module_positions, where=[1, 0, 0])
                np.subtract(self.module_positions, self.module_positions[0][1], self.module_positions, where=[0, 1, 0])
                self.loadImageFromNxim(0)
                self.series_max = self.hdf_imgs[0][data_path].shape[0] - 1
                self.series_pos = 0
            else:
                pass  # TODO: single image

        if nx_file[detector_identifiers[1][0]][0] == detector_identifiers[1][1]:
            if multifile:
                # the image data is spread over multiple files, so we compile a list of them here
                self.hdf_imgs = [
                    h5py.File(re.sub(regex_in, regex_out.format(moduleIndex), filename), "r")  # TODO: catch exception if not all modules are there
                    for moduleIndex in filenumber_list]
                self.nximdata = [
                    imageFile[data_path]
                    for imageFile in self.hdf_imgs
                ]
                module_positions_key = "/entry/instrument/detector/translation/distance"
                self.module_positions = np.array([np.array(nxim[module_positions_key]).astype(int) for nxim in self.hdf_imgs])
                np.subtract(self.module_positions, self.module_positions[:, 0].min(), self.module_positions, where=[1, 0, 0])
                np.subtract(self.module_positions, self.module_positions[0][1], self.module_positions, where=[0, 1, 0])
                self.loadImageFromNxim(0)
                self.series_max = self.hdf_imgs[0][data_path].shape[0] - 1
                self.series_pos = 0
            else:
                pass  # TODO: single image

    def loadImageFromNxim(self, image_nr):
        # remove any empty columns/rows to the left or top of the image data or shift any negative rows/columns into the positive
        nximages = np.empty((0, self.nximdata[2].shape[2] + self.module_positions[:, 0].max()))
        # the empty array needs to have the width of the detector data for concatenate()
        for modulenr, moduleImageData in enumerate(self.nximdata):
            # generate empty columns to the left and right of the data to match with the others
            imagedata = np.concatenate([np.zeros((moduleImageData.shape[1], self.module_positions[modulenr, 0])),
                moduleImageData[image_nr],
                np.zeros((moduleImageData.shape[1], self.module_positions[:,0].max()-self.module_positions[modulenr, 0]))], axis=1)
            nximages = np.concatenate(
                [nximages,
                 np.zeros((
                     # generate as many empty rows as needed to get to the position where the module data wants to be
                     int(self.module_positions[modulenr, 1]) -
                     nximages.shape[0],
                     moduleImageData.shape[2] + self.module_positions[:,0].max())),
                 imagedata])  # append the actual new image data
        self._img_data = nximages[::-1]

    def save(self, filename):
        """
        Saves the current file as another image file, the raw data is used for saving.
        :param filename: name of the saved file, extensions defines the format, please see fabio library for reference
        """
        try:
            self._img_data_fabio.save(filename)
        except AttributeError:
            im_array = np.int32(np.copy(np.flipud(self._img_data)))
            im = Image.fromarray(im_array)
            im.save(filename)

    def load_background(self, filename):
        """
        Loads an image file as background in any format known by fabIO. Automatically performs all previous img
        transformations, supersampling and recalculates background subtracted and absorption corrected image data.
        The img_changed signal will be emitted after the process.
        :param filename: path of the image file to be loaded
        """
        self.background_filename = filename
        try:
            im = Image.open(filename)
            self._background_data = np.array(im)[::-1]
        except IOError:
            self._background_data_fabio = fabio.open(filename)
            self._background_data = self._img_data_fabio.data[::-1]
        self._perform_background_transformations()

        if self._background_data.shape != self._img_data.shape:
            self._background_data = None
            self._calculate_img_data()
            self.img_changed.emit()
            raise BackgroundDimensionWrongException()

        self._calculate_img_data()
        self.img_changed.emit()

    def load_flat_image(self, filename):
        return fabio.open(filename).data[::-1]
        # TODO: make single image loading function for all formats

    def add(self, filename):
        """
        Adds an image file in any format known by fabIO. Automatically performs all previous img transformations,
        performs supersampling and recalculates background subtracted and absorption corrected image data.
        The img_changed signal will be emitted after the process.
        :param filename: path of the image file to be loaded
        """
        filename = str(filename)  # since it could also be QString
        try:
            im = Image.open(filename)
            img_data = np.array(im)[::-1]
        except IOError:
            if os.path.splitext(filename)[1].lower() == '.spe':
                spe = SpeFile(filename)
                img_data = spe.img
            else:
                img_data_fabio = fabio.open(filename)
                img_data = img_data_fabio.data[::-1]

        for transformation in self.img_transformations:
            img_data = transformation(img_data)

        if not self._img_data.shape == img_data.shape:
            return

        logger.info("Adding {0}.".format(filename))

        if self._img_data.dtype == np.uint16:  # if dtype is only uint16 we will convert to 32 bit, so that more
            # additions are possible
            self._img_data = self._img_data.astype(np.uint32)

        self._img_data += img_data

        self._calculate_img_data()
        self.img_changed.emit()

    def _image_and_background_shape_equal(self):
        """
        Tests if the original image and original background image have the same shape
        :return: Boolean
        """
        if self._background_data is None:
            return True
        if self._background_data.shape == self._img_data.shape:
            return True
        return False

    def _reset_background(self):
        """
        Resets the background data to None
        """
        self.background_filename = ''
        self._background_data = None
        self._background_data_fabio = None
        self._calculate_img_data()

    def reset_background(self):
        self._reset_background()
        self.img_changed.emit()

    def has_background(self):
        return self._background_data is not None

    @property
    def background_data(self):
        return self._background_data

    @background_data.setter
    def background_data(self, new_data):
        self._background_data = new_data
        self._calculate_img_data()
        self.img_changed.emit()

    @property
    def background_scaling(self):
        return self._background_scaling

    @background_scaling.setter
    def background_scaling(self, new_value):
        self._background_scaling = new_value
        self._calculate_img_data()
        self.img_changed.emit()

    @property
    def background_offset(self):
        return self._background_offset

    @background_offset.setter
    def background_offset(self, new_value):
        self._background_offset = new_value
        self._calculate_img_data()
        self.img_changed.emit()

    def load_series_img(self, nr):
        self.loadImageFromNxim(nr)

        self._perform_img_transformations()
        self._calculate_img_data()

        self.img_changed.emit()

    def load_next_file(self, step=1, pos=None):
        """
        Loads the next file based on the current iteration mode and the step you specify.
        :param pos:
        :param step: Defining how much you want to increment the file number. (default=1)
        """
        if self.series_max:
            self.series_pos = min(self.series_pos + step, self.series_max)
            self.load_series_img(self.series_pos)
        else:
            next_file_name = self.file_name_iterator.get_next_filename(mode=self.file_iteration_mode, step=step, pos=pos)
            if next_file_name is not None:
                self.load(next_file_name)

    def load_previous_file(self, step=1, pos=None):
        """
        Loads the previous file based on the current iteration mode and the step specified
        :param pos:
        :param step: Defining how much you want to decrement the file number. (default=1)
        """
        if self.series_max:
            self.series_pos = max(self.series_pos - step, 0)
            self.loadImageFromNxim(self.series_pos)

            self._perform_img_transformations()
            self._calculate_img_data()

            self.img_changed.emit()
        else:
            previous_file_name = self.file_name_iterator.get_previous_filename(mode=self.file_iteration_mode,
                                                                               step=step, pos=pos)
            if previous_file_name is not None:
                self.load(previous_file_name)

    def load_next_folder(self, mec_mode=False):
        """
        Loads a file with the current filename in the next folder, whereby the folder has to be iteratable by numbers.
        :param mec_mode:    Boolean which enables specific mode for MEC beamline at SLAC, where the folders and the
                            files change their during increment. (default = False)

        """
        next_file_name = self.file_name_iterator.get_next_folder(mec_mode=mec_mode)
        if next_file_name is not None:
            self.load(next_file_name)

    def load_previous_folder(self, mec_mode=False):
        """
        Loads a file with the current filename in the previous folder, whereby the folder has to be iteratable by
        numbers.
        :param mec_mode:    Boolean which enables specific mode for MEC beamline at SLAC, where the folders and the
                            files change their during increment. (default = False)
        """

        next_previous_name = self.file_name_iterator.get_previous_folder(mec_mode=mec_mode)
        if next_previous_name is not None:
            self.load(next_previous_name)

    def set_file_iteration_mode(self, mode):
        """
        Sets the file iteration mode for the load_next_file and load_previous_file functions. Possible modes:
            * 'number' will increment or decrement based on numbers in the filename.
            * 'time' will increment or decrement based on creation time for the files.
        """
        if mode == 'number':
            self.file_iteration_mode = 'number'
            self.file_name_iterator.create_timed_file_list = False
        elif mode == 'time':
            self.file_iteration_mode = 'time'
            self.file_name_iterator.create_timed_file_list = True
            self.file_name_iterator.update_filename(self.filename)

    def _calculate_img_data(self):
        """
        Calculates compound img_data based on the state of the object. This function is used internally to not compute
        those img arrays every time somebody requests the image data by get_img_data() and img_data.
        """

        # check that all data has the same dimensions
        if self._background_data is not None:
            if self._img_data.shape != self._background_data.shape:
                self._background_data = None
        if self._img_corrections.has_items():
            if self._img_data.shape != self._img_corrections.shape:
                self._img_corrections.clear()
                self.transfer_correction.reset()
                self.corrections_removed.emit()

        # calculate the current _img_data
        if self._background_data is not None and not self._img_corrections.has_items():
            self._img_data_background_subtracted = self._img_data - (self._background_scaling *
                                                                     self._background_data +
                                                                     self._background_offset)
        elif self._background_data is None and self._img_corrections.has_items():
            self._img_data_absorption_corrected = self._img_data / self._img_corrections.get_data()

        elif self._background_data is not None and self._img_corrections.has_items():
            self._img_data_background_subtracted_absorption_corrected = (self._img_data - (
                    self._background_scaling * self._background_data + self._background_offset)) / \
                                                                        self._img_corrections.get_data()

        # supersample the current image data
        if self.supersampling_factor > 1:
            if self._background_data is None and not self._img_corrections.has_items():
                self._img_data_supersampled = self.supersample_data(self._img_data, self.supersampling_factor)

            if self._background_data is not None and not self._img_corrections.has_items():
                self._img_data_supersampled_background_subtracted = \
                    self.supersample_data(self._img_data_background_subtracted, self.supersampling_factor)

            elif self._background_data is None and self._img_corrections.has_items():
                self._img_data_supersampled_absorption_corrected = \
                    self.supersample_data(self._img_data_absorption_corrected, self.supersampling_factor)

            elif self._background_data is not None and self._img_corrections.has_items():
                self._img_data_supersampled_background_subtracted_absorption_corrected = \
                    self.supersample_data(self._img_data_background_subtracted_absorption_corrected,
                                          self.supersampling_factor)

    @property
    def img_data(self):
        """
        :return:
            The image based on the current state of the ImgData object. If supersampling is set it will return a
            supersampled image array if background_data is set it will return a background_subtracted array and so on.
            It also works for combinations of all these options.
        """
        if self.supersampling_factor == 1:
            if self._background_data is None and not self._img_corrections.has_items():
                return self._img_data * self.factor

            elif self._background_data is not None and not self._img_corrections.has_items():
                return self._img_data_background_subtracted * self.factor

            elif self._background_data is None and self._img_corrections.has_items():
                return self._img_data_absorption_corrected * self.factor

            elif self._background_data is not None and self._img_corrections.has_items():
                return self._img_data_background_subtracted_absorption_corrected * self.factor

        else:
            if self._background_data is None and not self._img_corrections.has_items():
                return self._img_data_supersampled * self.factor

            elif self._background_data is not None and not self._img_corrections.has_items():
                return self._img_data_supersampled_background_subtracted * self.factor

            elif self._background_data is None and self._img_corrections.has_items():
                return self._img_data_supersampled_absorption_corrected * self.factor

            elif self._background_data is not None and self._img_corrections.has_items():
                return self._img_data_supersampled_background_subtracted_absorption_corrected * self.factor
        return self._img_data * self.factor

    @property
    def raw_img_data(self):
        return self._img_data

    def rotate_img_p90(self):
        """
        Rotates the image by 90 degree and updates the background accordingly (does not effect absorption correction).
        The transformation is saved and applied to every new image and background image loaded.
        The img_changed signal will be emitted after the process.
        """
        self._img_data = rotate_matrix_p90(self._img_data)

        if self._background_data is not None:
            self._background_data = rotate_matrix_p90(self._background_data)

        self.img_transformations.append(rotate_matrix_p90)

        self.transformations_changed.emit()
        self._calculate_img_data()
        self.img_changed.emit()

    def rotate_img_m90(self):
        """
        Rotates the image by -90 degree and updates the background accordingly (does not effect absorption correction).
        The transformation is saved and applied to every new image and background image loaded.
        The img_changed signal will be emitted after the process.
        """
        self._img_data = rotate_matrix_m90(self._img_data)
        if self._background_data is not None:
            self._background_data = rotate_matrix_m90(self._background_data)
        self.img_transformations.append(rotate_matrix_m90)
        self.transformations_changed.emit()

        self._calculate_img_data()
        self.img_changed.emit()

    def flip_img_horizontally(self):
        """
        Flips image about a horizontal axis and updates the background accordingly (does not effect absorption
        correction). The transformation is saved and applied to every new image and background image loaded.
        The img_changed signal will be emitted after the process.
        """
        self._img_data = np.fliplr(self._img_data)
        if self._background_data is not None:
            self._background_data = np.fliplr(self._background_data)
        self.img_transformations.append(np.fliplr)
        self.transformations_changed.emit()

        self._calculate_img_data()
        self.img_changed.emit()

    def flip_img_vertically(self):
        """
        Flips image about a vertical axis and updates the background accordingly (does not effect absorption
        correction). The transformation is saved and applied to every new image and background image loaded.
        The img_changed signal will be emitted after the process.
        """
        self._img_data = np.flipud(self._img_data)
        if self._background_data is not None:
            self._background_data = np.flipud(self._background_data)
        self.img_transformations.append(np.flipud)
        self.transformations_changed.emit()

        self._calculate_img_data()
        self.img_changed.emit()

    def reset_img_transformations(self):
        """
        Reverts all image transformations and resets the transformation stack.
        The img_changed signal will be emitted after the process.
        """
        for transformation in reversed(self.img_transformations):
            if transformation == rotate_matrix_p90:
                self._img_data = rotate_matrix_m90(self._img_data)
                if self._background_data is not None:
                    self._background_data = rotate_matrix_m90(self._background_data)
            elif transformation == rotate_matrix_m90:
                self._img_data = rotate_matrix_p90(self._img_data)
                if self._background_data is not None:
                    self._background_data = rotate_matrix_p90(self._background_data)
            else:
                self._img_data = transformation(self._img_data)
                if self._background_data is not None:
                    self._background_data = transformation(self._background_data)
        self.img_transformations = []
        self.transformations_changed.emit()

        self._calculate_img_data()
        self.img_changed.emit()

    def _perform_img_transformations(self):
        """
        Performs all saved image transformation on original image.
        """
        for transformation in self.img_transformations:
            self._img_data = transformation(self._img_data)

    def _revert_img_transformations(self):
        """
        Reverts all saved image transformations on the image. (Does not delete the transformations list, any new loaded
        image will be transformed again)
        """
        for transformation in reversed(self.img_transformations):
            self._img_data = transformation(self._img_data)

    def _perform_background_transformations(self):
        """
        Performs all saved image transformation on background image.
        """
        if self._background_data is not None:
            for transformation in self.img_transformations:
                self._background_data = transformation(self._background_data)

    def _revert_background_transformations(self):
        """
        Performs all saved image transformation on background image.
        """
        if self._background_data is not None:
            for transformation in reversed(self.img_transformations):
                self._background_data = transformation(self._background_data)

    def get_transformations_string_list(self):
        transformation_list = []
        for transformation in self.img_transformations:
            transformation_list.append(transformation.__name__)
        return transformation_list

    def load_transformations_string_list(self, transformations):
        self._revert_img_transformations()
        self._revert_background_transformations()
        self.img_transformations = []
        for transformation in transformations:
            if transformation == "flipud":
                self.img_transformations.append(np.flipud)
            elif transformation == "fliplr":
                self.img_transformations.append(np.fliplr)
            elif transformation == "rotate_matrix_m90":
                self.img_transformations.append(rotate_matrix_m90)
            elif transformation == "rotate_matrix_p90":
                self.img_transformations.append(rotate_matrix_p90)
        self._perform_img_transformations()
        self._perform_background_transformations()

    def set_supersampling(self, factor=None):
        """
        Stores the supersampling factor and calculates supersampled original and background image arrays.
        Updates all data calculations according to current ImgData object state.
        The img_changed signal will be emitted after the process.
        :param factor: int - supersampling factor
        """
        self.supersampling_factor = factor
        self._calculate_img_data()
        self.img_changed.emit()

    def supersample_data(self, img_data, factor):
        """
        Creates a supersampled array from img_data.
        :param img_data: image array
        :param factor: int - supersampling factor
        :return: supersampled image
        """
        if factor > 1:
            img_data_supersampled = np.zeros((img_data.shape[0] * factor,
                                              img_data.shape[1] * factor))
            for row in range(factor):
                for col in range(factor):
                    img_data_supersampled[row::factor, col::factor] = img_data

            return img_data_supersampled
        else:
            return img_data

    def add_img_correction(self, correction, name=None):
        """
        Adds a correction to be applied to the image. Corrections are applied multiplicative for each pixel and after
        each other, depending on the order of addition.
        :param external:
        :param correction: An Object inheriting the ImgCorrectionInterface.
        :type correction: ImgCorrectionInterface
        :param name: correction can be given a name, to selectively delete or obtain later.
        :type name: basestring
        """
        if self._img_corrections.add(correction, name, self._img_data.shape):
            self._calculate_img_data()
            self.img_changed.emit()

    def change_image_correction(self, name, params):
        self._img_corrections.set_params(name, params)
        self._calculate_img_data()
        self.img_changed.emit()

    def get_img_correction(self, name):
        """
        :param name: correction name which was specified during the addition of the image correction.
        :return: the specified correction
        """
        return self._img_corrections.get_correction(name)

    def delete_img_correction(self, name=None):
        """
        :param name: deletes a correction from the correction calculation with a specific name. if no name is specified
         the last added correction is deleted.
        """
        self._img_corrections.delete(name)
        self._calculate_img_data()
        self.img_changed.emit()

    def enable_transfer_function(self):
        if self.transfer_correction.get_data() is not None and \
                self.get_img_correction('transfer') is None:
            self.add_img_correction(self.transfer_correction, 'transfer')
        if self.get_img_correction('transfer') is not None:
            self._calculate_img_data()
            self.img_changed.emit()

    def disable_transfer_function(self):
        if self.get_img_correction('transfer') is not None:
            self.delete_img_correction('transfer')

    @property
    def img_corrections(self):
        return self._img_corrections

    def has_corrections(self):
        """
        :return: Whether the ImgData object has active absorption corrections or not
        """
        return self._img_corrections.has_items()

    def _get_file_info(self, image):
        """
        reads the file info from tif_tags and returns a file info
        """
        result = ""
        tags = image.tag
        useful_keys = []
        for key in tags.keys():
            if key > 300:
                useful_keys.append(key)

        useful_keys.sort()
        for key in useful_keys:
            tag = tags[key][0]
            if isinstance(tag, basestring):
                new_line = str(tag) + "\n"
                new_line = new_line.replace(":", ":\t", 1)
                result += new_line
        return result

    def _get_motors_info(self, image):
        """
        reads the file info from tif_tags and returns positions of vertical, horizontal, focus and omega motors
        """
        result = {}
        tags = image.tag

        useful_tags = ['Horizontal:', 'Vertical:', 'Focus:', 'Omega:']

        try:
            tag_values = tags.itervalues()
        except AttributeError:
            tag_values = tags.values()

        for value in tag_values:
            for key in useful_tags:
                if key in str(value):
                    k, v = str(value[0]).split(':')
                    result[str(k)] = float(v)
        return result

    @property
    def autoprocess(self):
        return self._autoprocess

    @autoprocess.setter
    def autoprocess(self, new_val):
        self._autoprocess = new_val
        if new_val:
            self._directory_watcher.activate()
        else:
            self._directory_watcher.deactivate()

    @property
    def factor(self):
        return self._factor

    @factor.setter
    def factor(self, new_value):
        self._factor = new_value
        self.img_changed.emit()


class BackgroundDimensionWrongException(Exception):
    pass


class NoHandlerForHDF5(Exception):
    pass
