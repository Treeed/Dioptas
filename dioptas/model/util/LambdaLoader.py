import numpy as np
import h5py
import re


class LambdaImage:
    def __init__(self, filename):
        """
        Loads an image produced by a Lambda detector.
        :param filename: path to the image file to be loaded
        """
        detector_identifiers = [["/entry/instrument/detector/description", "Lambda"], ["/entry/instrument/detector/description", b"Lambda"]]
        filenumber_list = [1, 2, 3]
        regex_in = r"(.+_m)\d(.nxs)"
        regex_out = r"\g<1>{}\g<2>"
        data_path = "entry/instrument/detector/data"
        module_positions_path = "/entry/instrument/detector/translation/distance"

        try:
            nx_file = h5py.File(filename, "r")
        except OSError:
            raise IOError("not a loadable hdf5 file")

        for identifier in detector_identifiers:
            try:
                if nx_file[identifier[0]][0] == identifier[1]:
                    break
            except KeyError:
                pass
        else:
            raise IOError("not a lambda image")


        # the image data is spread over multiple files, so we compile a list of them here
        lambda_files = []

        for moduleIndex in filenumber_list:
            try:
                lambda_files.append(h5py.File(re.sub(regex_in, regex_out.format(moduleIndex), filename), "r"))
            except OSError:
                pass

        self.full_img_data = [imageFile[data_path] for imageFile in lambda_files]

        self._module_pos = np.array([np.ravel(nxim[module_positions_path]).astype(int) for nxim in lambda_files])

        # remove any empty columns/rows to the left or top of the image data or shift any negative rows/columns into the positive
        np.subtract(self._module_pos, self._module_pos[:, 0].min(), self._module_pos, where=[1, 0, 0])
        np.subtract(self._module_pos, self._module_pos[0][1], self._module_pos, where=[0, 1, 0])
        self.series_len = lambda_files[0][data_path].shape[0]

    def get_image(self, image_nr):
        """
        Gets the data for the given image nr and stitches the tiles together
        :param image_nr: position from which to take the image from the image set
        :return: image_data
        """
        image_data = np.array([module[image_nr] for module in self.full_img_data])
        image = np.zeros([image_data[0].shape[0] + self._module_pos[-1, 1],
                          image_data[0].shape[1] + self._module_pos[:, 0].max()])

        for module_pos, module_image_data in zip(self._module_pos, image_data):
            image[module_pos[1]:module_pos[1]+module_image_data.shape[0],
                  module_pos[0]:module_pos[0]+module_image_data.shape[1]] = module_image_data

        return image[::-1]
