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
        data_path = "INSTRUMENT/HED_EXP_VAREX/CAM/1:daqOutput/data/image/pixels"

        try:
            nx_file = h5py.File(filename, "r")
        except OSError:
            raise IOError("not a loadable hdf5 file")

        img_data=np.array(nx_file[data_path])
        nozeros =img_data[np.any(img_data, axis=(1, 2))]
        sortdata = nozeros-np.mean(nozeros, axis=0)
        intensity = np.partition(np.reshape(sortdata, (nozeros.shape[0], -1)), -100, axis=1)[:,-100]

        partmean = np.mean(nozeros[intensity < 36], axis=0)
        datapictureidx = np.argwhere(intensity > 36)
        print(datapictureidx+2)
        self.series_len = nozeros.shape[0]+1
        self.full_img_data = nozeros-partmean
        self.full_img_data = np.concatenate([np.sum(self.full_img_data[datapictureidx],axis=0), self.full_img_data])

    def get_image(self, image_nr):
        """
        Gets the data for the given image nr and stitches the tiles together
        :param image_nr: position from which to take the image from the image set
        :return: image_data
        """
        image = self.full_img_data[image_nr]

        return image[::-1]
