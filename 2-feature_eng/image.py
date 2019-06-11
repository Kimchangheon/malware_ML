import numpy as np
import os, array
import scipy.misc
import glob
from PIL import Image


class IMAGE_feature():

    def __init__(self, in_path, out_path):

        self.in_path = in_path
        self.out_path = out_path


    def get_image(self, path, file):

            filename = path + file

            f = open(filename,'rb')
            ln = os.path.getsize(filename) 

            width = int(ln**0.5) 
            rem = ln % width 

            a = array.array("B") 
            a.fromfile(f,ln-rem) 
            f.close() 

            g = np.reshape(a, (int(len(a)/width), width)) 
            g = np.uint8(g)

            fpng = self.out_path + file + ".png"
            scipy.misc.imsave(fpng, g) 

            outfile = self.out_path + file + "_thumb.png"
            print(outfile)
            size = 256, 256

            if fpng != outfile:
                im = Image.open(fpng)
                im.thumbnail(size, Image.ANTIALIAS) 
                im.save(outfile, "PNG")

    def get_all(self):
        path = self.in_path

        for file in os.listdir(path): 
            self.get_image(path, file)



def main():

    mal_path = '../samples/malware/'
    nor_path = '../samples/normal/'

    mal_out_path = '../images/malware/'
    nor_out_path = '../images/normal/'

    im1 = IMAGE_feature(mal_path, mal_out_path)
    im1.get_all()

    im2 = IMAGE_feature(nor_path, nor_out_path)
    im2.get_all()


if __name__ == '__main__':
    main()