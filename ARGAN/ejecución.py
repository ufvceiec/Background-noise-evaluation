import argparse
import os
from PIL import Image
import cv2


for f in os.listdir("ejecuciones/imgs/recons/"):
    os.remove("ejecuciones/imgs/recons/" + f)
    os.remove("ejecuciones/imgs/segmentadas/" + f)
    os.remove("ejecuciones/imgs/segmentadas_recons/" + f)

ps = argparse.ArgumentParser()
ps.add_argument('--foto', required=True)
args = ps.parse_args()

# Automático
os.system('python -m test_copy2.py --model_path saved_models/512x512.pix2pix.segmentation.t1456789.h5 --samples_path ' + args.foto + ' --donde ejecuciones\\imgs\\segmentadas')
os.system('python -m test_copy2.py --model_path saved_models/512x512.pix2pix.color_reconstruction.t158.h5 --samples_path ejecuciones/imgs/segmentadas/ --donde ejecuciones\\imgs\\segmentadas_recons')

imagen = cv2.imread('ejecuciones/imgs/segmentadas_recons/img_0.png')
flip1 = cv2.flip(imagen,1)
cv2.imwrite('ejecuciones/imgs/segmentadas_recons/img_0.png', flip1)

os.system('python -m test.py --model_path saved_models/512x512.pix2pix.color_assisted.t1456789.h5 --samples_path ' + args.foto + ' --samples_path2 ejecuciones/imgs/segmentadas_recons/ --donde ejecuciones\\imgs\\recons')



