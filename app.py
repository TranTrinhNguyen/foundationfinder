import cv2
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import dlib
import os
import pickle
import logging
import torch

from colorthief import ColorThief
from classes import WBsRGB as wb_srgb
from PIL import Image
from arch import deep_wb_model
from numpy import asarray
import utilities.utils as utls
from utilities.deepWB import deep_wb
import arch.splitNetworks as splitter
from arch import deep_wb_single_task

def main():
    '''Main function that will run the whole app
    
    '''
    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.title('Foundation Finder')
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Main", "Deep White Balance", "KNN White Balance", "Dataset"])
    if app_mode == "Deep White Balance":
        st_deep_wb()
    elif app_mode == "Dataset":
        st_load_data()
    elif app_mode == "KNN White Balance":
        st_knn_wb()
    elif app_mode == "Main":
        st_all_wb()

def st_all_wb():
    '''Execute the app to recommend foundation shade based on input image
    
    
    '''
    model = load_knn_model()
    uploaded_file = st.file_uploader("Upload Selfie", type='jpg')
    if uploaded_file:
        img = Image.open(uploaded_file)
        # Resize image for faster processing time
        st.write("Uploaded Image")
        st.image(img, use_column_width=True)
        
        # Deep Learning Based White-Balancing
        img.save('uploaded_file.jpg')
        input = 'uploaded_file.jpg'
        deep_inImg, deep_outImg, deep_path = deep_white_balancing(input)

        # KNN Based White-Balancing
        knn_inImg = np.array(img)
        knn_inImg = cv2.cvtColor(knn_inImg, cv2.COLOR_RGB2BGR)
        knn_inImg, knn_outImg, knn_path = knn_white_balancing(knn_inImg, imshow=0, imwrite=1, imcrop=1)
        
        col1, col2, col3 = st.beta_columns(3)

        with col1:
            st.write("Original Image")
            st.image(knn_inImg, use_column_width=True, channels='BGR')
            dominant_color = get_dominant_color(input)
            dominant_palette = np.zeros((150,150,3), np.uint8)
            dominant_palette[:] = dominant_color
            st.image(dominant_palette, use_column_width=True)
            hex = '#%02x%02x%02x' % (dominant_color[0], dominant_color[1], dominant_color[2])
            st.markdown(f'''<div align="center"><b>{hex}</b></div>''', unsafe_allow_html=True) 
            st.write("Recommended Product")
            product = model.predict([dominant_color])[0]
            st.write(product)   

        with col2:
            st.write("KNN Image")
            st.image(knn_outImg, use_column_width=True, channels='BGR')
            dominant_color = get_dominant_color(knn_path)
            dominant_palette = np.zeros((150,150,3), np.uint8)
            dominant_palette[:] = dominant_color
            st.image(dominant_palette, use_column_width=True)
            hex = '#%02x%02x%02x' % (dominant_color[0], dominant_color[1], dominant_color[2])
            st.markdown(f'''<div align="center"><b>{hex}</b></div>''', unsafe_allow_html=True) 
            st.write("Recommended Product")
            product = model.predict([dominant_color])[0]
            st.write(product)   
            

        with col3:
            st.write("Deep Learning Image")
            st.image(deep_outImg, use_column_width=True, channels='RGB')
            dominant_color = get_dominant_color(deep_path)
            dominant_palette = np.zeros((150,150,3), np.uint8)
            dominant_palette[:] = dominant_color
            st.image(dominant_palette, use_column_width=True)
            hex = '#%02x%02x%02x' % (dominant_color[0], dominant_color[1], dominant_color[2])
            st.markdown(f'''<div align="center"><b>{hex}</b></div>''', unsafe_allow_html=True) 
            st.write("Recommended Product")
            product = model.predict([dominant_color])[0]
            st.write(product)   

def st_deep_wb():
    '''Execute the app to recommend foundation shade based on input image
    
    
    '''
    model = load_knn_model()
    uploaded_file = st.file_uploader("Upload Selfie", type='jpg')
    if uploaded_file:
        img = Image.open(uploaded_file)
        img.save('uploaded_file.jpg')
        input = 'uploaded_file.jpg'
        inImg, outImg, path = deep_white_balancing(input)
        col1, col2 = st.beta_columns(2)
 
        with col1:
            st.write("Original Image")
            st.image(inImg, use_column_width=True, channels='RGB')

        with col2:
            st.write("White-Balanced Image")
            st.image(outImg, use_column_width=True, channels='RGB')

        dominant_color = get_dominant_color(path)
        dominant_palette = np.zeros((150,150,3), np.uint8)
        dominant_palette[:] = dominant_color
        st.write("Foundation Shade Color")
        st.image(dominant_palette, use_column_width=True)

        st.write("Recommended Product")
        product = model.predict([dominant_color])[0]
        st.text(product)   

def st_knn_wb():
    '''Execute the app to recommend foundation shade based on input image
    
    '''
    model = load_knn_model()
    uploaded_file = st.file_uploader("Upload Selfie", type='jpg')
    if uploaded_file:
        inImg = Image.open(uploaded_file)
        inImg = np.array(inImg)
        inImg = cv2.cvtColor(inImg, cv2.COLOR_RGB2BGR)
        inImg, outImg, path = knn_white_balancing(inImg, imshow=0, imwrite=1, imcrop=1)
        col1, col2 = st.beta_columns(2)
        # To read file as bytes:
        with col1:
            st.write("Original Image")
            st.image(inImg, use_column_width=True, channels='BGR')

        with col2:
            st.write("White-Balanced Image")
            st.image(outImg, use_column_width=True, channels='BGR')

        dominant_color = get_dominant_color(path)
        dominant_palette = np.zeros((150,150,3), np.uint8)
        dominant_palette[:] = dominant_color
        st.write("Foundation Shade Color")
        st.image(dominant_palette, use_column_width=True)

        st.write("Recommended Product")
        product = model.predict([dominant_color])[0]
        st.text(product)

def st_load_data():
    data = './data/shades.json'
    df = pd.read_json(data)
    df['Label'] = df['brand'] + ' ' + df['product'] + ' ' + df['shade']
    df['R'] = df['hex'].apply(lambda x: hex_to_rgb(x)[0])
    df['G'] = df['hex'].apply(lambda x: hex_to_rgb(x)[1])
    df['B'] = df['hex'].apply(lambda x: hex_to_rgb(x)[2])
    st.dataframe(df)

@st.cache(allow_output_mutation=True)
def load_knn_model():
    # load the model from disk
    model = pickle.load(open('./models/knnpickle.sav', 'rb'))   
    return model

def hex_to_rgb(value):
    lv = len(value)
    return [int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)]

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
  (h, w) = image.shape[:2]

  if width is None and height is None:
    return image
  if width is None:
    r = height / float(h)
    dim = (int(w * r), height)
  else:
    r = width / float(w)
    dim = (width, int(h * r))

  return cv2.resize(image, dim, interpolation=inter)

# get the dominant color
def get_dominant_color(path, show=0):
    color_thief = ColorThief(path)
    dominant_color = color_thief.get_color(quality=1)
    
    if show == 1:
        dominant_palette = np.zeros((300,300,3), np.uint8)
        dominant_palette[:] = dominant_color
        plt.imshow(dominant_palette)
        plt.title('Dominant Color')
        plt.show()
        
    return list(dominant_color)

# KNN based white-balacing function    
def knn_white_balancing(inImg, imshow=1, imwrite=1, imcrop=0):
    # use upgraded_model= 1 to load our new model that is upgraded with new
    # training examples.
    upgraded_model = 0
    # use gamut_mapping = 1 for scaling, 2 for clipping (our paper's results
    # reported using clipping). If the image is over-saturated, scaling is
    # recommended.
    gamut_mapping = 2
    # processing
    # create an instance of the WB model
    wbModel = wb_srgb.WBsRGB(gamut_mapping=gamut_mapping,
                             upgraded=upgraded_model)
    # I = cv2.imread(in_img)  # read the image
    outImg = wbModel.correctImage(inImg) # white balance i
    outImg =(outImg*255).astype(np.uint8) 
    if imcrop == 1:
        detector = dlib.get_frontal_face_detector()
        faces = detector(outImg)
        x1 = faces[0].left() # left point
        y1 = faces[0].top() # top point
        x2 = faces[0].right() # right point
        y2 = faces[0].bottom() # bottom point
        outImg = outImg[y1:y2, x1:x2]
        inImg = inImg[y1:y2, x1:x2]
    
    #inImg = ResizeWithAspectRatio(inImg, width=600)
    #b,g,r = cv2.split(inImg)
    #inImg = cv2.merge((r,g,b))

    #outImg = ResizeWithAspectRatio(outImg, width=600)
    #b,g,r = cv2.split(outImg)
    #outImg = cv2.merge((g,b,r))

    path = ''
    if imwrite == 1:
        cv2.imwrite('/' + 'knn_result.jpg', outImg)  # save it   
        path = '/' + 'knn_result.jpg'
    
    return inImg, outImg, path

# Deep Learning based white-balacing function
def deep_white_balancing(input, out_dir='./result_images', task='awb', target_color_temp=None, S=960, show=True, save=True, device='cpu'):
    model_dir = './models'
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    if device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    fn = input
    tosave = save

    if target_color_temp:
        assert 2850 <= target_color_temp <= 7500, (
                'Color temperature should be in the range [2850 - 7500], but the given one is %d' % target_color_temp)

        if task != 'editing':
            raise Exception('The task should be editing when a target color temperature is specified.')

    logging.info(f'Using device {device}')

    if tosave:
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

    if task == 'all':
        if os.path.exists(os.path.join(model_dir, 'net_awb.pth')) and \
                os.path.exists(os.path.join(model_dir, 'net_t.pth')) and \
                os.path.exists(os.path.join(model_dir, 'net_s.pth')):
            # load awb net
            net_awb = deep_wb_single_task.deepWBnet()
            logging.info("Loading model {}".format(os.path.join(model_dir, 'net_awb.pth')))
            net_awb.to(device=device)
            net_awb.load_state_dict(torch.load(os.path.join(model_dir, 'net_awb.pth'),
                                               map_location=device))
            net_awb.eval()
            # load tungsten net
            net_t = deep_wb_single_task.deepWBnet()
            logging.info("Loading model {}".format(os.path.join(model_dir, 'net_t.pth')))
            net_t.to(device=device)
            net_t.load_state_dict(
                torch.load(os.path.join(model_dir, 'net_t.pth'), map_location=device))
            net_t.eval()
            # load shade net
            net_s = deep_wb_single_task.deepWBnet()
            logging.info("Loading model {}".format(os.path.join(model_dir, 'net_s.pth')))
            net_s.to(device=device)
            net_s.load_state_dict(
                torch.load(os.path.join(model_dir, 'net_s.pth'), map_location=device))
            net_s.eval()
            logging.info("Models loaded !")
        elif os.path.exists(os.path.join(model_dir, 'net.pth')):
            net = deep_wb_model.deepWBNet()
            logging.info("Loading model {}".format(os.path.join(model_dir, 'net.pth')))
            net.load_state_dict(torch.load(os.path.join(model_dir, 'net.pth')))
            net_awb, net_t, net_s = splitter.splitNetworks(net)
            net_awb.to(device=device)
            net_awb.eval()
            net_t.to(device=device)
            net_t.eval()
            net_s.to(device=device)
            net_s.eval()
        else:
            raise Exception('Model not found!')
    elif task == 'editing':
        if os.path.exists(os.path.join(model_dir, 'net_t.pth')) and \
                os.path.exists(os.path.join(model_dir, 'net_s.pth')):
            # load tungsten net
            net_t = deep_wb_single_task.deepWBnet()
            logging.info("Loading model {}".format(os.path.join(model_dir, 'net_t.pth')))
            net_t.to(device=device)
            net_t.load_state_dict(
                torch.load(os.path.join(model_dir, 'net_t.pth'), map_location=device))
            net_t.eval()
            # load shade net
            net_s = deep_wb_single_task.deepWBnet()
            logging.info("Loading model {}".format(os.path.join(model_dir, 'net_s.pth')))
            net_s.to(device=device)
            net_s.load_state_dict(
                torch.load(os.path.join(model_dir, 'net_s.pth'), map_location=device))
            net_s.eval()
            logging.info("Models loaded !")
        elif os.path.exists(os.path.join(model_dir, 'net.pth')):
            net = deep_wb_model.deepWBNet()
            logging.info("Loading model {}".format(os.path.join(model_dir, 'net.pth')))
            logging.info(f'Using device {device}')
            net.load_state_dict(torch.load(os.path.join(model_dir, 'net.pth')))
            _, net_t, net_s = splitter.splitNetworks(net)
            net_t.to(device=device)
            net_t.eval()
            net_s.to(device=device)
            net_s.eval()
        else:
            raise Exception('Model not found!')
    elif task == 'awb':
        if os.path.exists(os.path.join(model_dir, 'net_awb.pth')):
            # load awb net
            net_awb = deep_wb_single_task.deepWBnet()
            logging.info("Loading model {}".format(os.path.join(model_dir, 'net_awb.pth')))
            logging.info(f'Using device {device}')
            net_awb.to(device=device)
            net_awb.load_state_dict(torch.load(os.path.join(model_dir, 'net_awb.pth'),
                                               map_location=device))
            net_awb.eval()
        elif os.path.exists(os.path.join(model_dir, 'net.pth')):
            net = deep_wb_model.deepWBNet()
            logging.info("Loading model {}".format(os.path.join(model_dir, 'net.pth')))
            logging.info(f'Using device {device}')
            net.load_state_dict(torch.load(os.path.join(model_dir, 'net.pth')))
            net_awb, _, _ = splitter.splitNetworks(net)
            net_awb.to(device=device)
            net_awb.eval()
        else:
            raise Exception('Model not found!')
    else:
        raise Exception("Wrong task! Task should be: 'AWB', 'editing', or 'all'")
        
    logging.info("Processing image {} ...".format(fn))
    img = Image.open(fn)
    name = 'result'
    if task == 'all':  # awb and editing tasks
        out_awb, out_t, out_s = deep_wb(img, task=task, net_awb=net_awb, net_s=net_s, net_t=net_t,
                                        device=device, s=S)
        out_f, out_d, out_c = utls.colorTempInterpolate(out_t, out_s)
        if tosave:
            result_awb = utls.to_image(out_awb)
            result_t = utls.to_image(out_t)
            result_s = utls.to_image(out_s)
            result_f = utls.to_image(out_f)
            result_d = utls.to_image(out_d)
            result_c = utls.to_image(out_c)
            result_awb.save(os.path.join(out_dir, name + '_AWB.png'))
            result_s.save(os.path.join(out_dir, name + '_S.png'))
            result_t.save(os.path.join(out_dir, name + '_T.png'))
            result_f.save(os.path.join(out_dir, name + '_F.png'))
            result_d.save(os.path.join(out_dir, name + '_D.png'))
            result_c.save(os.path.join(out_dir, name + '_C.png'))

        if show:
            logging.info("Visualizing results for image: {}, close to continue ...".format(fn))
            utls.imshow(img, result_awb, result_t, result_f, result_d, result_c, result_s)

    elif task == 'awb':  # awb task
        out_awb = deep_wb(img, task=task, net_awb=net_awb, device=device, s=S)
        if tosave:
            result_awb = utls.to_image(out_awb)
            result_awb.save(os.path.join(out_dir, name + '_AWB.png'))

        if show:
            logging.info("Visualizing result for image: {}, close to continue ...".format(fn))
            utls.imshow(img, result_awb)
 
    else:  # editing
        out_t, out_s = deep_wb(img, task=task, net_s=net_s, net_t=net_t, device=device, s=S)

        if target_color_temp:
            out = utls.colorTempInterpolate_w_target(out_t, out_s, target_color_temp)
            if tosave:
                out = utls.to_image(out)
                out.save(os.path.join(out_dir, name + '_%d.png' % target_color_temp))

            if show:
                logging.info("Visualizing result for image: {}, close to continue ...".format(fn))
                utls.imshow(img, out, colortemp=target_color_temp)

        else:
            out_f, out_d, out_c = utls.colorTempInterpolate(out_t, out_s)
            if tosave:
                result_t = utls.to_image(out_t)
                result_s = utls.to_image(out_s)
                result_f = utls.to_image(out_f)
                result_d = utls.to_image(out_d)
                result_c = utls.to_image(out_c)
                result_s.save(os.path.join(out_dir, name + '_S.png'))
                result_t.save(os.path.join(out_dir, name + '_T.png'))
                result_f.save(os.path.join(out_dir, name + '_F.png'))
                result_d.save(os.path.join(out_dir, name + '_D.png'))
                result_c.save(os.path.join(out_dir, name + '_C.png'))

            if show:
                logging.info("Visualizing results for image: {}, close to continue ...".format(fn))
                utls.imshow(img, result_t, result_f, result_d, result_c, result_s)
    
    inImg = asarray(img)
    outImg = asarray(result_awb)
    detector = dlib.get_frontal_face_detector()
    faces = detector(inImg)
    x1 = faces[0].left() # left point
    y1 = faces[0].top() # top point
    x2 = faces[0].right() # right point
    y2 = faces[0].bottom() # bottom point
    outImg = outImg[y1:y2, x1:x2]
    inImg = inImg[y1:y2, x1:x2]

    cropped_img = Image.fromarray(outImg)
    path = 'result_images/cropped_AWB.jpg'
    cropped_img.save(path)
    
    return inImg, outImg, path

if __name__ == "__main__":
    main()
    