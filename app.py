import gradio as gr
from fastai.vision.all import *
import skimage
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

learn = load_learner('eksport1.pkl')

labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred, pred_idx, probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

title = 'Potato Disease Classifier'
description = 'A potato disease classifier trained on random images from internet with fastai. Created as a demo for Gradio and HuggingFace Spaces.'
examples = ['early_blight.JPG', 'healthy.JPG', 'late_blight.JPG']

gr.Interface(fn = predict,
    inputs = 'image',
    outputs = 'label',
    title = title,
    description = description,
    examples = examples).launch()