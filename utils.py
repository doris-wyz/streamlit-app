import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
import shap

def st_shap(plot, height=None):
            shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
            components.html(shap_html, height=height)