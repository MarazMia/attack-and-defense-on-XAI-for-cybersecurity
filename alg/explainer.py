import numpy as np
import shap
import torch

import sys
import os

current_dir = os.path.dirname(__file__)

foldery_path = os.path.join(current_dir, '..', 'Model')

sys.path.append(foldery_path)

# Now you can import code_to_import
import TabNet

class Explainer:
    def __init__(self, model, data, model_type):
        self.model = model
        self.data = data
        self.model_type = model_type

        try:
            if self.model_type.lower() == 't':
                self.explainer = lambda b, f: shap.explainers.Tree(model, data=b, model_output="probability")(f)
            elif self.model_type.lower() == 'l':
                self.explainer = lambda b, f: shap.explainers.Linear(model, masker=shap.maskers.Independent(data = b), data=b, model_output="probability")(f) 
            elif self.model_type.lower() == 'd':
                # For PyTorch models, we need to pass the model directly, not a prediction function
                if isinstance(data, np.ndarray) or isinstance(model, TabNet.PyTorchModelWrapper):
                    background_df = data.sample(n=100, random_state=1)
                    background_tensor = self.model._preprocess_data(background_df)
                    self.explainer = shap.DeepExplainer(model.pytorch_model, background_tensor)
                else:
                    #   Default to Kernel explainer for unknown types
                    raise ValueError("Unknown model type, falling back to KernelExplainer")
                
        except Exception as e:
            print(f"Warning: {str(e)} - Falling back to KernelExplainer")
            # For KernelExplainer, we need a prediction function
            if hasattr(model, 'predict_proba'):
                predict_func = model.predict_proba
            else:
                predict_func = model.predict
            self.explainer = shap.KernelExplainer(predict_func, data)



    def shap_values(self, xb, xf=None):
        if self.model_type=='d':
            sv = self.explainer.shap_values(self.model._preprocess_data(xb))
            # we are focusing on the attack label explnation thus used sv[1]
            # return np.mean(np.abs(sv[1]), axis=0)
            return sv[1]
        return np.absolute(self.explainer(xb, xb).values).mean(axis=0)
        # if xf is None:
        # else:
        #     return self.explainer(xb, xf).values


    def shap_values_ranking(self, xb, xf=None):
        return (-self.shap_values(xb, xf)).argsort().argsort()


    def shap_values_pop(self, xb, xf=None):
        xb_long = xb.reshape((xb.shape[0], xb.shape[1]*xb.shape[2]))
        if xf is None: # 2d
            return np.apply_along_axis(
                lambda x, d: self.shap_values(x.reshape((d[0], d[1]))),
                1, xb_long, d=xb[0].shape
            )
        else: # 1d
            return np.apply_along_axis(
                lambda x, f, d: self.shap_values(x.reshape((d[0], d[1])), f),
                1, xb_long, f=xf, d=xb[0].shape
            )


    def shap_values_ranking_pop(self, xb, xf=None):
        return (-self.shap_values_pop(xb, xf)).argsort(axis=1).argsort(axis=1)
    