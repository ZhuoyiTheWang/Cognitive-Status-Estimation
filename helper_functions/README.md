### **1. `ablation_train.py`**
- **Description**: Train and save models for each possible combinations of features. This creates new directories which are named after the set of features included (not ablated) in the model.

---

### **2. `ablation_predict.py`**
- **Description**: For all of the previous models trained and directories created by ablation_train.py, validate against a fixed testing set that has not been seen by the models in training. This outputs an autoregressive (prediction based on a previous prediction) prediction text document containing the predictions made by the model.

---